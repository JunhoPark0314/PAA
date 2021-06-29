from collections import defaultdict
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn
import os
from ..utils import concat_box_prediction_layers
from paa_core.layers import smooth_l1_loss
from paa_core.layers import SigmoidFocalLoss, sigmoid_focal_loss_jit
from paa_core.modeling.matcher import Matcher
from paa_core.structures.boxlist_ops import boxlist_iou
from paa_core.structures.boxlist_ops import cat_boxlist
import sklearn.mixture as skm

 
INF = 100000000


def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1


def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    import torch.distributed as dist
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


class SDLossComputation(object):

    def __init__(self, cfg, box_coder):
        self.cfg = cfg
        self.cls_loss_func = SigmoidFocalLoss(cfg.MODEL.SD.LOSS_GAMMA,
                                              cfg.MODEL.SD.LOSS_ALPHA)
        self.iou_pred_loss_func = nn.BCEWithLogitsLoss(reduction="sum")
        self.matcher = Matcher(cfg.MODEL.SD.IOU_THRESHOLD,
                               cfg.MODEL.SD.IOU_THRESHOLD,
                               True)
        self.box_coder = box_coder
        self.fpn_strides=[8, 16, 32, 64, 128]
        self.reg_loss_type = cfg.MODEL.SD.REG_LOSS_TYPE
        self.iou_loss_weight = cfg.MODEL.SD.IOU_LOSS_WEIGHT

    def GIoULoss(self, pred, target, anchor, weight=None):
        pred_boxes = self.box_coder.decode(pred.view(-1, 4), anchor.view(-1, 4))
        pred_x1 = pred_boxes[:, 0]
        pred_y1 = pred_boxes[:, 1]
        pred_x2 = pred_boxes[:, 2]
        pred_y2 = pred_boxes[:, 3]
        pred_x2 = torch.max(pred_x1, pred_x2)
        pred_y2 = torch.max(pred_y1, pred_y2)
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)

        gt_boxes = self.box_coder.decode(target.view(-1, 4), anchor.view(-1, 4))
        target_x1 = gt_boxes[:, 0]
        target_y1 = gt_boxes[:, 1]
        target_x2 = gt_boxes[:, 2]
        target_y2 = gt_boxes[:, 3]
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)

        x1_intersect = torch.max(pred_x1, target_x1)
        y1_intersect = torch.max(pred_y1, target_y1)
        x2_intersect = torch.min(pred_x2, target_x2)
        y2_intersect = torch.min(pred_y2, target_y2)
        area_intersect = torch.zeros(pred_x1.size()).to(pred)
        mask = (y2_intersect > y1_intersect) * (x2_intersect > x1_intersect)
        area_intersect[mask] = (x2_intersect[mask] - x1_intersect[mask]) * (y2_intersect[mask] - y1_intersect[mask])

        x1_enclosing = torch.min(pred_x1, target_x1)
        y1_enclosing = torch.min(pred_y1, target_y1)
        x2_enclosing = torch.max(pred_x2, target_x2)
        y2_enclosing = torch.max(pred_y2, target_y2)
        area_enclosing = (x2_enclosing - x1_enclosing) * (y2_enclosing - y1_enclosing) + 1e-7

        area_union = pred_area + target_area - area_intersect + 1e-7
        ious = area_intersect / area_union
        gious = ious - (area_enclosing - area_union) / area_enclosing

        losses = 1 - gious

        if weight is not None and weight.sum() > 0:
            return (losses * weight)

        assert losses.numel() != 0
        return losses

    def prepare_iou_based_targets(self, targets, anchors):
        """Compute IoU-based targets"""

        cls_labels = []
        reg_targets = []
        matched_idx_all = []
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyxy"
            bboxes_per_im = targets_per_im.bbox
            labels_per_im = targets_per_im.get_field("labels")
            anchors_per_im = cat_boxlist(anchors[im_i])
            num_gt = bboxes_per_im.shape[0]

            match_quality_matrix = boxlist_iou(targets_per_im, anchors_per_im)
            matched_idxs = self.matcher(match_quality_matrix)
            targets_per_im = targets_per_im.copy_with_fields(['labels'])
            matched_targets = targets_per_im[matched_idxs.clamp(min=0)]

            cls_labels_per_im = matched_targets.get_field("labels")
            cls_labels_per_im = cls_labels_per_im.to(dtype=torch.float32)

            # Background (negative examples)
            bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            cls_labels_per_im[bg_indices] = 0

            # discard indices that are between thresholds
            inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            cls_labels_per_im[inds_to_discard] = -1

            matched_gts = matched_targets.bbox
            matched_idx_all.append(matched_idxs.view(1, -1))

            reg_targets_per_im = self.box_coder.encode(matched_gts, anchors_per_im.bbox)
            cls_labels.append(cls_labels_per_im)
            reg_targets.append(reg_targets_per_im)

        return cls_labels, reg_targets, matched_idx_all

    def prepare_candidate_idx(self, anchors, num_anchors_per_level, labels_all_per_im, matched_idx_all_per_im, num_gt):
        # select candidates based on IoUs between anchors and GTs
        candidate_idxs = []
        for gt in range(num_gt):
            candidate_idxs_per_gt = []
            star_idx = 0
            for level, anchors_per_level in enumerate(anchors):
                end_idx = star_idx + num_anchors_per_level[level]
                labels_per_level = labels_all_per_im[star_idx:end_idx]
                matched_idx_per_level = matched_idx_all_per_im[star_idx:end_idx]
                match_idx = torch.nonzero(
                    (matched_idx_per_level == gt) & (labels_per_level > 0),
                    as_tuple=False
                )[:, 0]
                if match_idx.numel() > 0:
                    topk_idxs_per_level_per_gt = match_idx
                    candidate_idxs_per_gt.append(topk_idxs_per_level_per_gt + star_idx)
                star_idx = end_idx
            if candidate_idxs_per_gt:
                candidate_idxs.append(torch.cat(candidate_idxs_per_gt))
            else:
                candidate_idxs.append(None)

        return candidate_idxs
    
    def divide_clean_and_noise(self, candidate_idxs, loss_all_per_im, labels_per_im, bboxes_per_im, matched_gts, cls_labels_per_im, num_gt, batch_wise_id):
        per_gt_label = {}

        for gt in range(num_gt):
            curr_gt_label = {}
            if candidate_idxs[gt] is not None:
                if candidate_idxs[gt].numel() > 1:
                    candidate_loss = loss_all_per_im[candidate_idxs[gt]]
                    candidate_loss, inds = candidate_loss.sort()
                    components = torch.zeros_like(inds)
                    components[self.cfg.MODEL.SD.TOPK:] = 1
                    fgs = components == 0
                    bgs = components == 1
                    if torch.nonzero(fgs, as_tuple=False).numel() > 0:
                        is_pos = inds[fgs]
                        is_grey = inds[bgs]
                    else:
                        # just treat all samples as positive for high recall.
                        is_pos = inds
                        is_grey = None
                else:
                    is_pos = 0
                    is_grey = None
                if is_grey is not None:
                    grey_idx = candidate_idxs[gt][is_grey]
                    cls_labels_per_im[grey_idx] = -1
                    curr_gt_label["noise_idx"] = grey_idx + batch_wise_id

                pos_idx = candidate_idxs[gt][is_pos]
                cls_labels_per_im[pos_idx] = labels_per_im[gt].view(-1, 1)
                matched_gts[pos_idx] = bboxes_per_im[gt].view(-1, 4)
                curr_gt_label["clean_idx"] = pos_idx + batch_wise_id
                curr_gt_label["cls_label"] = labels_per_im[gt].item()
                curr_gt_label["reg_label"] = bboxes_per_im[gt]

            per_gt_label[gt] = curr_gt_label

        return per_gt_label

    def compute_clean_target(self, targets, anchors, labels_all, loss_all, matched_idx_all):
        """
        Args:
            targets (batch_size): list of BoxLists for GT bboxes
            anchors (batch_size, feature_lvls): anchor boxes per feature level
            labels_all (batch_size x num_anchors): assigned labels
            loss_all (batch_size x num_anchors): calculated loss
            matched_idx_all (batch_size x num_anchors): best-matched GG bbox indexes
        """
        device = loss_all.device
        cls_labels = []
        reg_targets = []
        per_image_target = {}
        batch_wise_id = 0

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyxy"
            bboxes_per_im = targets_per_im.bbox
            labels_per_im = targets_per_im.get_field("labels")
            anchors_per_im = cat_boxlist(anchors[im_i])
            labels_all_per_im = labels_all[im_i]
            loss_all_per_im = loss_all[im_i]
            matched_idx_all_per_im = matched_idx_all[im_i]
            assert labels_all_per_im.shape == matched_idx_all_per_im.shape
            num_gt = bboxes_per_im.shape[0]

            num_anchors_per_level = [len(anchors_per_level.bbox)
                for anchors_per_level in anchors[im_i]]

            candidate_idxs = self.prepare_candidate_idx(anchors[im_i], num_anchors_per_level, 
                                    labels_all_per_im, matched_idx_all_per_im, num_gt)

            # fit 2-mode GMM per GT box
            n_labels = anchors_per_im.bbox.shape[0]
            cls_labels_per_im = torch.zeros(n_labels, dtype=torch.long).to(device)
            matched_gts = torch.zeros_like(anchors_per_im.bbox)
            fg_inds = matched_idx_all_per_im >= 0
            matched_gts[fg_inds] = bboxes_per_im[matched_idx_all_per_im[fg_inds]]

            per_gt_labels = self.divide_clean_and_noise(candidate_idxs, loss_all_per_im, labels_per_im, bboxes_per_im, 
                matched_gts, cls_labels_per_im, num_gt, batch_wise_id)
            batch_wise_id += len(matched_idx_all_per_im)

            reg_targets_per_im = self.box_coder.encode(matched_gts, anchors_per_im.bbox)
            cls_labels.append(cls_labels_per_im)
            reg_targets.append(reg_targets_per_im)
            per_image_target[im_i] = per_gt_labels

        return cls_labels, reg_targets, per_image_target

    def compute_reg_loss(
        self, regression_targets, box_regression, all_anchors, labels, weights
    ):
        if 'iou' in self.reg_loss_type:
            reg_loss = self.GIoULoss(box_regression,
                                     regression_targets,
                                     all_anchors,
                                     weight=weights)
        elif self.reg_loss_type == 'smoothl1':
            reg_loss = smooth_l1_loss(box_regression,
                                      regression_targets,
                                      beta=self.bbox_reg_beta,
                                      size_average=False,
                                      sum=False)
            if weights is not None:
                reg_loss *= weights
        else:
            raise NotImplementedError
        return reg_loss[labels > 0].view(-1)

    def compute_ious(self, boxes1, boxes2):
        area1 = (boxes1[:, 2] - boxes1[:, 0] + 1) * (boxes1[:, 3] - boxes1[:, 1] + 1)
        area2 = (boxes2[:, 2] - boxes2[:, 0] + 1) * (boxes2[:, 3] - boxes2[:, 1] + 1)
        lt = torch.max(boxes1[:, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
        wh = (rb - lt + 1).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        return inter / (area1 + area2 - inter)

    def clean_loss(self, box_cls, box_regression, iou_pred, targets, anchors, locations):

        # get IoU-based anchor assignment first to compute anchor scores
        (iou_based_labels,
         iou_based_reg_targets,
         matched_idx_all) = self.prepare_iou_based_targets(targets, anchors)
        matched_idx_all = torch.cat(matched_idx_all, dim=0)

        N = len(iou_based_labels)
        iou_based_labels_flatten = torch.cat(iou_based_labels, dim=0).int()
        iou_based_reg_targets_flatten = torch.cat(iou_based_reg_targets, dim=0)
        box_cls_flatten, box_regression_flatten = concat_box_prediction_layers(
            box_cls, box_regression)
        anchors_flatten = torch.cat([cat_boxlist(anchors_per_image).bbox
            for anchors_per_image in anchors], dim=0)
        if iou_pred is not None:
            iou_pred_flatten = [ip.permute(0, 2, 3, 1).reshape(N, -1, 1) for ip in iou_pred]
            iou_pred_flatten = torch.cat(iou_pred_flatten, dim=1).reshape(-1)

        per_batch_pred = {
            "box_cls_flatten": box_cls_flatten,
            "box_regression_flatten": box_regression_flatten,
            "iou_pred_flatten": iou_pred_flatten,
            "anchors_flatten": anchors_flatten,
        }

        pos_inds = torch.nonzero(iou_based_labels_flatten > 0, as_tuple=False).squeeze(1)

        if pos_inds.numel() > 0:
            n_anchors = box_regression[0].shape[1] // 4
            n_loss_per_box = 1 if 'iou' in self.reg_loss_type else 4

            # compute anchor scores (losses) for all anchors
            iou_based_cls_loss = self.cls_loss_func(box_cls_flatten.detach(),
                                                    iou_based_labels_flatten,
                                                    sum=False)
            iou_based_reg_loss = self.compute_reg_loss(iou_based_reg_targets_flatten,
                                                       box_regression_flatten.detach(),
                                                       anchors_flatten,
                                                       iou_based_labels_flatten,
                                                       weights=None)
            iou_based_reg_loss_full = torch.full((iou_based_cls_loss.shape[0],),
                                                  fill_value=INF,
                                                  device=iou_based_cls_loss.device,
                                                  dtype=iou_based_cls_loss.dtype)
            iou_based_reg_loss_full[pos_inds] = iou_based_reg_loss.view(-1, n_loss_per_box).mean(1)
            combined_loss = iou_based_cls_loss.sum(dim=1) + iou_based_reg_loss_full
            assert not torch.isnan(combined_loss).any()

            # compute labels and targets using SD
            labels, reg_targets, per_image_target = self.compute_clean_target(
                targets,
                anchors,
                iou_based_labels_flatten.view(N, -1),
                combined_loss.view(N, -1),
                matched_idx_all)

            num_gpus = get_num_gpus()
            labels_flatten = torch.cat(labels, dim=0).int()
            reg_targets_flatten = torch.cat(reg_targets, dim=0)
            pos_inds = torch.nonzero(labels_flatten > 0, as_tuple=False).squeeze(1)
            total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
            num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)

            box_regression_flatten = box_regression_flatten[pos_inds]
            reg_targets_flatten = reg_targets_flatten[pos_inds]
            anchors_flatten = anchors_flatten[pos_inds]

            if iou_pred is not None:
                # compute iou prediction targets
                iou_pred_flatten = iou_pred_flatten[pos_inds]
                gt_boxes = self.box_coder.decode(reg_targets_flatten, anchors_flatten)
                boxes = self.box_coder.decode(box_regression_flatten, anchors_flatten).detach()
                ious = self.compute_ious(gt_boxes, boxes)

                # compute iou losses
                iou_pred_loss = self.iou_pred_loss_func(
                    iou_pred_flatten, ious) / num_pos_avg_per_gpu * self.iou_loss_weight
                sum_ious_targets_avg_per_gpu = reduce_sum(ious.sum()).item() / float(num_gpus)

                # set regression loss weights to ious between predicted boxes and GTs
                reg_loss_weight = ious
            else:
                reg_loss_weight = None

            reg_loss = self.compute_reg_loss(reg_targets_flatten,
                                             box_regression_flatten,
                                             anchors_flatten,
                                             labels_flatten[pos_inds],
                                             weights=reg_loss_weight)
            cls_loss = self.cls_loss_func(box_cls_flatten, labels_flatten.int(), sum=False)
        else:
            reg_loss = box_regression_flatten.sum()
            per_image_target = {}

        reg_norm = sum_ious_targets_avg_per_gpu if iou_pred is not None else num_pos_avg_per_gpu
        res = [cls_loss.sum() / num_pos_avg_per_gpu,
               reg_loss.sum() / reg_norm * self.cfg.MODEL.SD.REG_LOSS_WEIGHT]
        if iou_pred is not None:
            res.append(iou_pred_loss)
        return res, per_image_target, per_batch_pred
    
    def prepare_noise_pred_emb(self, pl_targets, box_emb, cls_emb, anchors_flatten, per_batch_pred):
        noise_idx = defaultdict(list)
        noise_pred = defaultdict(list)
        box_cls_flatten = per_batch_pred["box_cls_flatten"]
        box_reg_flatten = per_batch_pred["box_regression_flatten"]
        box_iou_flatten = per_batch_pred["iou_pred_flatten"]
        
        for im_i, per_image_target in pl_targets.items():
            for gt_id, target in per_image_target.items():
                if len(target) == 0 or "noise_idx" not in target:
                    continue
                box_emb[target["noise_idx"]] = target["reg_label"]
                box_emb[target["noise_idx"]] = self.box_coder.encode(box_emb[target["noise_idx"]],
                                                        anchors_flatten[target["noise_idx"]])
                cls_emb[target["noise_idx"]] = target["cls_label"]

                noise_pred["cls"].append(box_cls_flatten[target["noise_idx"]])
                noise_pred["reg"].append(box_reg_flatten[target["noise_idx"]])
                noise_pred["iou"].append(box_iou_flatten[target["noise_idx"]])
                noise_idx[im_i].append(target["noise_idx"])

        for k in list(noise_pred.keys()):
            noise_pred[k] = torch.cat(noise_pred[k], dim=0)

        noise_idx = torch.cat([idx for per_im_idx in list(noise_idx.values()) for idx in per_im_idx])
        return noise_idx, noise_pred

    def pl_loss(self, pl_module, per_image_target, anchors, locations, **per_batch_pred):
        anchors_flatten = per_batch_pred["anchors_flatten"]
        box_emb = torch.zeros(len(anchors_flatten), 4).cuda()
        cls_emb = torch.zeros(len(anchors_flatten), dtype=torch.long)
        noise_idx, noise_pred = self.prepare_noise_pred_emb(per_image_target, box_emb, cls_emb, anchors_flatten, per_batch_pred)
        pl_module_info = {
            "noise_idx": noise_idx,
            "box_emb": box_emb,
            "cls_emb": cls_emb,
        }

        noise_anchor = anchors_flatten[noise_idx,:]
        noise_gt_box = box_emb[noise_idx, :]
        noise_gt_cls = cls_emb[noise_idx]

        cls_pl_flatten, box_pl_flatten, iou_pl_flatten = pl_module(pl_module_info)

        gt_boxes = self.box_coder.decode(noise_gt_box, noise_anchor)
        boxes = self.box_coder.decode(noise_pred["reg"], noise_anchor).detach()
        ious = self.compute_ious(gt_boxes, boxes)

        num_gpus = get_num_gpus()
        total_num_pos = reduce_sum(noise_idx.new_tensor([noise_idx.numel()])).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)
        
        # compute iou losses
        iou_pred_loss = self.iou_pred_loss_func(noise_pred["iou"], ious * iou_pl_flatten.sigmoid())
        iou_pred_loss = iou_pred_loss / num_pos_avg_per_gpu * self.iou_loss_weight
        sum_ious_targets_avg_per_gpu = reduce_sum(ious.sum()).item() / float(num_gpus)

        # set regression loss weights to ious between predicted boxes and GTs
        reg_loss_weight = ious

        reg_loss = self.compute_reg_loss(box_pl_flatten,
                                            noise_pred["reg"],
                                            noise_anchor,
                                            noise_gt_cls,
                                            weights=reg_loss_weight)

        labels_flatten = torch.zeros_like(noise_pred["cls"])
        assert ((noise_gt_cls <= 80) * (noise_gt_cls >0)).all().item() and len(noise_gt_cls) == len(labels_flatten)
        labels_flatten[torch.arange(len(noise_gt_cls)),noise_gt_cls-1] = cls_pl_flatten.sigmoid()
        cls_loss = sigmoid_focal_loss_jit(noise_pred["cls"], labels_flatten, alpha=self.cls_loss_func.alpha, gamma=self.cls_loss_func.gamma, reduction="sum")

        reg_norm = sum_ious_targets_avg_per_gpu 
        res = [cls_loss.sum() / num_pos_avg_per_gpu,
               reg_loss.sum() / reg_norm * self.cfg.MODEL.SD.REG_LOSS_WEIGHT]
        res.append(iou_pred_loss)

        return res


def make_sd_loss_evaluator(cfg, box_coder):
    return SDLossComputation(cfg, box_coder)