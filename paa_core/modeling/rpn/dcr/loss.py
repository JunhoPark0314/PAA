from functools import reduce
from fvcore.nn.focal_loss import sigmoid_focal_loss_jit
import numpy as np
from numpy.lib import stride_tricks
import torch
from torch.nn import functional as F
from torch import nn
import os
from ..utils import concat_box_prediction_layers
from paa_core.layers import smooth_l1_loss
from paa_core.layers import SigmoidFocalLoss
from paa_core.modeling.matcher import Matcher
from paa_core.structures.boxlist_ops import boxlist_center, boxlist_iou
from paa_core.structures.boxlist_ops import cat_boxlist
import sklearn.mixture as skm
import torch.distributions.normal as tdn


INF = 100000000

def dice_loss(input, target, alpha, gamma, reduction="none"):
    input = input.sigmoid().contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1).float()

    p_t = (1 - target) * input + target
    #alpha_t = alpha * target + (1 - alpha) * (1 - target)
    focal_input = input * p_t ** gamma

    a = torch.sum(focal_input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    #print("tp: {}, pos: {}, true: {}, dice: {}".format(a.item(), b.item() , c.item(), d.item()))
    loss = 1 - d
    #loss *= alpha_t

    return  loss.sum()

def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1


def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    import torch.distributed as dist
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor

def per_level_to_im(per_level_list):

    per_im_list = []
    N, C, _, _ = per_level_list[0].shape
    for i in range(N): 
        curr_im_list = []
        for trg in per_level_list:
            curr_im_list.append(trg[i].permute(1,2,0).reshape(-1, C))
        curr_im_list = torch.cat(curr_im_list)
        per_im_list.append(curr_im_list)
    
    return per_im_list

def per_im_to_level(per_im_list, HW_list):

    per_level_list = []
    N = len(per_im_list)
    for idx, im_list in enumerate(per_im_list):
        if im_list is not None:
            C = im_list.shape[-1]
            break

    st = 0
    for (h, w) in HW_list:
        curr_level_list = []
        for trg in per_im_list:
            if trg is not None:
                curr_level_list.append(trg.squeeze(0)[st:st+h*w].reshape(1,h,w,-1).permute(0,3,1,2))
            else:
                curr_level_list.append(torch.zeros_like(per_im_list[idx].squeeze(0)[st:st+h*w].reshape(1,h,w,-1).permute(0,3,1,2)))
        curr_level_list = torch.cat(curr_level_list)
        per_level_list.append(curr_level_list)
        st += h*w
    return per_level_list

def get_hw_list(per_level_list):
    hw_list = []
    for trg in per_level_list:
        hw_list.append((trg.shape[2], trg.shape[3]))
    
    return hw_list

class DCRLossComputation(object):

    def __init__(self, cfg, box_coder, head):
        self.cfg = cfg
        self.cls_loss_func = SigmoidFocalLoss(cfg.MODEL.PAA.LOSS_GAMMA,
                                              cfg.MODEL.PAA.LOSS_ALPHA)
        #self.iou_pred_loss_func = nn.BCEWithLogitsLoss(reduction="sum")
        self.iou_pred_loss_func = sigmoid_focal_loss_jit
        #self.iou_pred_loss_func = dice_loss
        self.matcher = Matcher(cfg.MODEL.PAA.IOU_THRESHOLD,
                               cfg.MODEL.PAA.IOU_THRESHOLD,
                               True)
        self.box_coder = box_coder
        self.fpn_strides=[8, 16, 32, 64, 128]
        self.ppa_threshold=0.015
        self.head = head
        self.focal_alpha = cfg.MODEL.PAA.LOSS_ALPHA
        self.focal_gamma = cfg.MODEL.PAA.LOSS_GAMMA
        self.reg_loss_type = cfg.MODEL.PAA.REG_LOSS_TYPE
        self.iou_loss_weight = cfg.MODEL.PAA.IOU_LOSS_WEIGHT
        self.combined_loss = cfg.MODEL.PAA.USE_COMBINED_LOSS
        self.ppa_threshold = cfg.MODEL.PAA.PPA_THRESHOLD

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
        else:
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
            if len(targets_per_im) == 0:
                matched_idx_all.append(None)
                cls_labels.append(None)
                reg_targets.append(None)
                continue

            anchors_per_im = cat_boxlist(anchors[im_i])

            match_quality_matrix = boxlist_iou(targets_per_im, anchors_per_im)
            matched_idxs = self.matcher(match_quality_matrix)
            #assert len(targets_per_im) == len(matched_idxs.clamp(min=0).unique())
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

        targets_per_im = {
            "labels": cls_labels,
            "reg_targets": reg_targets,
            "matched_idx_all": matched_idx_all,
        }

        return targets_per_im

    def select_candidate_idxs_wi_target(self, num_gt, anchors_per_im, loss_per_im, num_anchors_per_level, labels_all_per_im, matched_idx_all_per_im, targets_per_im):
        # select candidates based on IoUs between anchors and GTs
        candidate_idxs = []
        iou_per_candidate = []
        for gt in range(num_gt):
            candidate_idxs_per_gt = []
            iou_per_candidate_per_gt = []

            A = torch.zeros(len(targets_per_im)).bool()
            A[gt] = 1
            curr_gt = targets_per_im[A]

            star_idx = 0
            for level, anchors_per_level in enumerate(anchors_per_im):
                end_idx = star_idx + num_anchors_per_level[level]
                loss_per_level = loss_per_im[star_idx:end_idx]
                labels_per_level = labels_all_per_im[star_idx:end_idx]
                matched_idx_per_level = matched_idx_all_per_im[star_idx:end_idx]
                match_idx = torch.nonzero(
                    (matched_idx_per_level == gt) & (labels_per_level > 0),
                    as_tuple=False
                )[:, 0]

                if match_idx.numel() > 0:
                    _, topk_idxs = loss_per_level[match_idx].topk(
                        min(match_idx.numel(), self.cfg.MODEL.PAA.TOPK), largest=False)
                    topk_idxs_per_level_per_gt = match_idx[topk_idxs]
                    #topk_idxs_per_level_per_gt = match_idx
                    candidate_idxs_per_gt.append(topk_idxs_per_level_per_gt + star_idx)

                    curr_anchor = anchors_per_level[topk_idxs_per_level_per_gt]
                    iou_anchor = boxlist_iou(curr_gt, curr_anchor)
                    iou_per_candidate_per_gt.append(iou_anchor[0])

                star_idx = end_idx
            if candidate_idxs_per_gt:
                candidate_idxs.append(torch.cat(candidate_idxs_per_gt))
                iou_per_candidate.append(torch.cat(iou_per_candidate_per_gt))
            else:
                candidate_idxs.append(None)
                iou_per_candidate.append(None)
        
        return candidate_idxs, iou_per_candidate


    def select_candidate_idxs(self, num_gt, anchors_per_im, loss_per_im, num_anchors_per_level, labels_all_per_im, matched_idx_all_per_im):
        # select candidates based on IoUs between anchors and GTs
        candidate_idxs = []
        for gt in range(num_gt):
            candidate_idxs_per_gt = []
            star_idx = 0
            for level, anchors_per_level in enumerate(anchors_per_im):
                end_idx = star_idx + num_anchors_per_level[level]
                loss_per_level = loss_per_im[star_idx:end_idx]
                labels_per_level = labels_all_per_im[star_idx:end_idx]
                matched_idx_per_level = matched_idx_all_per_im[star_idx:end_idx]
                match_idx = torch.nonzero(
                    (matched_idx_per_level == gt) & (labels_per_level > 0),
                    as_tuple=False
                )[:, 0]

                if match_idx.numel() > 0:
                    _, topk_idxs = loss_per_level[match_idx].topk(
                        min(match_idx.numel(), self.cfg.MODEL.PAA.TOPK), largest=False)
                    topk_idxs_per_level_per_gt = match_idx[topk_idxs]
                    candidate_idxs_per_gt.append(topk_idxs_per_level_per_gt + star_idx)

                star_idx = end_idx
            if candidate_idxs_per_gt:
                candidate_idxs.append(torch.cat(candidate_idxs_per_gt))
            else:
                candidate_idxs.append(None)
        
        return candidate_idxs

    def fit_FNP_per_GT(self, candidate_idxs, loss_all_per_im, num_gt, device):
        # fit 2-mode GMM per GT box

        is_grey = None
        pos_idxs = [None] * num_gt
        neg_idxs = [None] * num_gt
        grey_idxs = [None] * num_gt

        for gt in range(num_gt):
            if candidate_idxs[gt] is not None:
                if candidate_idxs[gt].numel() > 1:
                    candidate_loss = loss_all_per_im[candidate_idxs[gt]]
                    inds = torch.arange(len(candidate_loss)).to(candidate_loss.device)

                    if len(inds) > 10:
                        fgs = candidate_loss.topk(10, largest=False)[1]
                        bgs = (~(inds.unsqueeze(1) == fgs).any(dim=1)).nonzero().flatten()
                        is_pos = inds[fgs]
                        is_neg = inds[bgs]
                    else:
                        # just treat all samples as positive for high recall.
                        is_pos = inds
                        is_neg = is_grey = None
                else:
                    is_pos = 0
                    is_neg = None
                    is_grey = None
                if is_grey is not None:
                    grey_idx_per_gt = candidate_idxs[gt][is_grey]
                    grey_idxs[gt] = grey_idx_per_gt
                if is_neg is not None:
                    neg_idx_per_gt = candidate_idxs[gt][is_neg]
                    neg_idxs[gt] = neg_idx_per_gt

                pos_idx_per_gt = candidate_idxs[gt][is_pos]
                pos_idxs[gt] = (pos_idx_per_gt)

        idxs = {
            "pos": pos_idxs,
            "neg": neg_idxs,
            "grey": grey_idxs,
        }
        return idxs

    def fit_GMM_per_GT(self, candidate_idxs, loss_all_per_im, num_gt, device):
        # fit 2-mode GMM per GT box

        is_grey = None
        pos_idxs = [None] * num_gt
        neg_idxs = [None] * num_gt
        grey_idxs = [None] * num_gt
        for gt in range(num_gt):
            if candidate_idxs[gt] is not None:
                if candidate_idxs[gt].numel() > 1:
                    candidate_loss = loss_all_per_im[candidate_idxs[gt]]
                    candidate_loss, inds = candidate_loss.sort()
                    candidate_loss = candidate_loss.view(-1, 1).cpu().numpy()
                    min_loss, max_loss = candidate_loss.min(), candidate_loss.max()
                    means_init=[[min_loss], [max_loss]]
                    weights_init = [0.5, 0.5]
                    precisions_init=[[[1.0]], [[1.0]]]
                    gmm = skm.GaussianMixture(2,
                                                weights_init=weights_init,
                                                means_init=means_init,
                                                precisions_init=precisions_init)
                    gmm.fit(candidate_loss)
                    components = gmm.predict(candidate_loss)
                    scores = gmm.score_samples(candidate_loss)
                    components = torch.from_numpy(components).to(device)
                    scores = torch.from_numpy(scores).to(device)
                    fgs = components == 0
                    bgs = components == 1

                    if torch.nonzero(fgs, as_tuple=False).numel() > 0:
                        # Fig 3. (c)
                        fg_max_score = scores[fgs].max().item()
                        fg_max_idx = torch.nonzero(fgs & (scores == fg_max_score), as_tuple=False).min()
                        #is_neg = inds[fgs | bgs]
                        is_neg = inds[fg_max_idx+1:]
                        is_pos = inds[:fg_max_idx+1]
                    else:
                        # just treat all samples as positive for high recall.
                        is_pos = inds
                        is_neg = is_grey = None
                else:
                    is_pos = 0
                    is_neg = None
                    is_grey = None
                if is_grey is not None:
                    grey_idx_per_gt = candidate_idxs[gt][is_grey]
                    grey_idxs[gt] = grey_idx_per_gt
                if is_neg is not None:
                    neg_idx_per_gt = candidate_idxs[gt][is_neg]
                    neg_idxs[gt] = neg_idx_per_gt

                pos_idx_per_gt = candidate_idxs[gt][is_pos]
                pos_idxs[gt] = (pos_idx_per_gt)

        idxs = {
            "pos": pos_idxs,
            "neg": neg_idxs,
            "grey": grey_idxs,
        }
        return idxs
 
    def compute_dcr_positive(self, targets, anchors, labels_all, cls_loss, reg_loss, matched_idx_all):
        """
        Args:
            targets (batch_size): list of BoxLists for GT bboxes
            anchors (batch_size, feature_lvls): anchor boxes per feature level
            labels_all (batch_size x num_anchors): assigned labels
            loss_all (batch_size x num_anchors): calculated loss
            matched_idx_all (batch_size x num_anchors): best-matched GG bbox indexes
        """
        loss_all = cls_loss + reg_loss
        device = loss_all.device
        cls_labels = []
        reg_targets = []
        cls_pos_per_target = []
        reg_pos_per_target = []
        reg_whole_per_target = []
        num_object = 1

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

            num_anchors_per_level = [len(anchors_per_level.bbox)
                for anchors_per_level in anchors[im_i]]

            num_gt = bboxes_per_im.shape[0]

            if self.combined_loss:
                cls_candidate_idxs = self.select_candidate_idxs(num_gt, anchors[im_i], loss_all_per_im, 
                                        num_anchors_per_level, labels_all_per_im, matched_idx_all_per_im)
                """
                cls_candidate_idxs, iou_per_candidate = reg_candidate_idxs = self.select_candidate_idxs_wi_target(num_gt, anchors[im_i], loss_all_per_im, 
                                        num_anchors_per_level, labels_all_per_im, matched_idx_all_per_im, targets_per_im)
                """
            else:
                reg_candidate_idxs = self.select_candidate_idxs(num_gt, anchors[im_i], reg_loss[im_i], 
                                        num_anchors_per_level, labels_all_per_im, matched_idx_all_per_im)
                cls_candidate_idxs = self.select_candidate_idxs(num_gt, anchors[im_i], cls_loss[im_i], 
                                        num_anchors_per_level, labels_all_per_im, matched_idx_all_per_im)
                """
                reg_candidate_idxs, reg_iou_per_candidate = self.select_candidate_idxs_wi_target(num_gt, anchors[im_i], reg_loss[im_i], 
                                        num_anchors_per_level, labels_all_per_im, matched_idx_all_per_im, targets_per_im)
                cls_candidate_idxs, cls_iou_per_candidate = self.select_candidate_idxs_wi_target(num_gt, anchors[im_i], cls_loss[im_i], 
                                        num_anchors_per_level, labels_all_per_im, matched_idx_all_per_im, targets_per_im)
                """

            n_labels = anchors_per_im.bbox.shape[0]
            cls_labels_per_im = torch.zeros(n_labels, dtype=torch.long).to(device)
            cls_target_per_im = torch.zeros(n_labels, dtype=torch.long).to(device)
            reg_target_per_im = torch.zeros(n_labels, dtype=torch.long).to(device)
            reg_whole_per_im = torch.zeros(n_labels, dtype=torch.long).to(device)
            matched_gts = torch.zeros(n_labels, 4).to(device)

            if self.combined_loss:
                cls_idxs = reg_idxs = self.fit_GMM_per_GT(cls_candidate_idxs, loss_all_per_im, num_gt, device)
            else:
                cls_idxs = self.fit_FNP_per_GT(cls_candidate_idxs, cls_loss[im_i], num_gt, device)
                reg_idxs = self.fit_FNP_per_GT(reg_candidate_idxs, reg_loss[im_i], num_gt, device)

            for gt in range(num_gt):
                if cls_candidate_idxs[gt] is not None:
                    if cls_candidate_idxs[gt].numel() > 1:
                        reg_whole_per_gt = []
                        cls_pos_idx_per_gt = cls_idxs["pos"][gt]
                        reg_pos_idx_per_gt = reg_idxs["pos"][gt]
                        
                        if cls_idxs["grey"][gt] is not None:
                            cls_grey_idx_per_gt = cls_idxs["grey"][gt]
                            cls_labels_per_im[cls_grey_idx_per_gt] = -1
                        
                        if cls_idxs["neg"][gt] is not None:
                            cls_neg_idx_per_gt = cls_idxs["neg"][gt]
                            cls_labels_per_im[cls_neg_idx_per_gt] = 0

                        if reg_idxs["neg"][gt] is not None:
                            reg_neg_idx_per_gt = reg_idxs["neg"][gt]
                            matched_gts[reg_neg_idx_per_gt] = bboxes_per_im[gt].view(-1, 4)
                            reg_whole_per_gt.append(reg_neg_idx_per_gt)

                        cls_labels_per_im[cls_pos_idx_per_gt] = labels_per_im[gt].view(-1, 1)
                        matched_gts[reg_pos_idx_per_gt] = bboxes_per_im[gt].view(-1, 4)
                        cls_target_per_im[cls_pos_idx_per_gt] = gt + num_object
                        reg_target_per_im[reg_pos_idx_per_gt] = gt + num_object

                        reg_whole_per_gt.append(reg_pos_idx_per_gt)
                        reg_whole_per_im[torch.cat(reg_whole_per_gt)] = gt + num_object

            num_object += num_gt
            reg_targets_per_im = self.box_coder.encode(matched_gts, anchors_per_im.bbox)
            cls_labels.append(cls_labels_per_im)
            reg_targets.append(reg_targets_per_im)
            cls_pos_per_target.append(cls_target_per_im)
            reg_pos_per_target.append(reg_target_per_im)
            reg_whole_per_target.append(reg_whole_per_im)

        target = {
            "cls_labels": cls_labels,
            "reg_targets": reg_targets,
            "cls_pos_per_target": cls_pos_per_target,
            "reg_pos_per_target": reg_pos_per_target,
            "reg_whole_per_target": reg_whole_per_target,
        }

        return target
 
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

    def __call__(self, pred_per_level, targets, anchors, is_pa=True):
        loss = {}
        log_info = {}

        # get IoU-based anchor assignment first to compute anchor scores
        iou_based_targets = self.prepare_iou_based_targets(targets, anchors)

        sa_loss, sa_targets, sa_log_info = self.compute_single_anchor_loss(iou_based_targets, pred_per_level, anchors, targets)
        loss.update(sa_loss)
        log_info.update(sa_log_info)

        if is_pa:
            pa_loss, pa_log_info = self.compute_paired_anchor_loss(sa_targets, pred_per_level, anchors, targets)
            loss.update(pa_loss)
            log_info.update(pa_log_info)

        return loss, log_info

    def compute_single_anchor_loss(self, iou_based_targets, pred_per_level, anchors, targets):

        # prepare ingredients
        N = len(iou_based_targets["labels"])
        n_loss_per_box = 1 if 'iou' in self.reg_loss_type else 4

        matched_idx_all = torch.cat(iou_based_targets["matched_idx_all"], dim=0)

        iou_based_reg_targets_flatten = torch.cat(iou_based_targets["reg_targets"], dim=0)

        box_cls_flatten, box_regression_flatten = concat_box_prediction_layers(
            pred_per_level["cls_logits"], pred_per_level["box_regression"])

        anchors_flatten = torch.cat([cat_boxlist(anchors_per_image).bbox
            for anchors_per_image in anchors], dim=0)

        iou_pred_flatten = [ip.permute(0, 2, 3, 1).reshape(N, -1, 1) for ip in pred_per_level["iou_pred"]]
        iou_pred_flatten = torch.cat(iou_pred_flatten, dim=1).reshape(-1)

        iou_based_labels_flatten = torch.cat(iou_based_targets["labels"], dim=0).int()
        pos_inds = torch.nonzero(iou_based_labels_flatten > 0, as_tuple=False).squeeze(1)

        if pos_inds.numel() > 0:
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
                        
            dcr_targets = self.compute_dcr_positive(
                targets,
                anchors, 
                iou_based_labels_flatten.view(N, -1),
                iou_based_cls_loss.sum(dim=1).view(N, -1),
                iou_based_reg_loss_full.view(N, -1),
                matched_idx_all
            )

            num_gpus = get_num_gpus()
            cls_labels_flatten = torch.cat(dcr_targets["cls_labels"], dim=0).int()
            reg_labels_flatten = torch.cat(dcr_targets["reg_pos_per_target"], dim=0).int()
            reg_whole_labels_flatten = torch.cat(dcr_targets["reg_whole_per_target"], dim=0).int()
            reg_targets_flatten = torch.cat(dcr_targets["reg_targets"], dim=0)

            cls_pos_inds = torch.nonzero(cls_labels_flatten > 0, as_tuple=False).squeeze(1)
            total_cls_num_pos = reduce_sum(cls_pos_inds.new_tensor([cls_pos_inds.numel()])).item()
            num_cls_pos_avg_per_gpu = max(total_cls_num_pos / float(num_gpus), 1.0)

            reg_pos_inds = torch.nonzero(reg_labels_flatten > 0, as_tuple=False).squeeze(1)
            total_reg_num_pos = reduce_sum(reg_pos_inds.new_tensor([reg_pos_inds.numel()])).item()
            num_reg_pos_avg_per_gpu = max(total_reg_num_pos / float(num_gpus), 1.0)

            reg_whole_inds = torch.nonzero(reg_whole_labels_flatten > 0, as_tuple=False).squeeze(1)
            total_reg_num_whole = reduce_sum(reg_whole_inds.new_tensor([reg_whole_inds.numel()])).item()
            num_reg_whole_avg_per_gpu = max(total_reg_num_whole / float(num_gpus), 1.0)

            whole_box_regression_flatten = box_regression_flatten[reg_whole_inds]
            whole_reg_targets_flatten = reg_targets_flatten[reg_whole_inds]
            whole_anchors_flatten = anchors_flatten[reg_whole_inds]           

            whole_to_pos = (reg_pos_inds.unsqueeze(-1) == reg_whole_inds).any(dim=0)

            pos_box_regression_flatten = whole_box_regression_flatten[whole_to_pos]
            pos_reg_targets_flatten = whole_reg_targets_flatten[whole_to_pos]
            pos_anchors_flatten = whole_anchors_flatten[whole_to_pos]

            # compute iou prediction targets
            # TODO: change iou pred's target to iou rank between target object's candidates
            iou_pred_flatten = iou_pred_flatten
            whole_gt_boxes = self.box_coder.decode(whole_reg_targets_flatten, whole_anchors_flatten)
            whole_boxes = self.box_coder.decode(whole_box_regression_flatten, whole_anchors_flatten).detach()
            whole_ious = self.compute_ious(whole_gt_boxes, whole_boxes)
            iou_target = torch.zeros_like(iou_pred_flatten)
            iou_target[reg_whole_inds] = whole_ious

            # compute iou losses
            iou_pred_loss = self.iou_pred_loss_func(
                #iou_pred_flatten.unsqueeze(0), 
                #iou_target.unsqueeze(0),
                iou_pred_flatten, 
                iou_target,
                alpha= self.focal_alpha,
                gamma= self.focal_gamma,
                reduction="sum"
            ) 
            sum_ious_targets_avg_per_gpu = reduce_sum(whole_ious[whole_to_pos].sum()).item() / float(num_gpus)
            #num_reg_pos_avg_per_gpu += (iou_pred_flatten.sigmoid() > 0.4).sum().item()

            # set regression loss weights to ious between predicted boxes and GTs
            reg_loss_weight = whole_ious[whole_to_pos]

            reg_loss = self.compute_reg_loss(pos_reg_targets_flatten,
                                                pos_box_regression_flatten,
                                                pos_anchors_flatten,
                                                reg_labels_flatten[reg_pos_inds],
                                                weights=reg_loss_weight)
            cls_loss = self.cls_loss_func(box_cls_flatten, cls_labels_flatten.int(), sum=False)
        else:
            reg_loss = box_regression_flatten.sum()

        reg_norm = sum_ious_targets_avg_per_gpu
        loss = {
            "loss_cls": cls_loss.sum() / num_cls_pos_avg_per_gpu,
            "loss_reg": reg_loss.sum() / reg_norm * self.cfg.MODEL.PAA.REG_LOSS_WEIGHT,
            "loss_iou": iou_pred_loss / num_reg_whole_avg_per_gpu 
            #"loss_iou": iou_pred_loss
        }
        #loss["loss_iou"] *= (1 - loss["loss_reg"]).item()

        for k, v in loss.items():
            assert v.isfinite().item()

        log_info = {}

        log_info = {}
        cls_pos = (box_cls_flatten.sigmoid() > 0.3)

        cls_pos_score, cls_pos_inds = box_cls_flatten.flatten().topk(self.ppa_count)
        cls_true = torch.zeros_like(box_cls_flatten)
        cls_true[cls_labels_flatten!=0, (cls_labels_flatten[cls_labels_flatten!=0]-1).long()] = 1
        cls_tp = cls_true.flatten()[cls_pos_inds].sum() 
        log_info["cls_pr"] = (cls_tp / self.ppa_count)
        log_info["cls_rc"] = (cls_tp / cls_true.sum())

        iou_pos_score, iou_pos_inds = iou_pred_flatten.sigmoid().topk(self.ppa_count)
        iou_tp = (iou_target[iou_pos_inds] > 0.5).sum()

        log_info["iou_pr"] = iou_tp / self.ppa_count
        log_info["iou_rc"] = iou_tp / (iou_target > 0.5).sum()

        return loss, dcr_targets, log_info
    
    def compute_dcr_pair_positive(self, pred_with_pair_per_level ,pred_per_level, single_target):
        # 1. take patch where cls / reg score is topk-min(1000) per image in each level
        # 2. compute distance / score between patch and discard pair wich are too far from each other
        # 3. take 2000 most high score combination per level
        log_info = {}
        hw_list = get_hw_list(pred_per_level["cls_top_feature"])
                
        cls_pos_per_im = single_target["cls_pos_per_target"]
        reg_pos_per_im = single_target["reg_pos_per_target"]

        true_whole = []
        for cls_pos in cls_pos_per_im:
            true_whole.append(cls_pos.unique())
        true_whole = torch.cat(true_whole).clamp(min=1).unique()

        cls_pos_per_level = per_im_to_level(cls_pos_per_im, hw_list)
        reg_pos_per_level = per_im_to_level(reg_pos_per_im, hw_list)

        # 4. indicator of pair
        #    - cls target / reg target pair of same object : 1
        #    - cls target / reg target pair of different object : 0

        pair_logit_target_per_level = []
        tp_pair = []
        for cls_peak, reg_peak, cls_pos, reg_pos in zip(pred_with_pair_per_level["cls_peak"], pred_with_pair_per_level["reg_peak"], cls_pos_per_level, reg_pos_per_level):
            cls_target = cls_pos[cls_peak[:,0], :, cls_peak[:,2], cls_peak[:,3]]
            reg_target = reg_pos[reg_peak[:,0], :, reg_peak[:,2], reg_peak[:,3]]

            pair_target = torch.zeros_like(cls_target)
            pair_target[(cls_target == reg_target) * (cls_target != 0)] = 1
            pair_logit_target_per_level.append(pair_target)

            tp_per_level = cls_target[(cls_target == reg_target) * (cls_target != 0)]
            if len(tp_per_level):
                tp_pair.append(tp_per_level.unique())
 
        if len(tp_pair):
            tp_whole = torch.cat(tp_pair).unique()
            pr = len(tp_whole) / len(true_whole)
            log_info["object_precision"] = pr
        else:
            log_info["object_precision"] = 0
 
        return pair_logit_target_per_level, log_info
    
    def compute_paired_anchor_loss(self, single_target, pred_per_level, anchor, target):

        # ------------------------------------------------------------------
        # 1. take patch where cls / reg score is topk-min(1000) per image in each level
        # 2. compute distance / score between patch and discard pair wich are too far from each other
        # 3. take 2000 most high score combination per level
        # 4. indicator of pair
        #    - cls target / reg target pair of same object : 1
        #    - cls target / reg target pair of different object : 0
        # TODO:
        # 5. refine classification and regression with combined feature
        # 6. final output of paired anchor network
        #    - p(same object | anchor_cls, anchor_reg)
        #    - p(Class | anchor_cls, anchor_reg)_refine
        #    - p(Reg | anchor_cls, anchor_reg)_refine
        # ------------------------------------------------------------------
        pred_with_pair_per_level = self.head.forward_with_pair(pred_per_level, self.ppa_threshold)

        pair_logit_target_per_level, pa_log_info = self.compute_dcr_pair_positive(pred_with_pair_per_level, pred_per_level, single_target)
        
        num_gpus = get_num_gpus()
        pair_logit_target_flatten = torch.cat(pair_logit_target_per_level, dim=0).int()
        pair_pos_inds = torch.nonzero(pair_logit_target_flatten > 0, as_tuple=False).squeeze(1)
        total_pair_num_pos = reduce_sum(pair_pos_inds.new_tensor([pair_pos_inds.numel()])).item()
        num_pair_pos_avg_per_gpu = max(total_pair_num_pos / float(num_gpus), 1.0)
        pair_logit_flatten = torch.cat(pred_with_pair_per_level["pair_logit"])

        #pair_loss = self.cls_loss_func(pair_logit_flatten, pair_logit_target_flatten.int(), sum=True) / num_pair_pos_avg_per_gpu
        pair_loss = nn.BCEWithLogitsLoss(reduction="mean")(pair_logit_flatten, pair_logit_target_flatten.float())

        if len(pair_logit_flatten) == 0:
            pair_loss = None
            loss = {
            }
            log_info = {
            }
        else:
            loss = {
                "loss_pair": pair_loss
            }

            log_info = {
                "pair_tp": ((pair_logit_flatten.sigmoid() > 0.05) * (pair_logit_target_flatten == 1)).sum() / len(pair_logit_target_flatten),
                "pair_fp": ((pair_logit_flatten.sigmoid() > 0.05) * (pair_logit_target_flatten == 0)).sum() / len(pair_logit_target_flatten),
                "pair_fn": ((pair_logit_flatten.sigmoid() < 0.05) * (pair_logit_target_flatten == 1)).sum() / len(pair_logit_target_flatten),
            }

        log_info.update(pa_log_info)

        return loss, log_info



    def prepare_pair_based_target(self, peak_list_per_im):
        return

def make_dcr_loss_evaluator(cfg, box_coder, head):
    loss_evaluator = DCRLossComputation(cfg, box_coder, head)
    return loss_evaluator
