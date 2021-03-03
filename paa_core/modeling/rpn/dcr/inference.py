from copy import deepcopy
import torch
import torch.nn.functional as F
import numpy as np
from skimage.measure import label as sklabel
from ..utils import permute_and_flatten
from paa_core.structures.bounding_box import BoxList
from paa_core.structures.boxlist_ops import cat_boxlist
from paa_core.structures.boxlist_ops import boxlist_ml_nms
from paa_core.structures.boxlist_ops import remove_small_boxes
from paa_core.structures.boxlist_ops import boxlist_iou


class PAAPostProcessor(torch.nn.Module):
    def __init__(
        self,
        pre_nms_thresh,
        pre_nms_top_n,
        nms_thresh,
        fpn_post_nms_top_n,
        min_size,
        num_classes,
        box_coder,
        bbox_aug_enabled=False,
        bbox_aug_vote=False,
        score_voting=False,
    ):
        super(PAAPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes
        self.bbox_aug_enabled = bbox_aug_enabled
        self.box_coder = box_coder
        self.bbox_aug_vote = bbox_aug_vote
        #self.score_voting = score_voting
        self.score_voting = False

    def forward_for_single_feature_map(self, box_cls, box_regression, iou_pred, anchors, targets=None, cls_trg=None, reg_trg=None):
        N, _, H, W = box_cls.shape
        A = box_regression.size(1) // 4
        C = box_cls.size(1) // A

        # put in the same format as anchors
        box_cls = permute_and_flatten(box_cls, N, A, C, H, W)
        box_cls = box_cls.sigmoid()

        box_regression = permute_and_flatten(box_regression, N, A, 4, H, W)
        box_regression = box_regression.reshape(N, -1, 4)

        iou_pred = permute_and_flatten(iou_pred, N, A, 1, H, W)
        iou_pred = iou_pred.reshape(N, -1).sigmoid()

        cls_peak_inds = box_cls > self.pre_nms_thresh
        cls_pre_nms_top_n = cls_peak_inds.reshape(N, -1).sum(1)
        cls_pre_nms_top_n = cls_pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        iou_peak_inds = iou_pred > self.pre_nms_thresh
        iou_pre_nms_top_n = iou_peak_inds.reshape(N, -1).sum(1)
        iou_pre_nms_top_n = iou_pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        if cls_trg is not None:
            cls_trg = F.one_hot(cls_trg.long(), num_classes=81).squeeze(1)[:,:,:,1:].reshape(N, -1, C)
            reg_trg = reg_trg.squeeze(1).reshape(N, -1)

            cls_recall = []
            cls_precision = []

            iou_recall = []
            iou_precision = []

            miss_rate = []
        else:
            cls_trg = [None] * N
            reg_trg = [None] * N

        # multiply the classification scores with IoU scores
        box_cls = (iou_pred.unsqueeze(-1) * box_cls).sqrt()

        results = []

        for per_box_cls_, per_box_regression, per_iou_pred, \
            cls_per_pre_nms_top_n, cls_per_candidate_inds, iou_per_pre_nms_top_n, iou_per_candidate_inds, \
            per_anchors, per_cls_trg, per_reg_trg, per_im_trg \
                in zip(box_cls, box_regression, iou_pred, \
                    cls_pre_nms_top_n, cls_peak_inds, iou_pre_nms_top_n, iou_peak_inds,\
                    anchors, cls_trg, reg_trg, targets):
            
            if len(per_im_trg) == 0:
                result_per_im = deepcopy(per_im_trg)
                result_per_im.add_field(
                    "scores",torch.ones(len(result_per_im), device=result_per_im.bbox.device))
                del result_per_im.extra_fields['masks']
                result_per_im.extra_fields['labels'] = result_per_im.extra_fields['labels'].to(result_per_im.bbox.device)
                result_per_im = result_per_im.clip_to_image(remove_empty=False)
                result_per_im = remove_small_boxes(result_per_im, self.min_size)
                results.append(result_per_im)
                continue

            per_box_cls = per_box_cls_[cls_per_candidate_inds]
            per_box_cls, cls_top_k_indices = per_box_cls.topk(cls_per_pre_nms_top_n, sorted=False)

            cls_per_candidate_nonzeros = cls_per_candidate_inds.nonzero()[cls_top_k_indices, :]

            cls_detections = self.box_coder.decode(
                per_box_regression[cls_per_candidate_nonzeros[:,0], :].view(-1, 4),
                per_anchors.bbox[cls_per_candidate_nonzeros[:,0], :].view(-1, 4)
            )

            per_box_iou = per_iou_pred[iou_per_candidate_inds]
            per_box_iou, iou_top_k_indices = per_box_iou.topk(iou_per_pre_nms_top_n, sorted=False)

            iou_per_candidate_nonzeros = iou_per_candidate_inds.nonzero()[iou_top_k_indices, :]

            iou_detections = self.box_coder.decode(
                per_box_regression[iou_per_candidate_nonzeros, :].view(-1, 4),
                per_anchors.bbox[iou_per_candidate_nonzeros, :].view(-1, 4)
            )

            max_len = min([5, len(cls_detections), len(iou_detections)])

            if len(cls_detections) != 0:
                cls_detection_box = BoxList(cls_detections, per_anchors.size, mode="xyxy")
                cls_maxIoU, cls_idx = boxlist_iou(cls_detection_box , per_im_trg).topk(max_len, dim=0)

            if len(iou_detections) != 0:
                iou_detection_box = BoxList(iou_detections, per_anchors.size, mode="xyxy")
                iou_maxIoU, iou_idx = boxlist_iou(iou_detection_box , per_im_trg).topk(max_len, dim=0)

            if len(cls_detections) and len(iou_detections):
                miss = (iou_maxIoU > cls_maxIoU).squeeze(0).flatten()
                miss_rate.append(miss.sum()  / len(miss))

                cls_detection_box = cls_detection_box[cls_idx.flatten()][~miss]
                cls_detection_box.add_field(
                    "scores",cls_maxIoU.flatten()[~miss])
                cls_detection_box.extra_fields['labels'] =  per_im_trg.extra_fields['labels'].to(per_im_trg.bbox.device).unsqueeze(0).repeat(max_len,1).flatten()[~miss]

                iou_detection_box = iou_detection_box[iou_idx.flatten()][miss]
                iou_detection_box.add_field(
                    "scores",iou_maxIoU.flatten()[miss])
                iou_detection_box.extra_fields['labels'] = per_im_trg.extra_fields['labels'].to(per_im_trg.bbox.device).unsqueeze(0).repeat(max_len,1).flatten()[miss]

                result_per_im = cat_boxlist([cls_detection_box, iou_detection_box])
            else:
                result_per_im = deepcopy(per_im_trg)
                result_per_im.bbox = torch.zeros_like(result_per_im.bbox)
                result_per_im.add_field(
                    "scores",torch.zeros(len(result_per_im), device=result_per_im.bbox.device))
                del result_per_im.extra_fields['masks']
                result_per_im.extra_fields['labels'] = result_per_im.extra_fields['labels'].to(result_per_im.bbox.device)

            result_per_im = result_per_im.clip_to_image(remove_empty=False)
            result_per_im = remove_small_boxes(result_per_im, self.min_size)
            results.append(result_per_im)
        
        if targets is not None:
            log_info = {
                "miss_rate": miss_rate
            }

        return results , log_info

    def forward(self, preds_per_level, anchors, targets=None, iou_based_targets=None):
        sampled_boxes = []

        anchors = list(zip(*anchors))
        box_cls = preds_per_level["cls_logits"]
        box_regression = preds_per_level["box_regression"]
        iou_pred = preds_per_level["iou_pred"]
        if iou_based_targets is not None:
            cls_target = iou_based_targets["labels"]
            reg_target = iou_based_targets["matched_idx_all"]
        else:
            cls_target = [None] * 5
            reg_target = [None] * 5

        log_info = {}
        for _, (o, b, i, a, cls_trg, reg_trg) in enumerate(zip(box_cls, box_regression, iou_pred, anchors, cls_target, reg_target)):
            result, log_info_per_level = self.forward_for_single_feature_map(o, b, i, a, targets, cls_trg, reg_trg)
            sampled_boxes.append(result)
            log_info[_] = log_info_per_level

        log_info_clear = {
        }
        """
        acc_list = ['cls_iou_both', 'cls_only', 'iou_only', 'one_point']
        for key in acc_list:
            log_info_clear[key] = []
        """

        for lvl, v_dict in log_info.items():
            for k, v in v_dict.items():
                log_info_clear["{}_{}".format(k, lvl)] = torch.stack(v).mean().item()

        """
        num_targets = sum([len(trg) for trg in targets if trg is not None])
        for acc in acc_list:
            trg_acc = list(zip(*log_info_clear[acc]))
            trg_acc = sum([len(torch.cat(ci_both).unique()) for ci_both in trg_acc])
            log_info_clear[acc] = trg_acc / num_targets
        """

        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        if not (self.bbox_aug_enabled and not self.bbox_aug_vote):
            boxlists = self.select_over_all_levels(boxlists)

        return boxlists, log_info_clear

    # TODO very similar to filter_results from PostProcessor
    # but filter_results is per image
    # TODO Yang: solve this issue in the future. No good solution
    # right now.
    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            # multiclass nms
            result = boxlist_ml_nms(boxlists[i], self.nms_thresh)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.get_field("scores")
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            if self.score_voting:
                boxes_al = boxlists[i].bbox
                boxlist = boxlists[i]
                labels = boxlists[i].get_field("labels")
                scores = boxlists[i].get_field("scores")
                sigma = 0.025
                result_labels = result.get_field("labels")
                for j in range(1, self.num_classes):
                    inds = (labels == j).nonzero().view(-1)
                    scores_j = scores[inds]
                    boxes_j = boxes_al[inds, :].view(-1, 4)
                    boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
                    result_inds = (result_labels == j).nonzero().view(-1)
                    boxlist_for_class_nmsed = result[result_inds]
                    ious = boxlist_iou(boxlist_for_class_nmsed, boxlist_for_class)
                    voted_boxes = []
                    for bi in range(len(boxlist_for_class_nmsed)):
                        cur_ious = ious[bi]
                        pos_inds = (cur_ious > 0.01).nonzero().squeeze(1)
                        pos_ious = cur_ious[pos_inds]
                        pos_boxes = boxlist_for_class.bbox[pos_inds]
                        pos_scores = scores_j[pos_inds]
                        pis = (torch.exp(-(1-pos_ious)**2 / sigma) * pos_scores).unsqueeze(1)
                        voted_box = torch.sum(pos_boxes * pis, dim=0) / torch.sum(pis, dim=0)
                        voted_boxes.append(voted_box.unsqueeze(0))
                    if voted_boxes:
                        voted_boxes = torch.cat(voted_boxes, dim=0)
                        boxlist_for_class_nmsed_ = BoxList(
                            voted_boxes,
                            boxlist_for_class_nmsed.size,
                            mode="xyxy")
                        boxlist_for_class_nmsed_.add_field(
                            "scores",
                            boxlist_for_class_nmsed.get_field('scores'))
                        result.bbox[result_inds] = boxlist_for_class_nmsed_.bbox
            results.append(result)
        return results


def make_paa_postprocessor(config, box_coder):

    box_selector = PAAPostProcessor(
        pre_nms_thresh=config.MODEL.PAA.INFERENCE_TH,
        pre_nms_top_n=config.MODEL.PAA.PRE_NMS_TOP_N,
        nms_thresh=config.MODEL.PAA.NMS_TH,
        fpn_post_nms_top_n=config.TEST.DETECTIONS_PER_IMG,
        min_size=0,
        num_classes=config.MODEL.PAA.NUM_CLASSES,
        bbox_aug_enabled=config.TEST.BBOX_AUG.ENABLED,
        box_coder=box_coder,
        bbox_aug_vote=config.TEST.BBOX_AUG.VOTE,
        score_voting=config.MODEL.PAA.INFERENCE_SCORE_VOTING,
    )

    return box_selector
