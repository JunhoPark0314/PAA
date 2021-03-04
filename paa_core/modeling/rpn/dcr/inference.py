from copy import deepcopy
from paa_core.modeling.rpn.dcr.loss import get_hw_list, per_im_to_level
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

    def forward_for_single_feature_map(self, single_pred, pair_pred, anchors, targets=None):
        box_cls = single_pred["cls_logits"]
        box_regression = single_pred["box_regression"]
        iou_pred = single_pred["iou_pred"]

        pair_cls_peak = pair_pred["cls_peak"]
        pair_reg_peak = pair_pred["reg_peak"]
        pair_logit = pair_pred["pair_logit"]

        N, _, H, W = box_cls.shape
        A = box_regression.size(1) // 4
        C = box_cls.size(1) // A

        # put in the same format as anchors
        #box_cls = permute_and_flatten(box_cls, N, A, C, H, W)
        box_cls = box_cls.sigmoid()

        #box_regression = permute_and_flatten(box_regression, N, A, 4, H, W)
        #box_regression = box_regression.reshape(N, -1, 4)

        #iou_pred = permute_and_flatten(iou_pred, N, A, 1, H, W)
        iou_pred = iou_pred.sigmoid()

        cls_peak_inds = [pair_cls_peak[pair_cls_peak[:,0] == i] for i in range(N)]
        reg_peak_inds = [pair_reg_peak[pair_reg_peak[:,0] == i] for i in range(N)]
        pair_logit = [pair_logit[pair_cls_peak[:,0] == i].sigmoid() for i in range(N)]

        # multiply the classification scores with IoU scores

        results = []
        iou_per_im = []

        for per_box_cls_, per_box_regression, per_box_iou_, \
            per_pair_logit, per_cls_peak_inds, per_reg_peak_inds, per_anchors, per_im_trg \
                in zip(box_cls, box_regression, iou_pred, \
                    pair_logit, cls_peak_inds, reg_peak_inds, anchors, targets):
            
            positive = (per_pair_logit > self.pre_nms_thresh).flatten()

            cls_idx = per_cls_peak_inds.split(1, dim=1)
            per_box_cls = per_box_cls_[cls_idx[1], cls_idx[2], cls_idx[3]][positive]

            reg_idx = per_reg_peak_inds.split(1, dim=1)
            per_box_iou = per_box_iou_[reg_idx[1], reg_idx[2], reg_idx[3]][positive]

            detections = self.box_coder.decode(
                per_box_regression[:, reg_idx[2], reg_idx[3]].view(-1, 4),
                per_anchors.bbox.reshape(H, W, 4)[reg_idx[2], reg_idx[3], :].view(-1, 4)
            )[positive]

            per_pair_logit = per_pair_logit[positive].flatten()

            whole = BoxList(self.box_coder.decode(
                per_box_regression.reshape(4,-1).t(),
                per_anchors.bbox
            ), per_anchors.size, mode="xyxy")
            if len(per_im_trg):
                whole_iou, whole_idx = boxlist_iou(per_im_trg, whole).max(dim=1)

                #result_per_im = BoxList(whole[whole_idx], per_anchors.size, mode="xyxy")
                result_per_im = whole[whole_idx]
                result_per_im.add_field(
                    #"scores", (per_box_cls * per_box_iou * per_pair_logit).sqrt().flatten()
                    "scores", whole_iou
                )
                result_per_im.add_field(
                    "labels", per_im_trg.extra_fields["labels"]
                )
                iou_per_im.append(whole_iou)
            else:
                result_per_im = whole[:2]
                result_per_im.add_field(
                    #"scores", (per_box_cls * per_box_iou * per_pair_logit).sqrt().flatten()
                    "scores", torch.zeros(2, device=per_box_cls.device)
                )
                result_per_im.add_field(
                    "labels", torch.ones(2, device=per_box_cls.device)
                )

            result_per_im = result_per_im.clip_to_image(remove_empty=False)
            result_per_im = remove_small_boxes(result_per_im, self.min_size)
            results.append(result_per_im)
            """
            maxIoU , mi_idx = boxlist_iou(per_im_trg, result_per_im).topk(5,dim=1)
            mi_idx = torch.cat([mi_idx.unsqueeze(-1).cpu(), torch.arange(len(mi_idx)).unsqueeze(-1).repeat(1,5).unsqueeze(-1)], dim=-1).reshape(-1,2).cuda()
            cls_true = (per_im_trg.extra_fields["labels"] == (cls_idx[1]+1))
            max_true_positive = cls_true[mi_idx.split(1, dim=1)].reshape(-1,5)
            print(max_true_positive.any(dim=1).sum() / len(max_true_positive))
            """
        
        log_info = {
            "iou_per_im":iou_per_im
        }

        return results , log_info

    def forward(self, preds_per_level, pair_per_level, anchors, targets):
        sampled_boxes = []

        anchors = list(zip(*anchors))

        single_pred = []
        pair_pred = []

        for l in range(len(anchors)):
            single_pred.append({})
            pair_pred.append({})

        for k, v in preds_per_level.items():
            for i, level_v in enumerate(v):
                single_pred[i][k] = level_v 
        
        for k, v in pair_per_level.items():
            for i, level_v in enumerate(v):
                pair_pred[i][k] = level_v

        log_info = {}
        for _, (sp, pp, a) in enumerate(zip(single_pred, pair_pred, anchors)):
            result, log_info_per_level = self.forward_for_single_feature_map(sp, pp, a, targets)
            sampled_boxes.append(result)
            iou_per_im = log_info_per_level["iou_per_im"]
            for im, iou in enumerate(iou_per_im):
                if im not in log_info:
                    log_info[im] = []
                log_info[im].append(iou.unsqueeze(-1))

        log_info_clear = {
            "max_iou":[]
        }

        for k, v in log_info.items():
            max_iou = torch.cat(v, dim=-1).max(dim=-1)[0]
            log_info_clear["max_iou"].append(max_iou.mean())

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
