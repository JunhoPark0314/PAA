import torch
from copy import deepcopy
import torch.nn.functional as F
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
        self.score_voting = score_voting

    def forward_for_single_feature_map(self, box_cls, box_regression, iou_pred, anchors, targets=None):
        N, _, H, W = box_cls.shape
        A = box_regression.size(1) // 4
        C = box_cls.size(1) // A

        # put in the same format as anchors
        box_cls = permute_and_flatten(box_cls, N, A, C, H, W)
        box_cls = box_cls.sigmoid()

        box_regression = permute_and_flatten(box_regression, N, A, 4, H, W)
        box_regression = box_regression.reshape(N, -1, 4)

        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.reshape(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        # multiply the classification scores with IoU scores
        if iou_pred is not None:
            iou_pred = permute_and_flatten(iou_pred, N, A, 1, H, W)
            iou_pred = iou_pred.reshape(N, -1).sigmoid()
            box_cls = (box_cls * iou_pred[:, :, None]).sqrt()

        results = []
        min_distance = []
        iou_diff = []

        for per_box_cls_, per_box_regression, per_pre_nms_top_n, per_candidate_inds, per_anchors, per_im_trg \
                in zip(box_cls, box_regression, pre_nms_top_n, candidate_inds, anchors, targets):

            per_box_cls = per_box_cls_[per_candidate_inds]

            per_box_cls, top_k_indices = per_box_cls.topk(per_pre_nms_top_n, sorted=False)

            per_candidate_nonzeros = per_candidate_inds.nonzero()[top_k_indices, :]

            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1] + 1

            detections = self.box_coder.decode(
                per_box_regression[per_box_loc, :].view(-1, 4),
                per_anchors.bbox[per_box_loc, :].view(-1, 4)
            )

            whole_list = BoxList(self.box_coder.decode(
                per_box_regression.view(-1,4), per_anchors.bbox.view(-1,4)
            ), per_anchors.size, mode="xyxy")

            boxlist = BoxList(detections, per_anchors.size, mode="xyxy")
            boxlist.add_field("labels", per_class)
            boxlist.add_field("scores", per_box_cls)

            if len(per_im_trg) and len(per_box_loc):
                per_im_trg.bbox = per_im_trg.bbox.cuda()
                whole_iou, whole_idx = boxlist_iou(whole_list, per_im_trg).max(dim=0)
                det_iou = boxlist_iou(boxlist, per_im_trg)
                whole_idx_h, whole_idx_w = whole_idx // W, whole_idx % W
                per_box_loc_h, per_box_loc_w = per_box_loc // W, per_box_loc % W

                dist = (whole_idx_h - per_box_loc_h.unsqueeze(-1)) ** 2 + (whole_idx_w - per_box_loc_w.unsqueeze(-1)) ** 2
                dist = dist.float().sqrt()
                same_class = per_im_trg.extra_fields["labels"].cuda() == per_class.unsqueeze(-1)

                for i in range(len(per_im_trg)):
                    if same_class[:,i].sum() > 0 and whole_iou[i] > 0.5:
                        top_det_iou, top_idx = det_iou[same_class[:,i], i].topk(min(1,same_class[:,i].sum()))
                        dist_cond = torch.logical_and(dist[same_class[:,i], i][top_idx] > 0, dist[same_class[:,i], i][top_idx] <= 4)
                        if dist_cond.sum() > 0 and dist[same_class[:,i], i][top_idx][0] != 0 :
                            """
                            print("whole_iou: ", whole_iou[i])
                            print("top_det_iou:", top_det_iou[dist_cond])
                            print("dist:", dist[same_class[:,i], i][top_idx][dist_cond])
                            print("-------------------------------------------------")
                            """
                            min_distance.append(dist[same_class[:,i], i][top_idx][dist_cond])
                            iou_diff.append((whole_iou[i] - top_det_iou[dist_cond]))
                                
                 

                """
                boxlist = whole_list[whole_idx]
                boxlist.add_field("labels", per_im_trg.extra_fields["labels"].cuda())
                boxlist.add_field("scores", whole_iou.cuda())
                det_iou, det_idx = boxlist_iou(boxlist, per_im_trg).max(dim=0)
                """

            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)


        log = {
            "dist": torch.cat(min_distance) if len(min_distance) else torch.tensor([]).cuda(),
            "diff_iou": torch.cat(iou_diff) if len(iou_diff) else torch.tensor([]).cuda(),
        }
        
        return results, log


    def forward(self, box_cls, box_regression, iou_pred, anchors, targets=None):
        sampled_boxes = []
        anchors = list(zip(*anchors))
        if iou_pred is None:
            iou_pred = [None] * len(box_cls)
        log = {}
        for _, (o, b, i, a) in enumerate(zip(box_cls, box_regression, iou_pred, anchors)):
            sample_list, per_im_log = self.forward_for_single_feature_map(o, b, i, a, targets)
            sampled_boxes.append(
                sample_list
                #self.forward_for_single_feature_map(o, b, i, a, targets)
            )
            for k, v in per_im_log.items():
                log["{}_{}".format(k, 5 - _)] = v

        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        if not (self.bbox_aug_enabled and not self.bbox_aug_vote):
            boxlists = self.select_over_all_levels(boxlists)

        return boxlists, log

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
