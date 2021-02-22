import torch
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
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n * 10)

        # multiply the classification scores with IoU scores
        if iou_pred is not None:
            iou_pred = permute_and_flatten(iou_pred, N, A, 1, H, W)
            iou_pred = iou_pred.reshape(N, -1).sigmoid()
            box_cls = (box_cls * iou_pred[:, :, None]).sqrt()

        results = []
        bbox_iou = []
        det_iou = []
        pred_iou = []
        if targets is None:
            targets = [None] * len(box_cls)
        for per_box_cls_, per_box_regression, per_iou_pred, per_pre_nms_top_n, per_candidate_inds, per_anchors, per_target \
                in zip(box_cls, box_regression, iou_pred, pre_nms_top_n, candidate_inds, anchors, targets):
            
            per_box_cls = per_box_cls_[per_candidate_inds]

            per_box_cls, top_k_indices = per_box_cls.topk(per_pre_nms_top_n, sorted=False)

            per_candidate_nonzeros = per_candidate_inds.nonzero()[top_k_indices, :]

            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1] + 1

            per_iou_loc, top_k_indices = per_iou_pred[per_iou_pred >= 0.5].topk(per_box_loc.unique().shape[0], sorted=False)
            per_iou_candidate = (per_iou_pred >= 0.5).nonzero()[top_k_indices]

            detections = self.box_coder.decode(
                per_box_regression[per_box_loc, :].view(-1, 4),
                per_anchors.bbox[per_box_loc, :].view(-1, 4)
            )

            iou_detections = self.box_coder.decode(
                per_box_regression[per_iou_candidate, :].view(-1,4),
                per_anchors.bbox[per_iou_candidate, :].view(-1,4)
            )

            boxlist = BoxList(detections, per_anchors.size, mode="xyxy")
            boxlist.add_field("labels", per_class)
            boxlist.add_field("scores", per_box_cls)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)

            if per_target is not None:
                per_location = per_anchors.bbox.view(-1,4)
                per_location = (per_location[:,:2] + per_location[:,2:]) / 2

                whole_detections = self.box_coder.decode(per_box_regression.view(-1,4), per_anchors.bbox.view(-1,4))
                whole_detections = BoxList(whole_detections, per_anchors.size, mode="xyxy")
                per_target.bbox = per_target.bbox.to(whole_detections.bbox.device)

                if len(per_target):
                    iou_target = boxlist_iou(whole_detections, per_target)
                    bbox_iou.append(iou_target.max(dim=0)[0])

                    iou_detection = boxlist_iou(boxlist, per_target)
                    if len(iou_detection) == 0:
                        iou_detection = torch.zeros_like(iou_target)
                    det_iou.append(iou_detection.max(dim=0)[0])

                    iou_pred_detection = boxlist_iou(BoxList(iou_detections, per_anchors.size, mode="xyxy"), per_target)
                    if len(iou_pred_detection) == 0:
                        iou_pred_detection = torch.zeros_like(iou_target)
                    pred_iou.append(iou_pred_detection.max(dim=0)[0])
                else:
                    bbox_iou.append(torch.tensor([0], device=per_box_regression.device))
                    det_iou.append(torch.tensor([0], device=per_box_regression.device))
                    pred_iou.append(torch.tensor([0], device=per_box_regression.device))

        return results , bbox_iou, det_iou, pred_iou

    def compute_centroid(self, pred_disp, anchor, shape, stride):
        pred_ctr = self.box_coder.decode_disp(pred_disp, anchor) / stride
        pred_ctr_np = pred_ctr.round().cpu().numpy().astype(np.int)
        #per_class_np = per_class.cpu().numpy().astype(np.int)
        
        label = np.zeros(shape, dtype=np.int)
        label[pred_ctr_np[:,0], pred_ctr_np[:,1]] = 1
        new_label = sklabel(label)
        instance_label = new_label[pred_ctr_np[:,0], pred_ctr_np[:,1]].reshape(-1,1)
        instance_label = torch.tensor(instance_label,device=pred_disp.device)

        per_cls_centroid = torch.cat([pred_ctr, instance_label], dim=1)

        cls_centroid = []
        for i in instance_label.unique():
            grouped_cls = per_cls_centroid[per_cls_centroid[:,2] == i]
            cls_centroid.append(grouped_cls.mean(dim=0).unsqueeze(0))
        cls_centroid = torch.cat(cls_centroid, dim=0) * stride

        return cls_centroid

    def forward_with_disp_for_single_feature_map(self, box_cls, box_regression, iou_pred, anchors, targets=None, disp_pred=None):
        N, _, H, W = box_cls.shape
        A = box_regression.size(1) // 4
        C = box_cls.size(1) // A

        # put in the same format as anchors
        box_cls = permute_and_flatten(box_cls, N, A, C, H, W)
        box_cls = box_cls.sigmoid()

        box_regression = permute_and_flatten(box_regression, N, A, 4, H, W)
        box_regression = box_regression.reshape(N, -1, 4)

        # multiply the classification scores with IoU scores
        iou_pred = permute_and_flatten(iou_pred, N, A, 1, H, W)
        iou_pred = iou_pred.reshape(N, -1).sigmoid()
        box_cls = (box_cls * iou_pred[:, :, None]).sqrt()

        #candidate_inds = iou_pred >= 0.5
        candidate_inds = iou_pred >= 0.5
        pre_nms_top_n = candidate_inds.reshape(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        results = []
        bbox_iou = []
        det_iou = []
        pred_iou = []
        if targets is None:
            targets = [None] * len(box_cls)
        for per_box_cls_, per_box_regression, per_iou_pred_, per_pre_nms_top_n, per_candidate_inds, per_anchors, per_target \
                in zip(box_cls, box_regression, iou_pred, pre_nms_top_n, candidate_inds, anchors, targets):
            
            per_iou_pred = per_iou_pred_[per_candidate_inds]

            per_iou_pred, top_k_indices = per_iou_pred.topk(per_pre_nms_top_n, sorted=False)

            per_candidate_nonzeros = per_candidate_inds.nonzero()[top_k_indices, :]

            per_box_loc = per_candidate_nonzeros
            #per_class = per_candidate_nonzeros[:, 1] + 1

            detections = self.box_coder.decode(
                per_box_regression[per_box_loc, :].view(-1, 4),
                per_anchors.bbox[per_box_loc, :].view(-1, 4)
            )

            iou_detections = self.box_coder.decode(
                per_box_regression[per_iou_pred_ > 0.5, :].view(-1,4),
                per_anchors.bbox[per_iou_pred_ > 0.5, :].view(-1,4)
            )

            boxlist = BoxList(detections, per_anchors.size, mode="xyxy")
            #boxlist.add_field("labels", per_class)
            #boxlist.add_field("scores", per_box_cls)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)

            if per_target is not None:
                per_location = per_anchors.bbox.view(-1,4)
                per_location = (per_location[:,:2] + per_location[:,2:]) / 2

                whole_detections = self.box_coder.decode(per_box_regression.view(-1,4), per_anchors.bbox.view(-1,4))
                whole_detections = BoxList(whole_detections, per_anchors.size, mode="xyxy")
                per_target.bbox = per_target.bbox.to(whole_detections.bbox.device)

                if len(per_target):
                    iou_target = boxlist_iou(whole_detections, per_target)
                    bbox_iou.append(iou_target.max(dim=0)[0])

                    iou_detection = boxlist_iou(boxlist, per_target)
                    if len(iou_detection) == 0:
                        iou_detection = torch.zeros_like(iou_target)
                    det_iou.append(iou_detection.max(dim=0)[0])

                    iou_pred_detection = boxlist_iou(BoxList(iou_detections, per_anchors.size, mode="xyxy"), per_target)
                    if len(iou_pred_detection) == 0:
                        iou_pred_detection = torch.zeros_like(iou_target)
                    pred_iou.append(iou_pred_detection.max(dim=0)[0])
                else:
                    bbox_iou.append(torch.tensor([0], device=per_box_regression.device))
                    det_iou.append(torch.tensor([0], device=per_box_regression.device))
                    pred_iou.append(torch.tensor([0], device=per_box_regression.device))

        return results , bbox_iou, det_iou, pred_iou



    def forward(self, box_cls, box_regression, iou_pred, anchors, targets=None):
        sampled_boxes = []
        bbox_iou = []
        det_iou = []
        pred_iou = []
        bbox_iou_sup = []
        det_iou_sup = []
        pred_iou_sup = []

        anchors = list(zip(*anchors))
        if iou_pred is None:
            iou_pred = [None] * len(box_cls)
        for _, (o, b, i, a) in enumerate(zip(box_cls, box_regression, iou_pred, anchors)):
            result, bbox_iou_per_level, det_iou_per_level, pred_iou_per_level = self.forward_for_single_feature_map(o, b, i, a, targets)
            #result, bbox_iou_per_level, det_iou_per_level, pred_iou_per_level = self.forward_with_disp_for_single_feature_map(o, b, i, a, targets)
            sampled_boxes.append(result)
            bbox_iou.append(bbox_iou_per_level)
            det_iou.append(det_iou_per_level)
            pred_iou.append(pred_iou_per_level)

        if targets is not None:
            for i in range(box_cls[0].shape[0]):
                per_gt_iou = []
                per_gt_det = []
                per_gt_pred = []
                for j in range(len(bbox_iou)):
                    per_gt_iou.append(bbox_iou[j][i].unsqueeze(0))
                    per_gt_det.append(det_iou[j][i].unsqueeze(0))
                    per_gt_pred.append(pred_iou[j][i].unsqueeze(0))
                per_gt_max = torch.stack(per_gt_iou).max(dim=0)[0]
                per_det_max = torch.stack(per_gt_det).max(dim=0)[0]
                per_pred_max = torch.stack(per_gt_pred).max(dim=0)[0]
                bbox_iou_sup.append(per_gt_max.squeeze(0))
                det_iou_sup.append(per_det_max.squeeze(0))
                pred_iou_sup.append(per_pred_max.squeeze(0))

        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        if not (self.bbox_aug_enabled and not self.bbox_aug_vote):
            boxlists = self.select_over_all_levels(boxlists)

        return boxlists, bbox_iou_sup, det_iou_sup, pred_iou_sup

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
