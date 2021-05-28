import torch
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

    def forward_for_single_feature_map(self, box_cls, box_regression, iou_pred, anchors, targets):
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
        iou_per_im = []
        det_iou_per_im = []
        det_iou_per_box = []

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
            boxlist = BoxList(detections, per_anchors.size, mode="xyxy")
            boxlist.add_field("labels", per_class)
            boxlist.add_field("scores", per_box_cls)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)

            if per_im_trg is not None and len(per_im_trg) and len(boxlist):
                per_im_trg.bbox = per_im_trg.bbox.cuda()
                whole = BoxList(self.box_coder.decode(per_box_regression.reshape(4,-1).t(), per_anchors.bbox), per_anchors.size, mode="xyxy")
                whole_iou, whole_idx = boxlist_iou(per_im_trg, whole).max(dim=1)

                detection_iou = boxlist_iou(per_im_trg, boxlist)
                label_cond = (per_class.unsqueeze(-1) == per_im_trg.extra_fields["labels"].cuda()).t()

                detections_iou, detections_idx = (detection_iou * label_cond).max(dim=1)
                
                iou_per_im.append(whole_iou)
                det_iou_per_im.append(detections_iou)
                det_iou_per_box.append((detection_iou * label_cond).max(dim=0)[0])

                #result_per_im.extra_fields["scores"] = (detection_iou * label_cond).max(dim=0)[0]
            else:
                iou_per_im.append([])
                det_iou_per_im.append([])

        log_info = {
            "iou_per_im":iou_per_im,
            "det_iou_per_im": det_iou_per_im,
            "det_iou_per_box": det_iou_per_box,
        }

        return results, log_info

    def forward(self, box_cls, box_regression, iou_pred, anchors, targets):
        sampled_boxes = []
        anchors = list(zip(*anchors))
        if iou_pred is None:
            iou_pred = [None] * len(box_cls)

        log_whole = {}
        for _, (o, b, i, a) in enumerate(zip(box_cls, box_regression, iou_pred, anchors)):
            result, log_info_per_level = self.forward_for_single_feature_map(o, b, i, a, targets)
            sampled_boxes.append(
               result 
            )

            for k, v in log_info_per_level.items():
                if k not in log_whole:
                    log_whole[k] = {}
                for im, im_v in enumerate(v):
                    if im not in log_whole[k]:
                        log_whole[k][im] = []
                    log_whole[k][im].append(im_v)

        log_info_clear = {
            "max_nms" : []
        }

        for log_k, log_v in log_whole.items():
            log_info_clear[log_k] = []
            if log_k not in ["det_iou_per_box"]:
                for k, v in log_v.items():
                    v = [v_ele.unsqueeze(-1) for v_ele in v if len(v_ele)]
                    if len(v):
                        max_iou = torch.cat(v, dim=-1).max(dim=-1)[0]
                    else:
                        max_iou = []
                    log_info_clear[log_k].append(max_iou)
            else:
                for k, v in log_v.items():
                    v = [v_ele for v_ele in v if len(v_ele)]
                    if len(v):
                        max_iou = torch.cat(v, dim=-1)
                    else:
                        max_iou = []
                    log_info_clear[log_k].append(max_iou)


        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        if not (self.bbox_aug_enabled and not self.bbox_aug_vote):
            boxlists = self.select_over_all_levels(boxlists)

        if targets is not None:
            for target_per_im, per_im_box in zip(targets, boxlists):
                boxiou = boxlist_iou(target_per_im, per_im_box)
                labels = per_im_box.extra_fields["labels"]
                label_cond = (labels.unsqueeze(-1) == target_per_im.extra_fields["labels"].cuda()).t()
                boxiou *= label_cond

                if len(boxiou.flatten()):
                    log_info_clear["max_nms"].append(boxiou.max(dim=1)[0])
                else:
                    log_info_clear["max_nms"].append([])

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
