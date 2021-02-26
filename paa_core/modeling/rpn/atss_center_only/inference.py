import torch
from ..utils import concat_box_prediction_layers, permute_and_flatten
from paa_core.structures.bounding_box import BoxList
from paa_core.structures.boxlist_ops import cat_boxlist
from paa_core.structures.boxlist_ops import boxlist_ml_nms
from paa_core.structures.boxlist_ops import remove_small_boxes


class ATSS_CONLYPostProcessor(torch.nn.Module):
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
        bbox_aug_vote=False
    ):
        super(ATSS_CONLYPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes
        self.bbox_aug_enabled = bbox_aug_enabled
        self.box_coder = box_coder
        self.bbox_aug_vote = bbox_aug_vote

    def forward_for_single_feature_map(self, box_cls, box_regression, centerness, anchors):
        N, _, H, W = box_cls.shape
        A = box_regression.size(1) // 4
        C = box_cls.size(1) // A

        # put in the same format as anchors
        box_cls = permute_and_flatten(box_cls, N, A, C, H, W)
        box_cls = box_cls.sigmoid()

        box_regression = permute_and_flatten(box_regression, N, A, 4, H, W)
        box_regression = box_regression.reshape(N, -1, 4)

        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        centerness = permute_and_flatten(centerness, N, A, 1, H, W)
        centerness = centerness.reshape(N, -1).sigmoid()

        # multiply the classification scores with centerness scores
        box_cls = box_cls * centerness[:, :, None]

        results = []
        for per_box_cls, per_box_regression, per_pre_nms_top_n, per_candidate_inds, per_anchors \
                in zip(box_cls, box_regression, pre_nms_top_n, candidate_inds, anchors):

            per_box_cls = per_box_cls[per_candidate_inds]

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
            boxlist.add_field("scores", torch.sqrt(per_box_cls))
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)

        return results

    def forward_for_whole_feature_map(self, per_image_pred, anchors, targets, per_image_gt):
        N = len(anchors)
        thr_list = torch.arange(10) * 0.1 + 0.05
        pr_whole = []
        rc_whole = []
        disp_error_whole = []
        for im in range(N):
            per_image_pred_rank = per_image_pred["pred_rank"][im].sigmoid()
            per_image_pred_disp_vector = per_image_pred["pred_disp_vector"][im]
            #per_image_pred_disp_error= per_image_pred["per_disp_error"][im]

            per_image_gt_rank = per_image_gt["target_rank"][im]
            per_image_gt_disp_vector = per_image_gt["target_disp_vector"][im]
            #per_image_gt_disp_error = per_image_gt["target_disp_error"][im]
            
            if per_image_gt_rank is None:
                continue

            precision_list = []
            recall_list = []
            tp_disp_error_list = []
            for thr in thr_list:
                positive = per_image_pred_rank > thr
                true = per_image_gt_rank > thr

                tp = (true * positive).sum().item()
                fp = (~true * positive).sum().item()
                fn = (true * ~positive).sum().item()

                tp_disp_error_list.append(((per_image_gt_disp_vector[true * positive] - per_image_pred_disp_vector[true * positive]) ** 2).sum(dim=-1).sqrt().mean())
                precision_list.append(tp / (tp + fp + 1))
                recall_list.append(tp / (tp + fn + 1))

            #print('thr : ',thr_list.tolist())
            #print('precision : ',precision_list)
            #print('recall : ',recall_list)

            pr_whole.append(torch.tensor(precision_list).mean())
            rc_whole.append(torch.tensor(recall_list).mean())
            disp_error_whole.append(torch.tensor(tp_disp_error_list).mean())
        mean_pr = torch.stack(pr_whole).mean()
        mean_rc = torch.stack(rc_whole).mean()
        mean_disp_error = torch.stack(disp_error_whole).mean()

        return mean_pr, mean_rc, mean_disp_error

    def forward(self, per_image_pred, anchors, targets=None, per_image_gt=None):
        sampled_boxes = []
        anchors = [cat_boxlist(an) for an in anchors]

        mean_pr, mean_rc, mean_disp_error = self.forward_for_whole_feature_map(per_image_pred, anchors, targets, per_image_gt)

        """
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        if not (self.bbox_aug_enabled and not self.bbox_aug_vote):
            boxlists = self.select_over_all_levels(boxlists)
        """

        return mean_pr, mean_rc, mean_disp_error

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
            results.append(result)
        return results


def make_atss_conly_postprocessor(config, box_coder):

    box_selector = ATSS_CONLYPostProcessor(
        pre_nms_thresh=config.MODEL.ATSS_CONLY.INFERENCE_TH,
        pre_nms_top_n=config.MODEL.ATSS_CONLY.PRE_NMS_TOP_N,
        nms_thresh=config.MODEL.ATSS_CONLY.NMS_TH,
        fpn_post_nms_top_n=config.TEST.DETECTIONS_PER_IMG,
        min_size=0,
        num_classes=config.MODEL.ATSS_CONLY.NUM_CLASSES,
        bbox_aug_enabled=config.TEST.BBOX_AUG.ENABLED,
        box_coder=box_coder,
        bbox_aug_vote=config.TEST.BBOX_AUG.VOTE
    )

    return box_selector
