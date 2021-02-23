import torch
from torch import nn
import os
from ..utils import concat_box_prediction_layers
from paa_core.layers import SigmoidFocalLoss
from paa_core.modeling.matcher import Matcher
from paa_core.structures.boxlist_ops import boxlist_iou
from paa_core.structures.boxlist_ops import cat_boxlist
from paa_core.structures.boxlist_ops import boxlist_center
from torch.distributions.normal import Normal

from typing import Optional
from torch.nn import functional as F

INF = 100000000


def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1


def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    import torch.distributed as dist
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.reduce_op.SUM)
    return tensor

def sigmoid_focal_loss_with_limit(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    eps = 1e-4

    assert (p <= 1.0).all().item() and (p >= 0.0).all().item()
    assert (targets <= 1.0).all().item() and (targets >= 0.0).all().item()
    ce_loss = -(targets * (p + eps).log() + (1 - targets) * (1 - p + eps).log())
    #ce_loss = F.binary_cross_entropy(p, targets, reduction="none")
    assert ce_loss.isfinite().all().item()
    ce_loss = ce_loss.clamp(min=0, max=5)
    p_t = p * targets + (1 - p) * (1 - targets)
    assert (p_t <= 1.0).all().item() and (p_t >= 0.0).all().item()
    loss = ce_loss * ((1 - p_t).pow(gamma))

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        assert (alpha_t <= 1.0).all().item() and (alpha_t >= 0.0).all().item()
        loss = alpha_t * loss

    if weight is not None:
        loss *= weight

    #print(loss.max().item(), loss.min().item())

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


sigmoid_focal_loss_jit = torch.jit.script(
    sigmoid_focal_loss_with_limit
)  # type: torch.jit.ScriptModule



class ATSS_CONLYLossComputation(object):

    def __init__(self, cfg, box_coder):
        self.cfg = cfg
        self.focal_loss_func = SigmoidFocalLoss(cfg.MODEL.ATSS_CONLY.LOSS_GAMMA, cfg.MODEL.ATSS_CONLY.LOSS_ALPHA)
        self.bce_loss_func = nn.BCEWithLogitsLoss(reduction="sum")
        self.focal_alpha = cfg.MODEL.ATSS_CONLY.LOSS_ALPHA
        self.focal_gamma = cfg.MODEL.ATSS_CONLY.LOSS_GAMMA
        #self.matcher = Matcher(cfg.MODEL.ATSS_CONLY.FG_IOU_THRESHOLD, cfg.MODEL.ATSS_CONLY.BG_IOU_THRESHOLD, True)
        #self.box_coder = box_coder

    def prepare_conly_targets(self, targets, anchors):
        """
        prepare targets for ["disp_vector", "disp_vector_error", "rank"]
        return with per image order
        """

        target_rank = []
        target_disp_vector = []
        target_disp_pos = []

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            if len(targets_per_im) == 0:
                target_rank.append(None)
                target_disp_vector.append(None)
                target_disp_pos.append(None)
                continue

            assert targets_per_im.mode == "xyxy"
            bboxes_per_im = targets_per_im.bbox
            #labels_per_im = targets_per_im.get_field("labels")
            anchors_per_im = cat_boxlist(anchors[im_i])
            num_gt = bboxes_per_im.shape[0]
            #compute ious between anchor and target
            ious = boxlist_iou(anchors_per_im, targets_per_im)
            #compute center displacement vector between anchor and target

            stride_per_level = [ious.new_full(size=(len(anchors_per_level.bbox),),fill_value=stride) for anchors_per_level, stride in zip(anchors[im_i], self.cfg.MODEL.ATSS_CONLY.ANCHOR_STRIDES)]
            anchors_center = boxlist_center(anchors_per_im).unsqueeze(1)
            gt_center = boxlist_center(targets_per_im)
            disp_vectors_per_im = ((anchors_center - gt_center) / torch.cat(stride_per_level).reshape(-1,1,1))
            disp_target = (disp_vectors_per_im ** 2).pow(2).sum(-1).min(-1)[1]

            assert (len(disp_target.unique() == len(targets_per_im)))

            rank_target_per_im = torch.zeros_like(ious)

            for ng in range(num_gt):
                curr_ious = ious[:,ng]
                target_anchor = curr_ious > min(0.1, curr_ious.topk(1000)[0].min().item())
                mean_iou, std_iou = curr_ious[target_anchor].mean(dim=0), curr_ious[target_anchor].std(dim=0)
                iou_distribution = Normal(mean_iou.unsqueeze(0), std_iou.unsqueeze(0))
                iou_cdf = iou_distribution.cdf(curr_ious[target_anchor])

                assert iou_cdf.isfinite().all().item()
                rank_target_per_im[target_anchor, ng] = iou_cdf

            max_rank_per_anchor, _ = rank_target_per_im.max(dim=1)
            disp_vectors_per_im = disp_vectors_per_im[torch.arange(len(disp_target)), disp_target, :]
            rank_target_pos = (max_rank_per_anchor >= 0.5)

            #assert(len(max_rank_id_per_anchor[rank_target_pos].unique()) == len(targets_per_im))

            target_rank.append(max_rank_per_anchor)
            target_disp_vector.append(disp_vectors_per_im)
            target_disp_pos.append(rank_target_pos)

        per_image_gt = {
            "target_rank": target_rank,
            "target_disp_vector": target_disp_vector,
            "target_disp_pos": target_disp_pos
        }

        return per_image_gt

    def per_level_to_image(self, N, per_level_pred):
        per_image_pred = {k:[] for k in list(per_level_pred.keys())}
        for ng in range(N):
            for k in list(per_level_pred.keys()):
                C = per_level_pred[k][0].shape[1]
                per_image_pred_per_k = torch.cat([pred_val[ng].reshape(C, -1).t() for pred_val in per_level_pred[k]])
                per_image_pred[k].append(per_image_pred_per_k.squeeze(-1))
        
        return per_image_pred

    def __call__(self, per_level_pred, targets, anchors):
        #prepare target for prediction
        per_image_gt = self.prepare_conly_targets(targets, anchors)

        N = len(targets)

        per_image_pred = self.per_level_to_image(N, per_level_pred) 

        # compute per object loss after here
        num_gpus = get_num_gpus()

        # rank loss

        per_image_gt_pos = torch.cat(per_image_gt["target_disp_pos"])


        per_image_pred_rank = torch.cat(per_image_pred["pred_rank"])
        per_image_gt_rank = torch.cat(per_image_gt["target_rank"])

        local_num_pos = (per_image_gt_rank > 0).sum()
        total_num_pos = reduce_sum(local_num_pos).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)

        per_image_gt_rank = per_image_gt_rank.clamp(min=0, max=1.0)
        #assert (per_image_gt_rank >= 0).all().item() and (per_image_gt_rank <= 1.0).all().item()
        #print(num_pos_avg_per_gpu / len(per_image_gt_rank))

        """
        loss_rank = sigmoid_focal_loss_jit(
            inputs=per_image_pred_rank,
            targets=per_image_gt_rank,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction="sum"
        ) / num_pos_avg_per_gpu
        """

        precision_list = []
        recall_list = []
        fscore = []
        threshold_list = torch.arange(10) * 0.1 + 0.05
        for th in threshold_list:
            positive = (per_image_pred_rank.sigmoid() > th).detach()
            true = (per_image_gt_rank > th).detach()
            tp = (positive * true).sum()
            fp = (positive * ~true).sum()
            fn = (~positive * true).sum()

            precision = tp / (tp + fp + 1)
            recall = tp / (tp + fn + 1)

            precision_list.append(precision)
            recall_list.append(recall)
            fscore.append( 2 / (1 / (precision + 1e-6) + 1 / (recall + 1e-6)))
        
        tid = torch.stack(fscore).argmin()
        precision = precision_list[tid]
        recall = recall_list[tid]

        alpha = (precision + 5e-7) / (precision + recall + 1e-6)
        alpha = alpha.clamp(min = 1e-2, max=1 - 1e-2)
        gamma = 2.1 - 2 * threshold_list[tid]

        loss_rank = sigmoid_focal_loss_with_limit(
            inputs=per_image_pred_rank,
            targets=per_image_gt_rank,
            alpha= alpha.item(),
            gamma=1,
            reduction="sum"
        ) / num_pos_avg_per_gpu

        assert loss_rank.isfinite().item()

        # disp_vector loss

        per_image_pred_disp_vector = torch.cat(per_image_pred["pred_disp_vector"])
        per_image_pred_disp_error = torch.cat(per_image_pred["pred_disp_error"])
        per_image_gt_disp_vector = torch.cat(per_image_gt["target_disp_vector"])

        loss_disp_vector = F.smooth_l1_loss(
            input=per_image_pred_disp_vector[per_image_gt_pos], 
            target=per_image_gt_disp_vector[per_image_gt_pos],
            reduction="none"
        )

        per_image_gt_disp_error = loss_disp_vector.detach().sum(dim=-1).sqrt()
        loss_disp_vector = (loss_disp_vector * per_image_gt_rank[per_image_gt_pos].detach().unsqueeze(1)).mean()
        assert loss_disp_vector.isfinite().item()

        loss_disp_error = F.smooth_l1_loss(
            input = per_image_pred_disp_error[per_image_gt_pos],
            target = per_image_gt_disp_error,
            reduction="none"
        )
        loss_disp_error = (loss_disp_error * per_image_gt_rank[per_image_gt_pos].detach()).sqrt().mean()

        loss = {
            "loss_rank": loss_rank,
            "loss_disp_vector": loss_disp_vector,
            #"loss_disp_error": loss_disp_error
        }

        log_info = {
            "precision": precision,
            "recall": recall,
            "alpha": alpha,
            "gamma": gamma,
        }

        return loss , log_info


def make_atss_conly_loss_evaluator(cfg, box_coder):
    loss_evaluator = ATSS_CONLYLossComputation(cfg, box_coder)
    return loss_evaluator
