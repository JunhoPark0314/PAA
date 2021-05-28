import numpy as np
import torch
from torch.nn import functional as F
from torch import nn
import os
from ..utils import concat_box_prediction_layers
from paa_core.layers import smooth_l1_loss
from paa_core.layers import SigmoidFocalLoss
from paa_core.modeling.matcher import Matcher
from paa_core.structures.boxlist_ops import boxlist_iou
from paa_core.structures.boxlist_ops import cat_boxlist
from paa_core.structures.bounding_box import BoxList
import sklearn.mixture as skm
import math
from paa_core.modeling.rpn.dcr.loss import per_im_to_level, per_level_to_im

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

class ProxyLossComputation(object):

    def __init__(self, cfg, box_coder):
        self.cfg = cfg
        self.cls_loss_func = SigmoidFocalLoss(cfg.MODEL.PAA.LOSS_GAMMA,
                                              cfg.MODEL.PAA.LOSS_ALPHA)
        self.iou_pred_loss_func = nn.BCEWithLogitsLoss(reduction="mean")
        self.margin = 0.2
        self.matcher = Matcher(cfg.MODEL.PAA.IOU_THRESHOLD,
                               cfg.MODEL.PAA.IOU_THRESHOLD,
                               True)
        self.box_coder = box_coder

    def compute_ious(self, boxes1, boxes2):
        area1 = (boxes1[:, 2] - boxes1[:, 0] + 1) * (boxes1[:, 3] - boxes1[:, 1] + 1)
        area2 = (boxes2[:, 2] - boxes2[:, 0] + 1) * (boxes2[:, 3] - boxes2[:, 1] + 1)
        lt = torch.max(boxes1[:, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
        wh = (rb - lt + 1).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        return inter / (area1 + area2 - inter)

    def prepare_iou_based_proxy_targets(self, pred_box, targets, anchors, proxy_target_in=None):
        """Compute IoU-based targets"""

        #OTHDO: change pred box to anchors like shape
        #OTHDO: Need to output GT IoU, and is_target boolean per location

        gt_iou = []
        matched_idxs = []
        pred_box_per_im_list = per_level_to_im(pred_box)

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyxy"
            anchors_per_im = cat_boxlist(anchors[im_i])
            pred_box_per_im = BoxList(self.box_coder.decode(pred_box_per_im_list[im_i], anchors_per_im.bbox), anchors_per_im.size)

            if proxy_target_in is None:
                match_quality_matrix = boxlist_iou(targets_per_im, anchors_per_im)
                matched_idxs_per_im = self.matcher(match_quality_matrix)
            else:
                matched_idxs_per_im = proxy_target_in["matched_idxs"][im_i]

            matched_targets = targets_per_im[matched_idxs_per_im.clamp(min=0)]
            gt_iou_per_im = self.compute_ious(matched_targets.bbox, pred_box_per_im.bbox)
            gt_iou.append(gt_iou_per_im)
            matched_idxs.append(matched_idxs_per_im)

        proxy_target = {
            "gt_iou": gt_iou,
            "matched_idxs": matched_idxs
        }

        return proxy_target

    def proxy_in_loss(self, iou_pred, proxy_target, iou_target):
        iou_pred = per_level_to_im(iou_pred)
        matched_idxs = proxy_target["matched_idxs"]
        ip_mask = []
        log_info = {}

        for ip, mi in zip(iou_pred, matched_idxs):
            mask = mi >= 0 
            ip_mask.append(ip[mask])
        ip_mask = torch.cat(ip_mask, dim=0)
        if iou_target:
            iou_target = torch.ones_like(ip_mask)
        else:
            iou_target = torch.zeros_like(ip_mask)

        return self.iou_pred_loss_func(ip_mask, iou_target), log_info
    
    def proxy_out_loss(self, new_proxy_target, proxy_target, iou_target):
        new_ip_mask = []
        ip_mask = []

        for ip, mi in zip(new_proxy_target["gt_iou"], new_proxy_target["matched_idxs"]):
            mask = mi >= 0
            new_ip_mask.append(ip[mask])
        new_ip_mask = torch.cat(new_ip_mask, dim=0)
        

        for ip, mi in zip(proxy_target["gt_iou"], proxy_target["matched_idxs"]):
            mask = mi >= 0
            ip_mask.append(ip[mask])
        ip_mask = torch.cat(ip_mask, dim=0)

        if iou_target == 0:
            diff = new_ip_mask - ip_mask + self.margin
        else:
            diff = ip_mask - new_ip_mask + self.margin

        # OTHDO: add hyper parameter to train loss stable
        loss = torch.log(1.+torch.exp(diff)).mean()
        log_info = None

        return {
            "proxy_out_{}_loss".format(iou_target):loss
        }, log_info


def make_paa_proxy_loss_evaluator(cfg, box_coder):
    loss_evaluator = ProxyLossComputation(cfg, box_coder)
    return loss_evaluator
