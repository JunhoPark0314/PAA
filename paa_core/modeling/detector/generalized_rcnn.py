# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from paa_core.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        proposals, proposal_losses, log_info = self.rpn(images, features, targets)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, log_info

        return result, log_info
    
    def proxy_target(self, images, features, targets=None, proxy_target_in=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        proxy_target, log_info = self.rpn.proxy_target(images, features, targets, proxy_target_in)

        return proxy_target, log_info

    def proxy_in(self, images, features, proxy_target, iou_target=1, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        proxy_in_loss, log_info = self.rpn.proxy_in(images, features, targets, proxy_target, iou_target)

        return proxy_in_loss, log_info
    
    def proxy_out_loss(self, new_proxy_target, proxy_target, iou_target):
        return self.rpn.proxy_out_loss(new_proxy_target, proxy_target, iou_target)
    
    def param_list_key_add(self, key, param_list):
        for i, param in enumerate(param_list):
            param_list[i] = (key + param[0], param[1])
        return param_list
    
    def proxy_iou_parameters(self):
        bbox_tower_param = list(self.rpn.head.bbox_tower.named_parameters())
        bbox_tower_param = self.param_list_key_add("rpn.head.bbox_tower.", bbox_tower_param)

        iou_pred_param = list(self.rpn.head.iou_pred.named_parameters())
        iou_pred_param = self.param_list_key_add("rpn.head.iou_pred.", iou_pred_param)

        return bbox_tower_param + iou_pred_param
    
    def proxy_target_parameters(self):
        bbox_pred_param = list(self.rpn.head.bbox_pred.named_parameters())

        for i, param in enumerate(bbox_pred_param):
            bbox_pred_param[i] = ("rpn.head.bbox_pred." + param[0], param[1])
        
        return bbox_pred_param
    
    def proxy_whole_parameters(self):
        bbox_pred_param = list(self.rpn.head.bbox_pred.named_parameters())
        bbox_tower_param = list(self.rpn.head.bbox_tower.named_parameters())
        iou_pred_param = list(self.rpn.head.iou_pred.named_parameters())

        for i, param in enumerate(bbox_pred_param):
            bbox_pred_param[i] = ("rpn.head.bbox_pred." + param[0], param[1])
        
        for i, param in enumerate(bbox_tower_param):
            bbox_tower_param[i] = ("rpn.head.bbox_tower." + param[0], param[1])

        for i, param in enumerate(iou_pred_param):
            iou_pred_param[i] = ("rpn.head.iou_pred." + param[0], param[1])

        return bbox_pred_param + bbox_tower_param + iou_pred_param
