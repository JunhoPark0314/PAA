# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

from itertools import chain
import torch
from torch import nn

from paa_core.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..pseudo_label import build_pl


class PseudoLabelingRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(PseudoLabelingRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.pl_module = build_pl(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None, clean_only=False):
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
        images.tensors.requires_grad = True
        features = self.backbone(images.tensors)

        if clean_only:
            result, losses = self.rpn(images, features, targets)
        else:
            result, losses = self.rpn(images, features, targets, self.pl_module)

        if self.training:
            return losses, features

        return result
    
    def rpn_parameters(self):
        return self.rpn.parameters()