import math
import torch
import torch.nn.functional as F
from torch import nn

from .inference import make_paa_postprocessor
from .loss import make_dcr_loss_evaluator

from paa_core.layers import Scale
from paa_core.layers import DFConv2d
from ..anchor_generator import make_anchor_generator_paa
from ..atss.atss import BoxCoder
class DCRHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(DCRHead, self).__init__()
        self.cfg = cfg
        num_classes = cfg.MODEL.PAA.NUM_CLASSES - 1
        num_anchors = len(cfg.MODEL.PAA.ASPECT_RATIOS) * cfg.MODEL.PAA.SCALES_PER_OCTAVE

        # Per feature prediction network
        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.PAA.NUM_CONVS):
            if self.cfg.MODEL.PAA.USE_DCN_IN_TOWER and \
                    i == cfg.MODEL.PAA.NUM_CONVS - 1:
                conv_func = DFConv2d
            else:
                conv_func = nn.Conv2d

            cls_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())

            bbox_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))

        self.cls_logits = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=3, stride=1,
            padding=1
        )
        self.iou_pred = nn.Conv2d(
            in_channels, num_anchors * 1, kernel_size=3, stride=1,
            padding=1
        )

        # Per pair prediction network
        self.bbox_to_cls = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1,
            padding=1
        )
        self.cls_to_bbox = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1,
            padding=1
        )

        all_modules = [self.cls_tower, self.bbox_tower,
                       self.cls_logits, self.bbox_pred, self.iou_pred, self.bbox_to_cls, self.cls_to_bbox]

        # initialization
        for modules in all_modules:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.PAA.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)
        torch.nn.init.constant_(self.iou_pred.bias, bias_value)
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        logits = []
        bbox_reg = []
        iou_pred = []
        cls_top_feature = []
        reg_top_feature = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)
            cls_top_feature.append(cls_tower)
            reg_top_feature.append(box_tower)

            logits.append(self.cls_logits(cls_tower))

            bbox_pred = self.scales[l](self.bbox_pred(box_tower))
            bbox_reg.append(bbox_pred)

            iou_pred.append(self.iou_pred(box_tower))
        
        pred = {
            "cls_logits": logits,
            "box_regression": bbox_reg,
            "iou_pred": iou_pred,
            "cls_top_feature": cls_top_feature,
            "box_top_feature": reg_top_feature,
        }

        return pred
class DCRModule(torch.nn.Module):

    def __init__(self, cfg, in_channels):
        super(DCRModule, self).__init__()
        self.cfg = cfg
        self.head = DCRHead(cfg, in_channels)
        box_coder = BoxCoder(cfg)
        self.loss_evaluator = make_dcr_loss_evaluator(cfg, box_coder, self.head)
        """---------------------------------------------------"""
        self.box_selector_test = make_paa_postprocessor(cfg, box_coder)
        self.anchor_generator = make_anchor_generator_paa(cfg)
        self.fpn_strides = cfg.MODEL.PAA.ANCHOR_STRIDES

    def forward(self, images, features, targets=None):
        preds_per_level = self.head(features)
        anchors = self.anchor_generator(images, features)
 
        if self.training:
            return self._forward_train(preds_per_level, targets, anchors,)
        else:
            #return self._forward_test(preds_per_level, anchors)
            return self._forward_test(preds_per_level, anchors, targets)

    def _forward_train(self, preds_per_level, targets, anchors,):
        losses_dict, log_info = self.loss_evaluator(
            preds_per_level, targets, anchors,
        )
        return None, losses_dict, log_info

    def _forward_test(self, preds_per_level, anchors, targets=None):
        boxes, bbox_iou, det_iou, pred_iou = self.box_selector_test(preds_per_level, anchors, targets)
        return boxes, {}, [bbox_iou, det_iou, pred_iou]

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations

def build_dcr(cfg, in_channels):
    return DCRModule(cfg, in_channels)