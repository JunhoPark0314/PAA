import math
import torch
import torch.nn.functional as F
from torch import nn

from .inference import make_atss_conly_postprocessor
from .loss import make_atss_conly_loss_evaluator

from paa_core.layers import Scale
from paa_core.layers import DFConv2d
from ..anchor_generator import make_anchor_generator_atss


class BoxCoder(object):

    def __init__(self, cfg):
        self.cfg = cfg
    
    def encode_disp(self, gt_ctr, anchors):
        TO_REMOVE = 1  # TODO remove
        wx, wy = (10., 10.)

        ex_widths = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
        ex_heights = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
        ex_ctr_x = (anchors[:, 2] + anchors[:, 0]) / 2
        ex_ctr_y = (anchors[:, 3] + anchors[:, 1]) / 2

        gt_ctr_x = gt_ctr[:,0]
        gt_ctr_y = gt_ctr[:,1]

        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets = torch.stack((targets_dx, targets_dy), dim=1)

        return targets
    
    def decode_disp(self, preds, anchors):
        
        anchors = anchors.to(preds.dtype)

        TO_REMOVE = 1  # TODO remove
        widths = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
        heights = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
        ctr_x = (anchors[:, 2] + anchors[:, 0]) / 2
        ctr_y = (anchors[:, 3] + anchors[:, 1]) / 2

        wx, wy, ww, wh = (10., 10., 5., 5.)
        dx = preds[:, 0::4] / wx
        dy = preds[:, 1::4] / wy

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]

        pred_ctr = torch.zeros_like(preds)
        pred_ctr[:, 0::4] = pred_ctr_x 
        pred_ctr[:, 1::4] = pred_ctr_y

        return pred_ctr

    def encode(self, gt_boxes, anchors):
        if self.cfg.MODEL.ATSS_CONLY.REGRESSION_TYPE == 'POINT':
            TO_REMOVE = 1  # TODO remove
            anchors_w = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
            anchors_h = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
            anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
            anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2

            w = self.cfg.MODEL.ATSS_CONLY.ANCHOR_SIZES[0] / self.cfg.MODEL.ATSS_CONLY.ANCHOR_STRIDES[0]
            l = w * (anchors_cx - gt_boxes[:, 0]) / anchors_w
            t = w * (anchors_cy - gt_boxes[:, 1]) / anchors_h
            r = w * (gt_boxes[:, 2] - anchors_cx) / anchors_w
            b = w * (gt_boxes[:, 3] - anchors_cy) / anchors_h
            targets = torch.stack([l, t, r, b], dim=1)
        elif self.cfg.MODEL.ATSS_CONLY.REGRESSION_TYPE == 'BOX':
            TO_REMOVE = 1  # TODO remove
            ex_widths = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
            ex_heights = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
            ex_ctr_x = (anchors[:, 2] + anchors[:, 0]) / 2
            ex_ctr_y = (anchors[:, 3] + anchors[:, 1]) / 2

            gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0] + TO_REMOVE
            gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1] + TO_REMOVE
            gt_ctr_x = (gt_boxes[:, 2] + gt_boxes[:, 0]) / 2
            gt_ctr_y = (gt_boxes[:, 3] + gt_boxes[:, 1]) / 2

            wx, wy, ww, wh = (10., 10., 5., 5.)
            targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
            targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
            targets_dw = ww * torch.log(gt_widths / ex_widths)
            targets_dh = wh * torch.log(gt_heights / ex_heights)
            targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)

        return targets

    def decode(self, preds, anchors):
        if self.cfg.MODEL.ATSS_CONLY.REGRESSION_TYPE == 'POINT':
            TO_REMOVE = 1  # TODO remove
            anchors_w = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
            anchors_h = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
            anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
            anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2

            w = self.cfg.MODEL.ATSS_CONLY.ANCHOR_SIZES[0] / self.cfg.MODEL.ATSS_CONLY.ANCHOR_STRIDES[0]
            x1 = anchors_cx - preds[:, 0] / w * anchors_w
            y1 = anchors_cy - preds[:, 1] / w * anchors_h
            x2 = anchors_cx + preds[:, 2] / w * anchors_w
            y2 = anchors_cy + preds[:, 3] / w * anchors_h
            pred_boxes = torch.stack([x1, y1, x2, y2], dim=1)
        elif self.cfg.MODEL.ATSS_CONLY.REGRESSION_TYPE == 'BOX':
            anchors = anchors.to(preds.dtype)

            TO_REMOVE = 1  # TODO remove
            widths = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
            heights = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
            ctr_x = (anchors[:, 2] + anchors[:, 0]) / 2
            ctr_y = (anchors[:, 3] + anchors[:, 1]) / 2

            wx, wy, ww, wh = (10., 10., 5., 5.)
            dx = preds[:, 0::4] / wx
            dy = preds[:, 1::4] / wy
            dw = preds[:, 2::4] / ww
            dh = preds[:, 3::4] / wh

            # Prevent sending too large values into torch.exp()
            dw = torch.clamp(dw, max=math.log(1000. / 16))
            dh = torch.clamp(dh, max=math.log(1000. / 16))

            pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
            pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
            pred_w = torch.exp(dw) * widths[:, None]
            pred_h = torch.exp(dh) * heights[:, None]

            pred_boxes = torch.zeros_like(preds)
            pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * (pred_w - 1)
            pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * (pred_h - 1)
            pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * (pred_w - 1)
            pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * (pred_h - 1)
        return pred_boxes


class ATSS_CONLYHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ATSS_CONLYHead, self).__init__()
        self.cfg = cfg

        center_tower = []
        rank_tower = []
        for i in range(cfg.MODEL.ATSS_CONLY.NUM_CONVS):
            if self.cfg.MODEL.ATSS_CONLY.USE_DCN_IN_TOWER and \
                    i == cfg.MODEL.ATSS_CONLY.NUM_CONVS - 1:
                conv_func = DFConv2d
            else:
                conv_func = nn.Conv2d

            rank_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            center_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            center_tower.append(nn.GroupNorm(32, in_channels))
            center_tower.append(nn.ReLU())

        self.add_module('center_tower', nn.Sequential(*center_tower))
        self.add_module('rank_tower', nn.Sequential(*rank_tower))

        towers = [self.center_tower, self.rank_tower]
        
        self.pred_rank = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        self.disp_vector = nn.Conv2d(
            in_channels, 2, kernel_size=3, stride=1,
            padding=1
        )
        self.disp_error = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1           
        )

        end_layer = [self.pred_rank, self.disp_vector, self.disp_error]

        # initialization
        for modules in [*towers, *end_layer]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.ATSS_CONLY.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.pred_rank.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        pred_disp_vector = []
        pred_disp_error = []
        pred_rank = []

        for l, feature in enumerate(x):
            center_tower = self.center_tower(feature)
            rank_tower = self.rank_tower(feature)

            pred_disp_vector.append(self.scales[l](self.disp_vector(center_tower)))
            pred_disp_error.append(self.disp_error(center_tower).exp())
            pred_rank.append(self.pred_rank(rank_tower))

        per_level_pred = {
            "pred_disp_vector": pred_disp_vector,
            "pred_disp_error": pred_disp_error,
            "pred_rank": pred_rank
        }

        return per_level_pred

class ATSS_CONLYModule(torch.nn.Module):

    def __init__(self, cfg, in_channels):
        super(ATSS_CONLYModule, self).__init__()
        self.cfg = cfg
        self.head = ATSS_CONLYHead(cfg, in_channels)
        box_coder = BoxCoder(cfg)
        self.loss_evaluator = make_atss_conly_loss_evaluator(cfg, box_coder)
        self.box_selector_test = make_atss_conly_postprocessor(cfg, box_coder)
        self.anchor_generator = make_anchor_generator_atss(cfg)

    def forward(self, images, features, targets=None):
        per_level_pred = self.head(features)
        anchors = self.anchor_generator(images, features)
 
        if self.training:
            return self._forward_train(per_level_pred, targets, anchors)
        else:
            return self._forward_test(per_level_pred, anchors)

    def _forward_train(self, per_level_pred, targets, anchors):
        losses, log_info = self.loss_evaluator(
            per_level_pred, targets, anchors
        )
        return None, losses, log_info

    def _forward_test(self, per_level_pred, anchors):
        boxes = self.box_selector_test(per_level_pred, anchors)
        return boxes, {}


def build_atss_conly(cfg, in_channels):
    return ATSS_CONLYModule(cfg, in_channels)
