import math
import torch
import torch.nn.functional as F
from torch import nn

from .inference import make_paa_postprocessor
from .loss import make_dcr_loss_evaluator, per_im_to_level, get_hw_list

from paa_core.layers import Scale
from paa_core.layers import DFConv2d
from ..anchor_generator import make_anchor_generator_paa
from ..atss.atss import BoxCoder

def disassemble_by_image(per_level_list):
    N = per_level_list[0].shape[0]
    per_image_level_list = []

    for ng in range(N):
        curr_image_list = []
        for trg in per_level_list:
            curr_image_list.append(trg[ng])
        per_image_level_list.append(curr_image_list)

    return per_image_level_list

def compute_dist(pos1, pos2):
    return ((pos1.unsqueeze(1) - pos2) ** 2).sum(dim=-1).sqrt()

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
        self.bbox_to_pair = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1,
            padding=1
        )
        self.cls_to_pair = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1,
            padding=1
        )
        self.pair_pred = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
        )

        all_modules = [self.cls_tower, self.bbox_tower,
                       self.cls_logits, self.bbox_pred, self.iou_pred, 
                       self.bbox_to_pair, self.cls_to_pair, self.pair_pred]

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

    def take_patch_from_peak(self, per_level_feature, per_level_peak_list):
        kernel_size = 3
        patch_per_level = []
        C = per_level_feature[0].shape[1]
        for feature, peak in zip(per_level_feature, per_level_peak_list):
            # compute peak location to cover kernel size
            P = len(peak)
            patch = []
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    curr_peak = peak.clone().detach()
                    curr_peak[:,1] += i
                    curr_peak[:,2] += j
                    patch.append(curr_peak)
            patch = torch.cat(patch)
            patch[:,1] += 1
            patch[:,2] += 1
            idx = patch.split(1, dim=-1)
            feature = F.pad(input=feature, pad=(1, 1, 1, 1), mode='constant', value=0)
            patch = feature[idx[0], :, idx[1], idx[2]].reshape(3, 3, P, C).permute(2,3,0,1)
            patch_per_level.append(patch)

        return patch_per_level

    def forward_with_pair(self, pred_per_level, ppa_threshold):
        N = pred_per_level["cls_logits"][0].shape[0]
        L = len(pred_per_level["cls_logits"])
        per_image_level_cls_logit = disassemble_by_image(pred_per_level["cls_logits"])
        per_image_level_iou = disassemble_by_image(pred_per_level["iou_pred"])
        per_level_cls_peak = []
        per_level_iou_peak = []

        with torch.no_grad():
            peak_cls_list_per_image_level = [[torch.cat([torch.nonzero(cls_logit.sigmoid() > ppa_threshold), 
                                                cls_logit.sigmoid()[cls_logit.sigmoid() > ppa_threshold].unsqueeze(-1)], dim=-1)
                                                for cls_logit in per_level_cls_logit] 
                                                for per_level_cls_logit in per_image_level_cls_logit]
            peak_iou_list_per_image_level = [[torch.cat([torch.nonzero(iou.sigmoid() > ppa_threshold), 
                                                iou.sigmoid()[iou.sigmoid() > ppa_threshold].unsqueeze(-1)], dim=-1)
                                                for iou in per_level_iou] 
                                                for per_level_iou in per_image_level_iou]
            for lvl in range(L):
                per_level_cls_peak.append([])
                per_level_iou_peak.append([])

            for im in range(N):
                for lvl in range(L):
                    curr_cls_peak = peak_cls_list_per_image_level[im][lvl]
                    curr_iou_peak = peak_iou_list_per_image_level[im][lvl]
                    if len(curr_cls_peak) > 1000: 
                        _, idx = curr_cls_peak[:,-1].topk(1000)
                        curr_cls_peak = curr_cls_peak[idx]

                    if len(curr_iou_peak) > 1000: 
                        _, idx = curr_iou_peak[:,-1].topk(1000)
                        curr_iou_peak = curr_iou_peak[idx]

                    cls_pos = curr_cls_peak[:,[1,2]]
                    reg_pos = curr_iou_peak[:,[1,2]]
                    dist_per_pos = compute_dist(cls_pos, reg_pos)
                    accepted_pos_pair = torch.nonzero(dist_per_pos <= 4)

                    if len(accepted_pos_pair) > 100:
                        cls_iou_score = (curr_cls_peak[accepted_pos_pair[:,0], -1] * curr_iou_peak[accepted_pos_pair[:,1], -1]).sqrt()
                        _, idx = cls_iou_score.topk(100)
                        accepted_pos_pair = accepted_pos_pair[idx]

                    peak_cls_list_per_image_level[im][lvl] = curr_cls_peak[accepted_pos_pair[:,0]]
                    peak_iou_list_per_image_level[im][lvl] = curr_iou_peak[accepted_pos_pair[:,1]]

                    cls_peak_pos = torch.cat([torch.full((len(accepted_pos_pair),), fill_value=im, device=curr_cls_peak.device).reshape(-1,1), curr_cls_peak[accepted_pos_pair[:,0]][:,[1,2]]], dim=-1)
                    iou_peak_pos = torch.cat([torch.full((len(accepted_pos_pair),), fill_value=im, device=curr_iou_peak.device).reshape(-1,1), curr_iou_peak[accepted_pos_pair[:,1]][:,[1,2]]], dim=-1)
                    per_level_cls_peak[lvl].append(cls_peak_pos)
                    per_level_iou_peak[lvl].append(iou_peak_pos)

        for lvl in range(L):
            per_level_cls_peak[lvl] = torch.cat(per_level_cls_peak[lvl], dim=0).long()
            per_level_iou_peak[lvl] = torch.cat(per_level_iou_peak[lvl], dim=0).long()

        cls_patch_per_level = self.take_patch_from_peak(pred_per_level['cls_top_feature'], per_level_cls_peak)
        box_patch_per_level = self.take_patch_from_peak(pred_per_level['box_top_feature'], per_level_iou_peak)

        pair_logit_per_level = []
        for cls_patch, box_patch in zip(cls_patch_per_level, box_patch_per_level):
            cls_patch = self.cls_to_pair(cls_patch)
            box_patch = self.bbox_to_pair(box_patch)
            pair_logit_per_level.append(self.pair_pred(cls_patch + box_patch).reshape(-1, 1))
        
        pred_per_level = {
            "pair_logit": pair_logit_per_level,
            "cls_peak": per_level_cls_peak,
            "reg_peak": per_level_iou_peak
        }
        return pred_per_level

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

    def _forward_test(self, pred_per_level, anchors, targets=None):
        if targets is not None:
            for trg in targets:
                trg.bbox = trg.bbox.to(anchors[0][0].bbox.device)
            iou_based_targets = self.loss_evaluator.prepare_iou_based_targets(targets, anchors)
            hw_list = get_hw_list(pred_per_level["cls_top_feature"])
            for k, v in iou_based_targets.items():
                iou_based_targets[k] = per_im_to_level(v, hw_list)
            boxes, log_info = self.box_selector_test(pred_per_level, anchors, targets, iou_based_targets)
        else:
            boxes, log_info = self.box_selector_test(pred_per_level, anchors)

        return boxes, {}, log_info

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