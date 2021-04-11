from copy import deepcopy
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

import matplotlib.pyplot as plt

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
        self.pair_num = 200
        self.adj_dist = cfg.MODEL.PAA.ADJ_DIST
        self.add_loc_info = cfg.MODEL.PAA.LOC_INFO
        self.add_cls_info = cfg.MODEL.PAA.CLS_INFO

        num_classes = cfg.MODEL.PAA.NUM_CLASSES - 1
        num_anchors = len(cfg.MODEL.PAA.ASPECT_RATIOS) * cfg.MODEL.PAA.SCALES_PER_OCTAVE

        # Per feature prediction network

        feature_tower = {
            "cls_tower": [],
            "bbox_tower": [],
        }

        if cfg.MODEL.PAA.PAIR_TOWER:
            feature_tower["pair_tower"] = []

        self.cls_logits = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=3, stride=1,
            padding=1
        )

        single_modules = [self.cls_logits, self.bbox_pred]

        for k, v in feature_tower.items():
            for i in range(cfg.MODEL.PAA.NUM_CONVS):
                if self.cfg.MODEL.PAA.USE_DCN_IN_TOWER and \
                        i == cfg.MODEL.PAA.NUM_CONVS - 1:
                    conv_func = DFConv2d
                else:
                    conv_func = nn.Conv2d

                v.append(
                    conv_func(
                        in_channels,
                        in_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True
                    )
                )
                v.append(nn.GroupNorm(32, in_channels))
                v.append(nn.ReLU())
            self.add_module(k, nn.Sequential(*v))
            single_modules.append(getattr(self, k))

        # Per pair prediction network

        self.pair_pred = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
        )
        self.cls_to_pair = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1,
        )
        self.reg_to_pair = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1,
        )
        self.diff_to_code = nn.Linear(
            2, (2 * self.adj_dist + 1) ** 2
        )

        pair_modules = [self.pair_pred, self.cls_to_pair, self.reg_to_pair, self.diff_to_code]

        # initialization
        for modules in (single_modules + pair_modules):
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.PAA.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)
        #torch.nn.init.constant_(self.pair_pred.bias, bias_value)
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        logits = []
        bbox_reg = []
        cls_top_feature = []
        reg_top_feature = []

        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)

            if hasattr(self, "pair_tower"):
                pair_tower = self.pair_tower(feature)
                cls_top_feature.append(pair_tower)
                reg_top_feature.append(pair_tower)
            else:
                cls_top_feature.append(cls_tower)
                reg_top_feature.append(box_tower)

            logits.append(self.cls_logits(cls_tower))

            bbox_pred = self.scales[l](self.bbox_pred(box_tower))
            bbox_reg.append(bbox_pred)

        pred = {
            "cls_logits": logits,
            "box_regression": bbox_reg,
            "cls_top_feature": cls_top_feature,
            "reg_top_feature": reg_top_feature,
        }

        return pred

    def take_patch_from_peak(self, per_level_feature, per_level_peak_list):
        patch_per_level = []
        C = per_level_feature[0].shape[1]
        for _, peak in enumerate(per_level_peak_list):
            if len(peak) == 0:
                continue

            feature = per_level_feature[_]
            # compute peak location to cover kernel size
            _, _, H, W = feature.shape
            P = len(peak)
            if P == 0:
                patch_per_level.append(torch.tensor([]).to(feature.device))
                continue
            peak[:,2] = peak[:,2].clamp(min=0, max=H-1)
            peak[:,3] = peak[:,3].clamp(min=0, max=W-1)

            patch = []
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    curr_peak = peak.clone().detach()
                    curr_peak[:,2] += i
                    curr_peak[:,3] += j
                    patch.append(curr_peak)
            patch = torch.cat(patch)
            patch[:,2] += 1
            patch[:,3] += 1
            idx = patch.split(1, dim=-1)
            feature = F.pad(input=feature, pad=(1, 1, 1, 1), mode='constant', value=0)
            patch = feature[idx[0], :, idx[2], idx[3]].reshape(3, 3, P, C).permute(2,3,0,1)
            patch_per_level.append(patch)

        return patch_per_level

    def get_adjacent_pos(self, per_level_peak, dist_lim, hw_list):
        per_level_adj_peak = []
        per_level_rel_diff = []
        adjacent_matrix = []

        for i in torch.arange(-dist_lim, dist_lim + 1):
            for j in torch.arange(-dist_lim, dist_lim + 1):
                adjacent_matrix.append(torch.tensor([[0,0,i,j]]))

        adjacent_matrix = torch.cat(adjacent_matrix)
        D = len(adjacent_matrix)
        for peak, (H, W) in zip(per_level_peak, hw_list):
            if len(peak):
                curr_peak = (peak.unsqueeze(1) + adjacent_matrix.to(peak.device)).reshape(-1, 4)
                curr_peak[:,2] = curr_peak[:,2].clamp(min=0, max=H-1)
                curr_peak[:,3] = curr_peak[:,3].clamp(min=0, max=W-1)
                rel_diff = (peak.unsqueeze(1).repeat(1, D, 1).view(-1, 4) - curr_peak)[:,2:]

                per_level_adj_peak.append(curr_peak)
                per_level_rel_diff.append(rel_diff)
            else:
                per_level_adj_peak.append(peak)
                per_level_rel_diff.append([])
        
        return per_level_adj_peak, per_level_rel_diff

    def extract_cls_weight(self, per_level_cls_peak):
        per_level_cls_weight = []

        for cls_peak in per_level_cls_peak:
            if len(cls_peak):
                per_level_cls_weight.append(self.cls_logits.weight[cls_peak[:,1]])

        return per_level_cls_weight
    
    def convert_diff_to_code(self, per_level_rel_diff):

        rel_code_per_level = []

        C = self.bbox_pred.in_channels
        for rel_diff in per_level_rel_diff:
            N = len(rel_diff)
            if N:
                """
                rel_code = torch.zeros(N, 3, 3).cuda()
                rel_diff_wi_id = torch.cat([torch.arange(N).view(-1,1).cuda(), rel_diff], dim=1)
                rel_code[rel_diff_wi_id.split(1,dim=1)] = 1
                rel_code_per_level.append(rel_code.unsqueeze(1))
                """

                rel_code = self.diff_to_code(rel_diff.float())
                #rel_code_per_level.append(F.softmax(rel_code, dim=0).view(-1, 1, 3, 3))
                rel_code_per_level.append(rel_code.view(-1, 1, 3, 3))


        return rel_code_per_level

    def forward_with_pair(self, pred_per_level, min_ppa_threshold):
        N = pred_per_level["cls_logits"][0].shape[0]
        L = len(pred_per_level["cls_logits"])
        per_image_level_cls_logit = disassemble_by_image(pred_per_level["cls_logits"])
        per_level_cls_peak = []
        per_level_cls_score = []

        with torch.no_grad():
            # compute per image threshold for top 1000
            per_image_cls_thresh = []
            for cls_logit in per_image_level_cls_logit:
                per_image_cls_thresh.append(0)
            #    per_image_cls_thresh.append(torch.cat([logit.flatten() for logit in cls_logit]).topk(1000, sorted=False)[0].min().sigmoid())
            

            peak_cls_list_per_image_level = [[torch.cat([torch.nonzero(cls_logit.sigmoid() > max(ppa_threshold, min_ppa_threshold)), 
                                                cls_logit.sigmoid()[cls_logit.sigmoid() > max(ppa_threshold, min_ppa_threshold)].unsqueeze(-1)], dim=-1)
                                                for cls_logit in per_level_cls_logit] 
                                                for per_level_cls_logit, ppa_threshold in zip(per_image_level_cls_logit, per_image_cls_thresh)]
            for lvl in range(L):
                per_level_cls_peak.append([])
                per_level_cls_score.append([])

            for im in range(N):
                for lvl in range(L):
                    curr_cls_peak = peak_cls_list_per_image_level[im][lvl]
                    if len(curr_cls_peak) > self.pair_num: 
                        _, idx = curr_cls_peak[:,-1].topk(self.pair_num)
                        curr_cls_peak = curr_cls_peak[idx]

                    if len(curr_cls_peak):
                        curr_cls_peak = curr_cls_peak.reshape(-1,4)
                        im_vector = torch.full((len(curr_cls_peak),), fill_value=im, device=curr_cls_peak.device).reshape(-1,1)
                        #cls_peak_pos = torch.cat([im_vector, curr_cls_peak[:,:3]], dim=-1)
                        cls_peak_pos = torch.cat([im_vector, curr_cls_peak[:,:3]], dim=-1)
                        cls_peak_score = curr_cls_peak[:,3].view(-1,1)
                        per_level_cls_peak[lvl].append(cls_peak_pos)
                        per_level_cls_score[lvl].append(cls_peak_score)

        for lvl in range(L):
            if len(per_level_cls_peak[lvl]):
                per_level_cls_peak[lvl] = torch.cat(per_level_cls_peak[lvl], dim=0).long()
                per_level_cls_score[lvl] = torch.cat(per_level_cls_score[lvl], dim=0).float()

        cls_patch_per_level = self.take_patch_from_peak(pred_per_level['cls_top_feature'], per_level_cls_peak)
        cls_weight_per_level = self.extract_cls_weight(per_level_cls_peak)
        per_level_iou_peak, per_level_rel_diff = self.get_adjacent_pos(per_level_cls_peak, self.adj_dist, get_hw_list(pred_per_level["cls_top_feature"]))
        rel_code_per_level = self.convert_diff_to_code(per_level_rel_diff)
        box_patch_per_level = self.take_patch_from_peak(pred_per_level['reg_top_feature'], per_level_iou_peak)

        pair_logit_per_level = []

        D = (self.adj_dist * 2 + 1) ** 2
        for cls_patch, box_patch, cls_weight, rel_code in zip(cls_patch_per_level, box_patch_per_level, cls_weight_per_level, rel_code_per_level):
            if len(cls_patch):
                C = cls_patch.shape[1]

                if self.add_cls_info:
                    cls_patch = self.cls_to_pair(cls_patch * cls_weight)
                else:
                    cls_patch = self.cls_to_pair(cls_patch)

                if self.add_loc_info:
                    box_patch = self.reg_to_pair(box_patch * rel_code).reshape(-1, D, C, 3, 3)
                else:
                    box_patch = self.reg_to_pair(box_patch).reshape(-1, D, C, 3, 3)

                pair_patch = (cls_patch.unsqueeze(1) + box_patch).reshape(-1, C, 3, 3)
                pair_logit_per_level.append(self.pair_pred(pair_patch).reshape(-1, D, 1))
            
        per_level_cls_score = [score for score in per_level_cls_score if len(score)]
        
        pred_per_level = {
            "pair_logit": pair_logit_per_level,
            "cls_peak": per_level_cls_peak,
            "reg_peak": per_level_iou_peak,
            "cls_score": per_level_cls_score,
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
            preds_per_level, targets, anchors, is_pa=True
        )
        return None, losses_dict, log_info

    def _forward_test(self, pred_per_level, anchors, targets=None):
        pred_per_pair = self.head.forward_with_pair(pred_per_level, 0.05)
        new_targets = deepcopy(targets)
        for nt in new_targets:
            nt.bbox = nt.bbox.cuda()
        boxes, log_info = self.box_selector_test(pred_per_level, pred_per_pair, anchors, new_targets)

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
