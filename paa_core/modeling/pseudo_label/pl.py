# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math
import torch
import torch.nn.functional as F
from torch import nn
from paa_core.layers import DFConv2d
from paa_core.layers import Scale

class GeneralizedPL(nn.Module):

	def __init__(self, cfg, in_channels):
		super(GeneralizedPL, self).__init__()
		self.cfg = cfg
		num_anchors = len(cfg.MODEL.SD.ASPECT_RATIOS) * cfg.MODEL.SD.SCALES_PER_OCTAVE
		num_classes = cfg.MODEL.SD.NUM_CLASSES

		self.use_iou_pred = cfg.MODEL.SD.USE_IOU_PRED
		self.class_embedding = nn.Parameter(torch.rand(num_classes, in_channels), requires_grad=True) - 0.5

		self.cls_logits = nn.Conv2d(
			in_channels, num_anchors, kernel_size=3, stride=1,
			padding=1
		)
		self.bbox_pred = nn.Conv2d(
			in_channels, num_anchors * 4 * 2, kernel_size=3, stride=1,
			padding=1
		)
		self.bbox_converter = nn.Conv2d(
			4, in_channels, kernel_size=1, stride=1 
		)

		all_modules = [self.cls_logits, self.bbox_pred, self.bbox_converter]
		if self.use_iou_pred:
			self.iou_pred = nn.Conv2d(
				in_channels, num_anchors * 1, kernel_size=3, stride=1,
				padding=1
			)
			all_modules.append(self.iou_pred)

		# initialization
		for modules in all_modules:
			for l in modules.modules():
				if isinstance(l, nn.Conv2d):
					torch.nn.init.normal_(l.weight, std=0.01)
					torch.nn.init.constant_(l.bias, 0)

		# initialize the bias for focal loss
		prior_prob = cfg.MODEL.SD.PRIOR_PROB
		bias_value = -math.log((1 - prior_prob) / prior_prob)
		#torch.nn.init.constant_(self.cls_logits.bias, bias_value)
		self.scales = nn.ModuleList([Scale(init_value=0.1) for _ in range(5)])
	
	def forward(self, x, pl_module_info):
		logits = []
		bbox_reg = []
		iou_pred = []
		B = len(x["cls"][0])
		pl_module_info["cls_emb"] = pl_module_info["cls_emb"].reshape(B, -1)
		pl_module_info["box_emb"] = pl_module_info["box_emb"].reshape(B, -1, 4)

		cls_emb_level = []
		box_emb_level = []

		start_idx = 0
		for cls_tower in x["cls"]:
			_, _, H, W = cls_tower.shape
			curr_len = H*W
			per_level_cls_emb = pl_module_info["cls_emb"][:,start_idx:start_idx + curr_len].reshape(B, H, W)
			per_level_box_emb = pl_module_info["box_emb"][:,start_idx:start_idx + curr_len].reshape(B, H, W, -1).permute(0, 3, 1, 2)
			cls_emb_level.append(self.class_embedding[per_level_cls_emb].reshape(B, H, W, -1).permute(0, 3, 1, 2).cuda())
			box_emb_level.append(per_level_box_emb)
			start_idx += curr_len

		for l, (cls_tower, box_tower, cls_emb, box_emb) in enumerate(zip(x["cls"], x["reg"], cls_emb_level, box_emb_level)):
			logits.append(self.cls_logits(cls_tower+cls_emb).permute(0, 2, 3, 1).reshape(B, -1))
			curr_box_emb = self.bbox_converter(box_emb)
			bbox_pred = self.bbox_pred(box_tower + curr_box_emb)
			mean_pred = self.scales[l](bbox_pred[:,:4,:,:])
			sigma_pred = self.scales[l](bbox_pred[:,4:,:,:].sigmoid())

			bbox_pred = torch.cat([mean_pred, sigma_pred], dim=1)
			bbox_reg.append(bbox_pred.permute(0, 2, 3, 1).reshape(B, -1, 8))

			if self.use_iou_pred:
				iou_pred.append(self.iou_pred(box_tower + curr_box_emb).permute(0, 2, 3, 1).reshape(B, -1))

		noise_logits = torch.cat(logits, dim=1).reshape(-1)[pl_module_info["noise_idx"]]
		noise_bbox_reg = torch.cat(bbox_reg, dim=1).reshape(-1, 8)[pl_module_info["noise_idx"],:]
		#origin_bbox = torch.cat([b.permute(0, 2, 3, 1).reshape(B, -1, 4) for b in box_emb_level], dim=1).reshape(-1, 4)[pl_module_info["noise_idx"],:]
		noise_bbox = torch.normal(mean = noise_bbox_reg[:,:4], std=noise_bbox_reg[:,4:])

		res = [noise_logits, noise_bbox]
		if self.use_iou_pred:
			noise_iou_pred = torch.cat(iou_pred, dim=1).reshape(-1)[pl_module_info["noise_idx"]]
			res.append(noise_iou_pred)

		return res

def build_pl(cfg, in_channels):
	return GeneralizedPL(cfg, in_channels)
