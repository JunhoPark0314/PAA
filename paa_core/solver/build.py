# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import logging
from .lr_scheduler import WarmupMultiStepLR
from operator import itemgetter


def make_optimizer(cfg, model, interest=None):
    logger = logging.getLogger("paa_core.trainer")
    params = []
    interest_id = []
    if interest is not None:
        interest = [i[0] for i in interest]

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if key.endswith(".offset.weight") or key.endswith(".offset.bias"):
            logger.info("set lr factor of {} as {}".format(
                key, cfg.SOLVER.DCONV_OFFSETS_LR_FACTOR
            ))
            lr *= cfg.SOLVER.DCONV_OFFSETS_LR_FACTOR
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay, "key": key}]

        if interest is not None and key in interest:
            interest_id.append(len(params) - 1)

    optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)

    return optimizer, interest_id

def make_optimizer_wi_param_list(cfg, interest, optimizer, param_list):
    logger = logging.getLogger("paa_core.trainer")
    params = []
    interest_param = itemgetter(*interest)(optimizer.state_dict()["param_groups"])
    interest_param = {ip["key"]:ip for ip in interest_param}
    for (key, value) in param_list:
        if not value.requires_grad:
            continue
        if key not in interest_param:
            print("error")
        else:
            lr = interest_param[key]['lr']
            weight_decay = interest_param[key]['weight_decay']
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer


def make_lr_scheduler(cfg, optimizer):
    return WarmupMultiStepLR(
        optimizer,
        cfg.SOLVER.STEPS,
        cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
    )
