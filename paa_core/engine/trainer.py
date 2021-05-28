# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
import copy

import torch
import torch.distributed as dist

from paa_core.utils.comm import get_world_size, is_main_process, is_pytorch_1_1_0_or_later
from torch.nn.parallel import DistributedDataParallel as DDP
from paa_core.utils.metric_logger import MetricLogger
from torch.utils.tensorboard import SummaryWriter


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    proxy_train,
):
    logger = logging.getLogger("paa_core.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    pytorch_1_1_0_or_later = is_pytorch_1_1_0_or_later()

    if is_main_process():
        writer = SummaryWriter(log_dir=checkpointer.save_dir)

    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        # in pytorch >= 1.1.0, scheduler.step() should be run after optimizer.step()
        if not pytorch_1_1_0_or_later:
            scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict, log_info = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if pytorch_1_1_0_or_later:
            scheduler.step()
        
        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        if proxy_train is not None:
            do_proxy_train(model, optimizer, images, targets, meters, proxy_train, log_info["backbone_feature"])
            proxy_time = time.time() - end
            end = time.time()
            meters.update(proxy_time=proxy_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if (iteration % 20 == 0 or iteration == max_iter) and is_main_process():
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

            for k, v in loss_dict.items():
                writer.add_scalar(k, v, iteration)

            if len(log_info.keys()):
                logger.info(
                    meters.delimiter.join(
                        ["{}: {:.4f}".format(k, v) for k, v in log_info.items() if not isinstance(v, list)]
                    )
                )
                logger.info(
                    "----------------------------------------------------------------------"
                )

                for k, v in log_info.items():
                    if not isinstance(v, list):
                        writer.add_scalar(k, v, iteration)

        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

def do_proxy_train(model, optimizer, images, targets, meters, proxy_train_cfg, backbone_feature):

    proxy_grad = []
    proxy_loss = []
    torch.autograd.set_detect_anomaly(True)

    for iou_target in [1]:
        end = time.time()

        if isinstance(model, DDP):
            new_model = copy.deepcopy(model).module 
        else:
            new_model = copy.deepcopy(model)

        proxy_target, proxy_target_log_info = new_model.proxy_target(images, backbone_feature, targets,)
        proxy_target_time = time.time() - end

        # OTHDO: set requires_grad false to non-proxy task parameters
        partial_optimizer = proxy_train_cfg["partial_optimizer"](optimizer, new_model.proxy_iou_parameters())

        # OTHDO: Check learning rate from partial optimizer
        #eta = partial_optimizer.param_groups[0]["lr"]

        end = time.time()
        proxy_in_loss_dict, proxy_in_log_info = new_model.proxy_in(images, backbone_feature, proxy_target, iou_target, targets)

        proxy_in_losses = sum(loss for loss in proxy_in_loss_dict.values())

        partial_optimizer.zero_grad()
        proxy_in_losses.backward()
        partial_optimizer.step()
        proxy_in_time = time.time() - end


        new_proxy_target, proxy_target_log_info = new_model.proxy_target(images, backbone_feature, targets, proxy_target)

        end = time.time()
        proxy_out_loss_dict, proxy_out_log_info = new_model.proxy_out_loss(new_proxy_target, proxy_target, iou_target)

        proxy_out_losses = sum(loss for loss in proxy_out_loss_dict.values())
        # OTHDO: check if gradient of old and new model are same

        proxy_loss_dict_reduced = reduce_loss_dict(proxy_out_loss_dict)
        proxy_losses_reduced = sum(loss for loss in proxy_loss_dict_reduced.values())
        proxy_loss.append(proxy_losses_reduced)

        #make_dot(proxy_out_losses, params=dict(new_model.named_parameters())).render("graph", format="png")
        proxy_param = new_model.proxy_target_parameters()
        proxy_param = [pa[1] for pa in proxy_param]
        proxy_grad.append(torch.autograd.grad(proxy_out_losses, proxy_param, allow_unused=True))
        proxy_out_time = time.time() - end

    if isinstance(model, DDP): 
        proxy_target_param = model.module.proxy_target_parameters()
    else:
        proxy_target_param = model.proxy_target_parameters()

    proxy_optimizer = proxy_train_cfg["partial_optimizer"](optimizer ,proxy_target_param)
    proxy_optimizer.zero_grad()
    grad_std = []

#    for v, g0, g1 in zip(proxy_target_param, proxy_grad[0], proxy_grad[1]):
#        grad = (g0.data + g1.data) / 2
    for v, g0 in zip(proxy_target_param, proxy_grad[0]):
        grad = g0.data
        grad_std.append(grad.std())
        if v[1].grad is None:
            v[1].grad = grad
        else:
            v[1].grad.data.copy_(grad)
    
    proxy_optimizer.step()
    #meters.update(proxy_loss_down=proxy_loss[0], proxy_loss_up=proxy_loss[1], grad_std = sum(grad_std) / len(grad_std))
    meters.update(proxy_loss_up=proxy_loss[0], grad_std = sum(grad_std) / len(grad_std))