# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os
import pprint

import torch
from tqdm import tqdm

from paa_core.config import cfg
from paa_core.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from .bbox_aug import im_detect_bbox_aug
from .bbox_aug_vote import im_detect_bbox_aug_vote
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def compute_on_dataset(model, data_loader, device, timer=None):
    model.eval()
    results_dict = {}
    log_info_whole = {
    }
    iou_dict = [None] * 3
    cpu_device = torch.device("cpu")
    num_trg = 0
    num_det = 0
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        num_trg += sum([len(trg) for trg in targets])
        with torch.no_grad():
            if timer:
                timer.tic()
            if cfg.TEST.BBOX_AUG.ENABLED:
                if cfg.TEST.BBOX_AUG.VOTE:
                    output = im_detect_bbox_aug_vote(model, images, device)
                else:
                    output = im_detect_bbox_aug(model, images, device)
            else:
                output, log_info = model(images.to(device), targets)
            if timer:
                torch.cuda.synchronize()
                timer.toc()
            if output is not None:
                output = [o.to(cpu_device) for o in output]

        if output is not None:
            num_det += sum([len(trg) for trg in output])
            results_dict.update(
                {img_id: result for img_id, result in zip(image_ids, output)}
            )
        if targets is not None:
            for k, v in log_info.items():
                if k not in log_info_whole:
                    log_info_whole[k] = [v]
                else:
                    log_info_whole[k].append(v)
        if _ > 20:
            break
        
    #fig, axes = plt.subplots(10, figsize=(5,15))
    for _, (k, v) in enumerate(log_info_whole.items()):
        v = torch.cat([item for sublist in v for item in sublist if len(item)])
        log_info_whole[k] = v
        #df = pd.DataFrame({k: log_info_whole[k]})
        #sns.histplot(data=df, x=k, ax=axes[_])
    
    for thr in [0.5, 0.75, 0.9]:
        print(thr)
        for k, v in log_info_whole.items():
            print((v >= thr).sum() / len(v), (v == 0).sum() / len(v))
    
    #plt.tight_layout()
    #fig.savefig("output/test.png")
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("paa_core.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("paa_core.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, device, inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
