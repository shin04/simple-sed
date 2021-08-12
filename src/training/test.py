from typing import Union
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dcase_util.containers import MetaDataContainer

from utils.label_encoder import strong_label_decoding
from .metrics import (
    calc_sed_weak_f1,
    calc_sed_eval_metrics,
    calc_psds_eval_metrics
)


def generate_gt(targets, filenames, sr, hop_length,  pooling_rate, class_map, thr):
    for i, target in enumerate(targets):
        label = target.to('cpu').detach().numpy().copy()
        gt = strong_label_decoding(
            label, filenames[i], sr, hop_length, pooling_rate, class_map, thr
        )

    return gt


def test(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_map: dict,
    thresholds: list,
    psds_params: dict,
    meta_strong: Path,
    meta_duration: Path,
    sr: int,
    hop_length: int,
    pooling_rate: int,
) -> Union[float, float]:
    model.eval()

    n_batch = len(dataloader)
    weak_f1_sum = 0
    results = {}
    gt = []
    for thr in thresholds:
        results[thr] = []

    with torch.no_grad():
        for _, item in enumerate(dataloader):
            data = item['waveform'].to(device)
            weak_labels = item['weak_label'].to(device)

            strong_pred, weak_pred = model(data)

            weak_f1_sum += calc_sed_weak_f1(weak_labels, weak_pred)

            # generate ground truth df
            gt += generate_gt(
                strong_pred, item['filename'], sr, hop_length, pooling_rate, class_map, 0.5
            )

            for i, pred in enumerate(strong_pred):
                label = pred.to('cpu').detach().numpy().copy()
                for thr in thresholds:
                    result = strong_label_decoding(
                        label, item['filename'][i], sr, hop_length, pooling_rate, class_map, thr
                    )
                    results[thr] += result

        sed_evals = calc_sed_eval_metrics(
            meta_strong, MetaDataContainer(results[0.5]), 0.1, 0.2
        )

        psds_eval_list, psds_macro_f1_list = [], []
        for i in range(psds_params['val_num']):
            psds_eval, psds_macro_f1 = calc_psds_eval_metrics(
                # meta_strong,
                gt,
                meta_duration,
                results,
                dtc_threshold=psds_params['dtc_thresholds'][i],
                gtc_threshold=psds_params['gtc_thresholds'][i],
                cttc_threshold=psds_params['cttc_thresholds'][i],
                alpha_ct=psds_params['alpha_cts'][i],
                alpha_st=psds_params['alpha_sts'][i],
            )

            psds_eval_list.append(psds_eval)
            psds_macro_f1_list.append(psds_macro_f1)

        weak_f1 = weak_f1_sum / n_batch

    return (
        psds_eval_list,
        psds_macro_f1_list,
        weak_f1,
        sed_evals,
    )
