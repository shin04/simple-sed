from typing import Union
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.label_encoder import strong_label_decoding
from .metrics import (
    calc_sed_weak_f1,
    calc_sed_eval_metrics,
    calc_psds_eval_metrics,
    search_best_threshold
)


def decide_class_threshold(
    pred_dir,
    meta_strong: Path,
    sr: int,
    hop_length: int,
    pooling_rate: int,
    class_map: dict
):
    pred_dict = np.load(pred_dir, allow_pickle=True).item()
    preds = list(pred_dict.values())
    filenames = list(pred_dict.keys())

    best_th = search_best_threshold(
        0.1, meta_strong, preds, filenames,
        sr, hop_length, pooling_rate, class_map
    )

    return best_th


def test(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_map: dict,
    thresholds: list,
    sed_eval_thr: float,
    psds_params: dict,
    meta_strong: Path,
    meta_duration: Path,
    sr: int,
    hop_length: int,
    pooling_rate: int,
    best_th: dict
) -> Union[float, float]:
    model.eval()

    n_batch = len(dataloader)
    weak_f1_sum = 0
    results = {}
    # best_results = []
    preds = np.zeros(
        (len(dataloader.dataset), *dataloader.dataset[0]['target'].shape))
    filenames = []
    for thr in thresholds:
        results[thr] = []

    with torch.no_grad():
        for ite, item in enumerate(dataloader):
            data = item['waveform'].to(device)
            weak_labels = item['weak_label'].to(device)

            strong_pred, weak_pred = model(data)

            weak_f1_sum += calc_sed_weak_f1(weak_labels, weak_pred, sed_eval_thr)

            for i, pred in enumerate(strong_pred):
                pred = pred.to('cpu').detach().numpy().copy()

                preds[ite*len(strong_pred) + i] = pred
                filenames.append(item['filename'][i])

                for thr in thresholds:
                    result = strong_label_decoding(
                        pred, item['filename'][i],
                        sr, hop_length, pooling_rate, class_map,
                        thr
                    )
                    results[thr] += result

                # best_results += strong_label_decoding(
                #     pred, item['filename'][i],
                #     sr, hop_length, pooling_rate, class_map,
                #     best_th,
                # )

        # best_sed_evals = calc_sed_eval_metrics(
        #     meta_strong, pd.DataFrame(best_results), 0.1, 0.2
        # )
        # print('valid best sed evals', best_sed_evals)

        sed_evals = {}
        for thr in thresholds:
            sed_evals[thr] = calc_sed_eval_metrics(
                meta_strong, pd.DataFrame(results[thr]), 0.1, 0.2
            )
        print(sed_eval_thr)

        psds_eval_list, psds_macro_f1_list = [], []
        for i in range(psds_params['val_num']):
            psds_eval, psds_macro_f1 = calc_psds_eval_metrics(
                meta_strong,
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

        pred_dict = dict(zip(filenames, preds))

    return (
        psds_eval_list,
        psds_macro_f1_list,
        weak_f1,
        sed_evals[sed_eval_thr],
        pred_dict
    )
