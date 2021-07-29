from typing import Union
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from dcase_util.containers import MetaDataContainer
import mlflow

from utils.label_encoder import strong_label_decoding
from .metrics import (
    sed_average_precision,
    # calc_sed_eval_metrics,
    calc_psds_eval_metrics
)


def train(
    global_step: int,
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
):
    model.train()

    n_batch = len(dataloader)
    train_loss_sum = 0
    map_sum = 0

    for i, item in enumerate(dataloader):
        optimizer.zero_grad()

        data = item['waveform'].to(device)
        labels = item['target'].to(device)

        outputs = model(data)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss_sum += loss.item()

        map_sum += sed_average_precision(labels, outputs)

        mlflow.log_metric('step_train_loss', loss.item(), step=global_step+i+1)
        mlflow.log_metric('step_train_map',
                          map_sum/(i+1), step=global_step+i+1)

    train_loss = train_loss_sum / n_batch
    train_map = map_sum / n_batch

    return train_loss, train_map


def valid(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    class_map: dict,
    thresholds: list,
    psds_params: dict
) -> Union[float, float]:
    model.eval()

    n_batch = len(dataloader)
    valid_loss_sum = 0
    map_sum = 0
    results = {}
    for thr in thresholds:
        results[thr] = []

    with torch.no_grad():
        for _, item in enumerate(dataloader):
            data = item['waveform'].to(device)
            labels = item['target'].to(device)

            outputs = model(data)

            loss = criterion(outputs, labels)
            valid_loss_sum += loss.item()

            map_sum += sed_average_precision(labels, outputs)

            for i, pred in enumerate(outputs):
                label = pred.to('cpu').detach().numpy().copy()
                for thr in thresholds:
                    result = strong_label_decoding(
                        label, item['filename'][i], 44100, 256, class_map, thr
                    )
                    results[thr] += result

        # sed_evals = calc_sed_eval_metrics(
        #     Path('/ml/meta/valid_meta_strong.csv'),
        #     MetaDataContainer(results),
        #     0.1, 0.2
        # )

        psds_eval_list, psds_macro_f1_list = [], []
        for i in range(psds_params['val_num']):
            dtc_threshold = psds_params['dtc_thresholds'][i]
            gtc_threshold = psds_params['gtc_thresholds'][i]
            cttc_threshold = psds_params['cttc_thresholds'][i]
            alpha_ct = psds_params['alpha_cts'][i]
            alpha_st = psds_params['alpha_sts'][i]

            psds_eval, psds_macro_f1 = calc_psds_eval_metrics(
                Path('/ml/meta/valid_meta_strong.csv'),
                Path('/ml/meta/valid_meta_duration.csv'),
                results,
                dtc_threshold=dtc_threshold,
                gtc_threshold=gtc_threshold,
                cttc_threshold=cttc_threshold,
                alpha_ct=alpha_ct,
                alpha_st=alpha_st
            )

            psds_eval_list.append(psds_eval)
            psds_macro_f1_list.append(psds_macro_f1)

        valid_loss = valid_loss_sum / n_batch
        valid_map = map_sum / n_batch

    return valid_loss, valid_map, psds_eval_list, psds_macro_f1_list
