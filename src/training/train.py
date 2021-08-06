from typing import Union
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dcase_util.containers import MetaDataContainer
import mlflow

from utils.label_encoder import strong_label_decoding
from .metrics import (
    calc_sed_weak_f1,
    calc_sed_eval_metrics,
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
    train_strong_loss_sum = 0
    train_weak_loss_sum = 0
    train_tot_loss_sum = 0

    for i, item in enumerate(dataloader):
        optimizer.zero_grad()

        data = item['waveform'].to(device)
        labels = item['target'].to(device)
        weak_labels = item['weak_label'].to(device)

        strong_pred, weak_pred = model(data)

        strong_loss = criterion(strong_pred, labels)
        weak_loss = criterion(weak_pred, weak_labels)
        tot_loss = strong_loss + weak_loss

        strong_loss.backward()
        optimizer.step()
        train_strong_loss_sum += strong_loss.item()
        train_weak_loss_sum += weak_loss.item()
        train_tot_loss_sum += tot_loss.item()

        mlflow.log_metric('step_train/strong/loss',
                          strong_loss.item(), step=global_step+i+1)
        mlflow.log_metric('step_train/weak/loss',
                          weak_loss.item(), step=global_step+i+1)
        mlflow.log_metric('step_train/tot/loss',
                          tot_loss.item(), step=global_step+i+1)

    train_strong_loss = train_strong_loss_sum / n_batch
    train_weak_loss = train_weak_loss_sum / n_batch
    train_tot_loss = train_tot_loss_sum / n_batch

    return train_strong_loss, train_weak_loss, train_tot_loss


def valid(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    class_map: dict,
    thresholds: list,
    psds_params: dict,
    meta_strong: Path,
    meta_duration: Path,
) -> Union[float, float]:
    model.eval()

    n_batch = len(dataloader)
    valid_strong_loss_sum = 0
    valid_weak_loss_sum = 0
    valid_tot_loss_sum = 0
    weak_f1_sum = 0
    results = {}
    for thr in thresholds:
        results[thr] = []

    with torch.no_grad():
        for _, item in enumerate(dataloader):
            data = item['waveform'].to(device)
            labels = item['target'].to(device)
            weak_labels = item['weak_label'].to(device)

            strong_pred, weak_pred = model(data)

            strong_loss = criterion(strong_pred, labels)
            weak_loss = criterion(weak_pred, weak_labels)
            tot_loss = strong_loss + weak_loss

            valid_strong_loss_sum += strong_loss.item()
            valid_weak_loss_sum += weak_loss.item()
            valid_tot_loss_sum += tot_loss.item()

            weak_f1_sum += calc_sed_weak_f1(weak_labels, weak_pred)

            for i, pred in enumerate(strong_pred):
                label = pred.to('cpu').detach().numpy().copy()
                for thr in thresholds:
                    result = strong_label_decoding(
                        label, item['filename'][i], 44100, 256, class_map, thr
                    )
                    results[thr] += result

        sed_evals = calc_sed_eval_metrics(
            meta_strong, MetaDataContainer(results[0.5]), 0.1, 0.2
        )

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

        pred_df = pd.DataFrame(results[0.5])
        pred_df.to_csv('/ml/pred.csv')

        valid_strong_loss = valid_strong_loss_sum / n_batch
        valid_weak_loss = valid_weak_loss_sum / n_batch
        valid_tot_loss = valid_tot_loss_sum / n_batch
        valid_weak_f1 = weak_f1_sum / n_batch

    return (
        valid_strong_loss,
        valid_weak_loss,
        valid_tot_loss,
        psds_eval_list,
        psds_macro_f1_list,
        valid_weak_f1,
        sed_evals,
    )
