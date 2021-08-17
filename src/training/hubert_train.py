from typing import Union
from pathlib import Path

# import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow

from utils.label_encoder import strong_label_decoding
from .metrics import (
    calc_sed_weak_f1,
    calc_sed_eval_metrics,
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

        data = item['feat'].to(device)
        labels = item['target'].to(device)
        weak_labels = item['weak_label'].to(device)

        strong_pred, weak_pred = model(data)

        strong_loss = criterion(strong_pred, labels)
        weak_loss = criterion(weak_pred, weak_labels)
        tot_loss = strong_loss + weak_loss

        tot_loss.backward()
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
    sed_eval_thr: float,
    meta_strong: Path,
) -> Union[float, float]:
    model.eval()

    n_batch = len(dataloader)
    valid_strong_loss_sum = 0
    valid_weak_loss_sum = 0
    valid_tot_loss_sum = 0
    weak_f1_sum = 0
    results = {}
    # preds = np.zeros(
    #     (len(dataloader.dataset), *dataloader.dataset[0]['target'].shape))
    # filenames = []
    for thr in thresholds:
        results[thr] = []

    with torch.no_grad():
        for _, item in enumerate(dataloader):
            data = item['feat'].to(device)
            labels = item['target'].to(device)
            weak_labels = item['weak_label'].to(device)

            strong_pred, weak_pred = model(data)

            strong_loss = criterion(strong_pred, labels)
            weak_loss = criterion(weak_pred, weak_labels)
            tot_loss = strong_loss + weak_loss

            valid_strong_loss_sum += strong_loss.item()
            valid_weak_loss_sum += weak_loss.item()
            valid_tot_loss_sum += tot_loss.item()

            weak_f1_sum += calc_sed_weak_f1(weak_labels,
                                            weak_pred, sed_eval_thr)

            for i, pred in enumerate(strong_pred):
                pred = pred.to('cpu').detach().numpy().copy()

                for thr in thresholds:
                    result = strong_label_decoding(
                        pred, item['filename'][i], 16000, 1, 320, class_map, thr
                    )
                    results[thr] += result

        sed_evals = calc_sed_eval_metrics(
            meta_strong, pd.DataFrame(results[sed_eval_thr]), 0.1, 0.2
        )

        valid_strong_loss = valid_strong_loss_sum / n_batch
        valid_weak_loss = valid_weak_loss_sum / n_batch
        valid_tot_loss = valid_tot_loss_sum / n_batch
        valid_weak_f1 = weak_f1_sum / n_batch

        # pred_dict = dict(zip(filenames, preds))

    return (
        valid_strong_loss,
        valid_weak_loss,
        valid_tot_loss,
        valid_weak_f1,
        sed_evals,
        # pred_dict
    )