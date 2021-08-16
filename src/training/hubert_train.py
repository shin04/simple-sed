import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow


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
