import os
import time
from pathlib import Path
from datetime import datetime

import hydra
import mlflow
from omegaconf import DictConfig
# import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from model.crnn import CRNN
from dataset.hubert_feat import HuBERTDataset
from training.hubert_train import train
from utils.callback import EarlyStopping
from utils.param_util import log_params_from_omegaconf_dict

TIME_TEMPLATE = '%Y%m%d%H%M%S'


@hydra.main(config_path='../config', config_name='hubert.yaml')
def run(cfg: DictConfig) -> None:
    ts = datetime.now().strftime(TIME_TEMPLATE)

    """prepare parameters"""
    ex_name = cfg['ex_name']
    device = torch.device(cfg['device'])
    print(f'start {ex_name} {str(ts)}')

    # result_path = Path(cfg['result']['vaild_pred_dir'])

    feat_path = Path(cfg['dataset']['feat_path'])
    train_meta = Path(cfg['dataset']['train_meta'])
    # valid_meta = Path(cfg['dataset']['valid_meta'])
    # test_meta = Path(cfg['dataset']['test_meta'])
    train_weak_label = Path(cfg['dataset']['train_weak_label'])
    # valid_weak_label = Path(cfg['dataset']['valid_weak_label'])
    # test_weak_label = Path(cfg['dataset']['test_weak_label'])
    # train_duration = Path(cfg['dataset']['train_duration'])
    # valid_duration = Path(cfg['dataset']['valid_duration'])
    # test_duration = Path(cfg['dataset']['test_duration'])
    model_path = Path(cfg['model']['save_path']) / f'{ex_name}-{ts}-best.pt'

    net_pooling_rate = cfg['dataset']['net_pooling_rate']

    n_epoch = cfg['training']['n_epoch']
    batch_size = cfg['training']['batch_size']
    lr = cfg['training']['lr']
    num_workers = cfg['training']['num_workers']
    if num_workers == 0:
        pin_memory = False
    elif num_workers == -1:
        num_workers = os.cpu_count()
        pin_memory = True
    else:
        pin_memory = True
    es_patience = cfg['training']['early_stop_patience']

    """prepare datasets"""
    train_dataset = HuBERTDataset(
        feat_path=feat_path,
        metadata_path=train_meta,
        weak_label_path=train_weak_label,
        net_pooling_rate=net_pooling_rate,
        transforms=None
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )

    """prepare training"""
    model = CRNN(
        **cfg['model']['dence'],
        cnn_cfg=dict(cfg['model']['cnn']),
        rnn_cfg=dict(cfg['model']['rnn']),
        attention=True
    ).to(device)
    early_stopping = EarlyStopping(patience=es_patience)
    optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=False)
    criterion = nn.BCELoss()

    """training and validation"""
    best_loss = 10000
    global_step = 0
    mlflow.set_tracking_uri(
        "file://" + hydra.utils.get_original_cwd() + "/../log/mlruns")
    mlflow.set_experiment(ex_name)
    with mlflow.start_run(run_name=str(ts)):
        log_params_from_omegaconf_dict(dict(cfg))
        for epoch in range(n_epoch):
            start = time.time()

            train_strong_loss, train_weak_loss, train_tot_loss = train(
                global_step, model, train_dataloader, device, optimizer, criterion
            )

            mlflow.log_metric('train/strong/loss',
                              train_strong_loss, step=epoch)
            mlflow.log_metric('train/weak/loss', train_weak_loss, step=epoch)
            mlflow.log_metric('train/tot/loss', train_tot_loss, step=epoch)

            print(
                '====================\n'
                f'[EPOCH {epoch}/{n_epoch}]({time.time()-start: .1f}sec) '
            )
            print(
                '[TRAIN]\n',
                f'train loss(strong):{train_strong_loss: .4f}, '
                f'train loss(weak):{train_weak_loss: .4f}, '
                f'train loss(total):{train_tot_loss: .4f}'
            )

            if best_loss > train_tot_loss:
                best_loss = train_tot_loss
                with open(model_path, 'wb') as f:
                    torch.save(model.state_dict(), f)
                print(f'update best model (loss: {best_loss})')

            early_stopping(train_tot_loss)
            if early_stopping.early_stop:
                print('Early Stopping')
                break

            global_step += len(train_dataset)

    print(f'ex "{str(ts)}" complete !!')


if __name__ == '__main__':
    run()
