import os
import time
from pathlib import Path

import hydra
import mlflow
from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as T

from model.crnn import CRNN
from dataset.urban_sed import StrongDataset
from training.train import train, valid
from utils.augmentation import GaussianNoise
from utils.callback import EarlyStopping
from utils.param_util import log_params_from_omegaconf_dict


@hydra.main(config_path='../config', config_name='urban_sed.yaml')
def run(cfg: DictConfig) -> None:
    """prepare parameters"""
    ex_name = cfg['ex_name']
    device = torch.device(cfg['device'])

    audio_path = Path(cfg['dataset']['audio_path'])
    train_meta = Path(cfg['dataset']['train_meta'])
    valid_meta = Path(cfg['dataset']['valid_meta'])
    # test_meta = Path(cfg['dataset']['test_meta'])
    train_weak_label = Path(cfg['dataset']['train_weak_label'])
    valid_weak_label = Path(cfg['dataset']['valid_weak_label'])
    # test_weak_label = Path(cfg['dataset']['test_weak_label'])
    model_path = Path(cfg['model']['save_path'])

    sr = cfg['dataset']['sr']
    sample_sec = cfg['dataset']['sec']

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
    thresholds = cfg['training']['thresholds']

    psds_params = cfg['validation']['psds']

    transforms = T.Compose([GaussianNoise()])

    train_dataset = StrongDataset(
        audio_path=audio_path / 'train',
        metadata_path=train_meta,
        weak_label_path=train_weak_label,
        sr=sr,
        sample_sec=sample_sec,
        transforms=transforms
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )

    valid_dataset = StrongDataset(
        audio_path=audio_path / 'validate',
        metadata_path=valid_meta,
        weak_label_path=valid_weak_label,
        sr=sr,
        sample_sec=sample_sec,
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    if train_dataset.classes != valid_dataset.classes:
        raise RuntimeError("class_map is wrong")

    model = CRNN(
        sr=sr,
        **cfg['feature'],
        **cfg['model']['dence'],
        cnn_cfg=dict(cfg['model']['cnn']),
        rnn_cfg=dict(cfg['model']['rnn']),
    ).to(device)
    early_stopping = EarlyStopping(patience=es_patience)
    optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=False)
    criterion = nn.BCELoss()

    best_loss = 10000
    global_step = 0
    mlflow.set_tracking_uri(
        "file://" + hydra.utils.get_original_cwd() + "/../log/mlruns")
    mlflow.set_experiment(ex_name)
    with mlflow.start_run():
        log_params_from_omegaconf_dict(dict(cfg))
        for epoch in range(n_epoch):
            start = time.time()

            train_strong_loss, train_weak_loss, train_tot_loss = train(
                global_step, model, train_dataloader, device, optimizer, criterion
            )
            (
                valid_strong_loss,
                valid_weak_loss,
                valid_tot_loss,
                psds_score_list,
                psds_macro_f1_list,
                valid_weak_f1,
                valid_sed_evals
            ) = valid(
                model, valid_dataloader, device, criterion,
                valid_dataset.class_map, thresholds, psds_params
            )

            mlflow.log_metric('train/strong/loss',
                              train_strong_loss, step=epoch)
            mlflow.log_metric('train/weak/loss', train_weak_loss, step=epoch)
            mlflow.log_metric('train/tot/loss', train_tot_loss, step=epoch)
            mlflow.log_metric('valid/strong/loss',
                              valid_strong_loss, step=epoch)
            mlflow.log_metric('valid/weak/loss', valid_weak_loss, step=epoch)
            mlflow.log_metric('valid/tot/loss', valid_tot_loss, step=epoch)
            mlflow.log_metric('valid/weak/f1', valid_weak_f1, step=epoch)
            mlflow.log_metric('valid/sed_eval/segment/class_wise_f1',
                              valid_sed_evals['segment']['class_wise_f1'], step=epoch)
            mlflow.log_metric('valid/sed_eval/segment/overall_f1',
                              valid_sed_evals['segment']['overall_f1'], step=epoch)
            mlflow.log_metric('valid/sed_eval/event/class_wise_f1',
                              valid_sed_evals['event']['class_wise_f1'], step=epoch)
            mlflow.log_metric('valid/sed_eval/event/overall_f1',
                              valid_sed_evals['event']['overall_f1'], step=epoch)

            print(
                f'[EPOCH {epoch}/{n_epoch}]({time.time() - start: .1f}sec) '
                f'=========='
            )
            print(
                f'[TRAIN]\n',
                f'train loss(strong):{train_strong_loss: .4f}, '
                f'train loss(weak):{train_weak_loss: .4f}, '
                f'train loss(total):{train_tot_loss: .4f}'
            )
            print(
                f'[VALID]\n'
                f'valid loss(strong):{valid_strong_loss: .4f}, '
                f'valid loss(weak):{valid_weak_loss: .4f}, '
                f'valid loss(total):{valid_tot_loss: .4f}'
            )
            print(
                f'[VALID SED EVAL]n'
                f'segment/class_wise_f1: {valid_sed_evals["segment"]["class_wise_f1"]}',
                f'segment/overall_f1: {valid_sed_evals["segment"]["overall_f1"]}',
                f'event/class_wise_f1: {valid_sed_evals["event"]["class_wise_f1"]}',
                f'event/overall_f1: {valid_sed_evals["event"]["overall_f1"]}',
            )

            for score, f1 in zip(psds_score_list, psds_macro_f1_list):
                print(
                    f'psds score: {score: .4f}, '
                    f'macro f1: {f1[0]: .4f}'
                )
                mlflow.log_metric('psds_score', score, step=epoch)
                mlflow.log_metric('psds_macro_f1', f1[0], step=epoch)

            if best_loss > valid_tot_loss:
                best_loss = valid_tot_loss
                with open(model_path / f'{ex_name}-best.pt', 'wb') as f:
                    torch.save(model.state_dict(), f)
                print(f'update best model (loss: {best_loss})')

            early_stopping(valid_tot_loss)
            if early_stopping.early_stop:
                print('Early Stopping')
                break

            global_step += len(train_dataset)


if __name__ == '__main__':
    run()
