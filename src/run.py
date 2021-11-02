import os
import time
import logging
from pathlib import Path
from datetime import datetime

import hydra
import mlflow
from omegaconf import DictConfig
from dotenv import load_dotenv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as T

from model.crnn import CRNN
from dataset.urban_sed import StrongDataset
from training.train import train, valid
from training.test import test  # , decide_class_threshold
from utils.transformers import (
    GetMelSpectrogram,
    TimeMasking,
    FrequencyMasking,
    # Normalize
)
from utils.callback import EarlyStopping
from utils.param_util import log_params_from_omegaconf_dict

TIME_TEMPLATE = '%Y%m%d%H%M%S'
log = logging.getLogger(__name__)


@hydra.main(config_path='../config', config_name='baseline.yaml')
def run(cfg: DictConfig) -> None:
    base_dir = Path(cfg['base_dir'])

    load_dotenv(verbose=True)
    load_dotenv(base_dir / cfg['environments'])
    tracking_url = os.environ.get('TRACKING_URL')
    ts = datetime.now().strftime(TIME_TEMPLATE)

    """prepare parameters"""
    ex_name = cfg['ex_name']
    device = torch.device(cfg['device'])
    base_dir = Path(cfg['base_dir'])
    log.info(f'start {ex_name} {str(ts)}')

    is_save = cfg['result']['save']
    result_path = base_dir / cfg['result']['vaild_pred_dir']

    audio_path = base_dir / cfg['dataset']['audio_path']
    train_meta = base_dir / cfg['dataset']['train_meta']
    valid_meta = base_dir / cfg['dataset']['valid_meta']
    test_meta = base_dir / cfg['dataset']['test_meta']
    train_weak_label = base_dir / cfg['dataset']['train_weak_label']
    valid_weak_label = base_dir / cfg['dataset']['valid_weak_label']
    test_weak_label = base_dir / cfg['dataset']['test_weak_label']
    test_duration = base_dir / cfg['dataset']['test_duration']
    model_path = base_dir / \
        cfg['model']['save_path'] / f'{ex_name}-{ts}-best.pt'

    sr = cfg['dataset']['sr']
    sample_sec = cfg['dataset']['sec']
    hop_length = cfg['feature']['hop_length']
    net_pooling_rate = cfg['dataset']['net_pooling_rate']

    n_epoch = cfg['training']['n_epoch']
    batch_size = cfg['training']['batch_size']
    lr = cfg['training']['lr']
    num_workers = cfg['training']['num_workers']
    pin_memory = cfg['training']['pin_memory']

    es_patience = cfg['training']['early_stop_patience']
    thresholds = cfg['training']['thresholds']

    psds_params = cfg['evaluate']['psds']

    """prepare datasets"""
    get_melspec = GetMelSpectrogram(sr=sr, **cfg['feature'], log_scale=True)
    time_mask = TimeMasking(**cfg['augmentation']['time_masking'])
    freq_mask = FrequencyMasking(**cfg['augmentation']['freq_masking'])
    # normalzie = Normalize(mode='min_max')
    transforms = T.Compose([
        get_melspec,
        T.RandomApply([
            time_mask,
            freq_mask
        ], p=0.5),
        # normalzie
    ])

    train_dataset = StrongDataset(
        audio_path=audio_path / 'train-16k',
        metadata_path=train_meta,
        weak_label_path=train_weak_label,
        sr=sr,
        sample_sec=sample_sec,
        frame_hop=hop_length,
        net_pooling_rate=net_pooling_rate,
        hubert_feat_path=Path(cfg['dataset']['feat_path']/'train'),
        transforms=transforms
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )

    valid_dataset = StrongDataset(
        audio_path=audio_path / 'validate-16k',
        metadata_path=valid_meta,
        weak_label_path=valid_weak_label,
        sr=sr,
        frame_hop=hop_length,
        sample_sec=sample_sec,
        net_pooling_rate=net_pooling_rate,
        hubert_feat_path=Path(cfg['dataset']['feat_path']/'valid'),
        transforms=T.Compose([get_melspec])
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    if train_dataset.classes != valid_dataset.classes:
        raise RuntimeError("class_map is wrong")

    """prepare training"""
    model = CRNN(
        **cfg['model']['dence'],
        cnn_cfg=dict(cfg['model']['cnn']),
        rnn_cfg=dict(cfg['model']['rnn']),
        attention=True,
        layer_init=cfg['model']['initialize']
    ).to(device)
    early_stopping = EarlyStopping(patience=es_patience)
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=cfg['training']['weight_decay'],
        amsgrad=False
    )
    if cfg['training']['scheduler']:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    else:
        scheduler = None
    criterion = nn.BCELoss()

    """training and validation"""
    best_loss = 10000
    mlflow.set_tracking_uri(tracking_url)
    mlflow.set_experiment(ex_name)
    with mlflow.start_run(run_name=str(ts)):
        log_params_from_omegaconf_dict(dict(cfg))
        for epoch in range(n_epoch):
            start = time.time()

            train_strong_loss, train_weak_loss, train_tot_loss, used_lr = train(
                model, train_dataloader, device, optimizer, criterion, scheduler
            )

            (
                valid_strong_loss, valid_weak_loss, valid_tot_loss,
                valid_weak_f1, valid_sed_evals, pred_dict
            ) = valid(
                model, valid_dataloader, device, criterion,
                valid_dataset.class_map, thresholds,
                cfg['evaluate']['median_filter'], cfg['training']['sed_eval_thr'],
                valid_meta,
                sr, hop_length, net_pooling_rate
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

            log.info(
                f'[EPOCH {epoch}/{n_epoch}]({time.time()-start: .1f}sec) lr={used_lr}')
            log.info(
                f'[TRAIN] train loss(strong):{train_strong_loss: .4f}, ' +
                f'train loss(weak):{train_weak_loss: .4f}, ' +
                f'train loss(total):{train_tot_loss: .4f}'
            )
            log.info(
                f'[VALID] valid loss(strong):{valid_strong_loss: .4f}, ' +
                f'valid loss(weak):{valid_weak_loss: .4f}, ' +
                f'valid loss(total):{valid_tot_loss: .4f}'
            )
            log.info(
                f'[VALID F1(segment)] segment/class_wise_f1:{valid_sed_evals["segment"]["class_wise_f1"]: .4f} ' +
                f'segment/overall_f1:{valid_sed_evals["segment"]["overall_f1"]: .4f}'
            )
            log.info(
                f'[VALID F1(event)] event/class_wise_f1:{valid_sed_evals["event"]["class_wise_f1"]: .4f} ' +
                f'event/overall_f1:{valid_sed_evals["event"]["overall_f1"]: .4f}'
            )
            if len(valid_sed_evals['event']['detail']) != 0:
                for event in list(valid_dataset.class_map.keys()):
                    log.info(
                        f'{event} : {valid_sed_evals["event"]["detail"][event]: .4f}')

            if best_loss > valid_tot_loss:
                best_loss = valid_tot_loss

                if is_save:
                    np.save(
                        result_path / f'{ex_name}-{ts}-valid.npy', pred_dict
                    )

                with open(model_path, 'wb') as f:
                    torch.save(model.state_dict(), f)
                log.info(f'update best model (loss: {best_loss})')

            log.info(f'best loss: {best_loss}')
            mlflow.log_metric('valid/best_loss', best_loss, step=epoch)

            early_stopping(valid_tot_loss)
            if early_stopping.early_stop:
                log.info('Early Stopping')
                break

        """test step"""
        log.info("start evaluate ...")
        test_dataset = StrongDataset(
            audio_path=audio_path / 'test-16k',
            metadata_path=test_meta,
            weak_label_path=test_weak_label,
            sr=sr,
            frame_hop=hop_length,
            sample_sec=sample_sec,
            net_pooling_rate=net_pooling_rate,
            hubert_feat_path=Path(cfg['dataset']['feat_path']/'test'),
            transforms=T.Compose([get_melspec])
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory
        )

        model = CRNN(
            **cfg['model']['dence'],
            cnn_cfg=dict(cfg['model']['cnn']),
            rnn_cfg=dict(cfg['model']['rnn']),
            attention=True
        ).to(device)
        model.load_state_dict(torch.load(model_path))

        # best_th = decide_class_threshold(
        #     result_path / f'{ex_name}-valid.npy', valid_meta, sr, hop_length, net_pooling_rate,
        #     valid_dataset.class_map
        # )
        # log.info('best valid thresholds', best_th)
        (
            test_psds_eval_list,
            test_psds_macro_f1_list,
            test_weak_f1,
            test_sed_evals,
            test_pred_dict
        ) = test(
            model, test_dataloader, device, test_dataset.class_map,
            cfg['evaluate']['thresholds'], cfg['evaluate']['median_filter'], cfg['training']['sed_eval_thr'],
            psds_params, test_meta, test_duration,
            sr, hop_length, net_pooling_rate,
            # best_th
            {}
        )

        log.info('[TEST EVAL]')
        log.info(f'weak_f1:{test_weak_f1: .4f}')
        log.info(
            f'segment/class_wise_f1:{test_sed_evals["segment"]["class_wise_f1"]: .4f} ' +
            f'segment/overall_f1:{test_sed_evals["segment"]["overall_f1"]: .4f}'
        )
        log.info(
            f'event/class_wise_f1:{test_sed_evals["event"]["class_wise_f1"]: .4f} ' +
            f'event/overall_f1:{test_sed_evals["event"]["overall_f1"]: .4f}'
        )

        for i in range(cfg['evaluate']['psds']['val_num']):
            score = test_psds_eval_list[i]
            f1 = test_psds_macro_f1_list[i]
            log.info(
                f'psds score ({i}):{score: .4f}, macro f1 ({i}):{f1: .4f}')

        if is_save:
            p = result_path / f'{ex_name}-{ts}-test.npy'
            np.save(p, test_pred_dict)
            log.info(f'saved predict at {p}')

        if not cfg['model']['save']:
            try:
                model_path.unlink()
                log.info(f'deleted model at {model_path}')
            except Exception as e:
                log.info(f'failed deleting model at {model_path}')
                log.error(e)

        mlflow.log_metric('test/weak/f1', test_weak_f1, step=epoch)
        mlflow.log_metric('test/sed_eval/segment/class_wise_f1',
                          test_sed_evals['segment']['class_wise_f1'], step=epoch)
        mlflow.log_metric('test/sed_eval/segment/overall_f1',
                          test_sed_evals['segment']['overall_f1'], step=epoch)
        mlflow.log_metric('test/sed_eval/event/class_wise_f1',
                          test_sed_evals['event']['class_wise_f1'], step=epoch)
        mlflow.log_metric('test/sed_eval/event/overall_f1',
                          test_sed_evals['event']['overall_f1'], step=epoch)

        mlflow.log_artifact('.hydra/config.yaml')
        mlflow.log_artifact('.hydra/hydra.yaml')
        mlflow.log_artifact('.hydra/overrides.yaml')
        mlflow.log_artifact(f'{__file__[:-3]}.log')

    log.info(f'ex "{str(ts)}" complete !!')

    return best_loss


if __name__ == '__main__':
    run()
