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

from model.hucrnn import HuCRNN
from dataset.hubert_feat import HuBERTDataset
from training.hubert_train import train, valid
from training.hubert_test import test
from utils.callback import EarlyStopping
from utils.param_util import log_params_from_omegaconf_dict

TIME_TEMPLATE = '%Y%m%d%H%M%S'
log = logging.getLogger(__name__)


@hydra.main(config_path='../config', config_name='hubert.yaml')
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

    feat_pathes = [base_dir / p for p in cfg['dataset']['feat_pathes']]
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
    sec = cfg['dataset']['sec']
    net_pooling_rate = cfg['dataset']['net_pooling_rate']
    data_percentage = cfg['dataset']['percentage']

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

    psds_params = cfg['evaluate']['psds']

    """prepare datasets"""
    train_dataset = HuBERTDataset(
        feat_pathes=[p / 'train' for p in feat_pathes],
        metadata_path=train_meta,
        weak_label_path=train_weak_label,
        sr=sr,
        sec=sec,
        net_pooling_rate=net_pooling_rate,
        transforms=None,
        percentage=data_percentage,
    )
    valid_dataset = HuBERTDataset(
        feat_pathes=[p / 'valid' for p in feat_pathes],
        metadata_path=valid_meta,
        weak_label_path=valid_weak_label,
        sr=sr,
        sec=sec,
        net_pooling_rate=net_pooling_rate,
        transforms=None,
    )
    test_dataset = HuBERTDataset(
        feat_pathes=[p / 'test' for p in feat_pathes],
        metadata_path=test_meta,
        weak_label_path=test_weak_label,
        sr=sr,
        sec=sec,
        net_pooling_rate=net_pooling_rate,
        transforms=None,
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    if train_dataset.classes != valid_dataset.classes:
        raise RuntimeError("class_map is wrong")

    """prepare training"""
    model = HuCRNN(
        **cfg['model']['dence'],
        cnn_cfg=dict(cfg['model']['cnn']),
        rnn_cfg=dict(cfg['model']['rnn']),
        attention=True,
        n_feats=len(feat_pathes)
    ).to(device)
    if cfg['device'] == 'cuda':
        model = torch.nn.DataParallel(model)

    early_stopping = EarlyStopping(patience=es_patience)
    optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=False)
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
                valid_strong_loss,
                valid_weak_loss,
                valid_tot_loss,
                valid_weak_f1,
                valid_sed_evals
            ) = valid(
                model, valid_dataloader, device, criterion,
                valid_dataset.class_map,
                cfg['training']['thresholds'],
                cfg['training']['sed_eval_thr'],
                valid_meta
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
            if len(valid_sed_evals['event']['by_event']) != 0:
                for event in list(valid_dataset.class_map.keys()):
                    log.info(
                        f'{event}: ' +
                        f'segment based f1 {valid_sed_evals["segment"]["by_event"][event]["f_measure"]: .4f}, ' +
                        f'event based f1 {valid_sed_evals["event"]["by_event"][event]["f_measure"]: .4f}'
                    )

            if best_loss > valid_tot_loss:
                best_loss = valid_tot_loss

                with open(model_path, 'wb') as f:
                    if cfg['device'] == 'cuda':
                        torch.save(model.module.state_dict(), f)
                    else:
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

        model = HuCRNN(
            **cfg['model']['dence'],
            cnn_cfg=dict(cfg['model']['cnn']),
            rnn_cfg=dict(cfg['model']['rnn']),
            attention=True,
            n_feats=len(feat_pathes)
        ).to(device)
        model.load_state_dict(torch.load(model_path))
        if cfg['device'] == 'cuda':
            model = torch.nn.DataParallel(model)

        (
            test_psds_eval_list,
            test_psds_macro_f1_list,
            test_weak_f1,
            test_sed_evals,
            test_pred_dict
        ) = test(
            model, test_dataloader, device, test_dataset.class_map,
            cfg['evaluate']['thresholds'], cfg['training']['sed_eval_thr'],
            psds_params, test_meta, test_duration,
            sr, 1, net_pooling_rate, {}
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
        if len(test_sed_evals['event']['by_event']) != 0:
            for event in list(test_dataset.class_map.keys()):
                log.info(
                    f'{event}: ' +
                    f'segment based f1 {test_sed_evals["segment"]["by_event"][event]["f_measure"]: .4f}, ' +
                    f'event based f1 {test_sed_evals["event"]["by_event"][event]["f_measure"]: .4f}'
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
