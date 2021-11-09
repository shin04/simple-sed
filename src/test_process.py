import os
import yaml
from pathlib import Path
from typing import List, Callable
import argparse
from collections import OrderedDict
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from training.test import test as normal_test_fn
from training.hubert_test import test as hubert_test_fn
from model.crnn import CRNN
from model.hucrnn import HuCRNN
from dataset.urban_sed import StrongDataset
from dataset.hubert_feat import HuBERTDataset

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
)
log = logging.getLogger(__name__)


def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict


def test_process(
    device: torch.device,
    model: nn.Module,
    model_path: Path,
    dataset: Dataset,
    dataloader: DataLoader,
    test_fn: Callable,
    thresholds: List[float],
    sed_eval_thr: float,
    psds_params: dict,
    test_meta: Path,
    test_duration: Path,
    sr: int,
    net_pooling_rate: int,
    save_path: Path
):
    state_dict = torch.load(model_path)
    model.load_state_dict(fix_model_state_dict(state_dict))

    (
        test_psds_eval_list,
        test_psds_macro_f1_list,
        test_weak_f1,
        test_sed_evals,
        test_pred_dict
    ) = test_fn(
        model, dataloader, device, dataset.class_map,
        thresholds, sed_eval_thr, psds_params, test_meta, test_duration,
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
        for event in list(dataset.class_map.keys()):
            log.info(f'{event}: ')
            log.info(
                f'segment based f1: {test_sed_evals["segment"]["by_event"][event]["f_measure"]: .4f}, ' +
                f'event based f1: {test_sed_evals["event"]["by_event"][event]["f_measure"]: .4f}'
            )

    for i in range(psds_params['val_num']):
        score = test_psds_eval_list[i]
        f1 = test_psds_macro_f1_list[i]
        log.info(
            f'psds score ({i}):{score: .4f}, macro f1 ({i}):{f1: .4f}'
        )

    if save_path is not None:
        np.save(save_path, test_pred_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('model_path')
    parser.add_argument('feat_path')
    parser.add_argument('-g', '--gpu', action='store_true')
    parser.add_argument('-a', '--all', action='store_true')
    parser.add_argument('-s', '--save_path')
    args = parser.parse_args()

    model_path = Path(args.model_path)
    feat_path = Path(args.feat_path)
    test_meta = Path('/home/kajiwara21/work/sed/meta/test_meta_strong.csv')
    test_weak_label = Path('/home/kajiwara21/work/sed/meta/test_meta_weak.csv')
    test_duration = Path(
        '/home/kajiwara21/work/sed/meta/test_meta_duration.csv')

    if args.save_path is not None:
        save_path = Path(args.save_path)
    else:
        save_path = None

    with open('../config/hubert.yaml') as f:
        cfg = yaml.load(f)

    if args.gpu:
        print('using gpu')
        device = torch.device('cuda')
    else:
        print('using cpu')
        device = torch.device('cpu')

    if args.model == 'crnn':
        dataset = StrongDataset(
            audio_path=feat_path/'test',
            metadata_path=test_meta,
            weak_label_path=test_weak_label,
            sr=cfg['dataset']['sr'],
            sec=cfg['dataset']['sec'],
            net_pooling_rate=cfg['dataset']['net_pooling_rate'],
            transforms=None
        )

        model = CRNN(
            **cfg['model']['dence'],
            cnn_cfg=dict(cfg['model']['cnn']),
            rnn_cfg=dict(cfg['model']['rnn']),
            attention=True
        ).to(device)

        test_fn = normal_test_fn
    elif args.model == 'hucrnn':
        if args.all:
            feat_pathes = [feat_path /
                           f'ite2_layer_{i+1}/test' for i in range(12)]
            n_feats = 12
        else:
            feat_pathes = [feat_path/'test']
            n_feats = 1

        dataset = HuBERTDataset(
            feat_pathes=feat_pathes,
            metadata_path=test_meta,
            weak_label_path=test_weak_label,
            sr=cfg['dataset']['sr'],
            sec=cfg['dataset']['sec'],
            net_pooling_rate=cfg['dataset']['net_pooling_rate'],
            transforms=None
        )

        model = HuCRNN(
            **cfg['model']['dence'],
            cnn_cfg=dict(cfg['model']['cnn']),
            rnn_cfg=dict(cfg['model']['rnn']),
            attention=True,
            n_feats=n_feats
        ).to(device)

        test_fn = hubert_test_fn
    else:
        raise RuntimeError(f'{args.model} is not defined')

    dataloader = DataLoader(
        dataset, batch_size=cfg['training']['batch_size'], shuffle=False,
        num_workers=cfg['training']['batch_size'], pin_memory=True
    )

    test_process(
        device,
        model,
        model_path,
        dataset,
        dataloader,
        test_fn,
        cfg['evaluate']['thresholds'],
        cfg['training']['sed_eval_thr'],
        cfg['evaluate']['psds'],
        test_meta,
        test_duration,
        cfg['dataset']['sr'],
        cfg['dataset']['net_pooling_rate'],
        save_path
    )
