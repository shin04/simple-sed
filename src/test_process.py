import yaml
from pathlib import Path
from typing import List, Callable
import argparse

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
    model.load_state_dict(torch.load(model_path))

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

    print(
        '===============\n'
        '[test EVAL]\n'
        f'weak_f1:{test_weak_f1: .4f}\n',
        f'segment/class_wise_f1:{test_sed_evals["segment"]["class_wise_f1"]: .4f}\n',
        f'segment/overall_f1:{test_sed_evals["segment"]["overall_f1"]: .4f}\n',
        f'event/class_wise_f1:{test_sed_evals["event"]["class_wise_f1"]: .4f}\n',
        f'event/overall_f1:{test_sed_evals["event"]["overall_f1"]: .4f}\n',
    )

    for i in range(psds_params['val_num']):
        score = test_psds_eval_list[i]
        f1 = test_psds_macro_f1_list[i]
        print(
            f'psds score ({i}):{score: .4f}, '
            f'macro f1 ({i}):{f1: .4f}'
        )

    if save_path is not None:
        np.save(save_path, test_pred_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('model_path')
    parser.add_argument('feat_path')
    parser.add_argument('-g', '--gpu', action='store_true')
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
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if args.model == 'crnn':
        dataset = StrongDataset(
            feat_path=feat_path/'test',
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
        dataset = HuBERTDataset(
            feat_pathes=[feat_path/'test'],
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
            n_feats=1
        )

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
