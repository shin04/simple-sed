from __future__ import annotations
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from utils.label_encoder import strong_label_encoding


class HuBERTDataset(Dataset):
    def __init__(
        self,
        feat_pathes: list[Path],
        metadata_path: Path,
        weak_label_path: Path,
        class_map: dict = None,
        sr: int = 16000,
        sec: int = 10,
        net_pooling_rate: int = 320,
        transforms: T.Compose = None,
        percentage: float = None,
    ) -> None:

        if percentage is not None:
            if percentage > 1 and percentage < 0:
                raise RuntimeError(f'argument "percentage" value "{percentage}" is invalid.')

        self.feat_pathes = feat_pathes
        self.n_layers = len(feat_pathes)

        self.meta_df = pd.read_csv(metadata_path)
        self.filenames = self.meta_df['filename'].unique().tolist()
        self.classes = sorted(self.meta_df['event_label'].unique().tolist())
        if class_map is not None:
            self.class_map = class_map
        else:
            self.class_map = {c: i for i, c in enumerate(self.classes)}

        weak_label_df = pd.read_csv(weak_label_path, index_col=0)
        self.weak_labels = weak_label_df[self.class_map.keys()].values

        self.sr = sr
        self.sec = sec
        self.net_pooling_rate = net_pooling_rate

        self.transforms = transforms

        _filename = self.filenames[0]
        _feat = np.load(feat_pathes[0] / f'{_filename[:-4]}.npy')
        self.shape = _feat.shape

        if percentage is not None and percentage != 1.0:
            n_samples = int(len(self.filenames) * percentage)
            indexes = [i for i in range(len(self.filenames))]
            indexes = random.sample(indexes, n_samples)

            self.filenames = [f for i, f in enumerate(self.filenames) if i in indexes]
            self.weak_labels = [l for i, l in enumerate(self.weak_labels) if i in indexes]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        # feat (n_layers, n_frames, 768)
        feats = np.zeros((self.n_layers, self.shape[0], self.shape[1]))
        for i, feat_path in enumerate(self.feat_pathes):
            p = feat_path / f'{filename[:-4]}.npy'
            feats[i] = np.load(p)

        feats = torch.from_numpy(feats).float()
        # (n_layers, n_frames, 768) -> (n_frames, 768, n_layers)
        feats = feats.permute(1, 2, 0)

        if self.transforms is not None:
            feats = self.transforms(feats)

        label = strong_label_encoding(
            self.sr, self.sr*self.sec, 1, self.net_pooling_rate,
            self.meta_df[self.meta_df['filename'] == filename], self.class_map
        )
        # FIXME: fix label encoding for hubert feature
        label = label[:499, ]

        item = {
            'filename': filename,
            'feat': feats,
            'target': torch.from_numpy(label.T).float(),
            'weak_label': torch.from_numpy(self.weak_labels[idx]).float()
        }

        return item


if __name__ == '__main__':
    dataset = HuBERTDataset(
        feat_pathes=[
            Path('/home/kajiwara21/nas02/home/dataset/hubert_feat/urbansed_audioset/nmf/ite2_layer_12/train'),
        ],
        metadata_path=Path(
            '/home/kajiwara21/work/sed/meta/train_meta_strong.csv'),
        weak_label_path=Path(
            '/home/kajiwara21/work/sed/meta/train_meta_weak.csv'),
        sr=16000,
        sec=10,
        net_pooling_rate=320,
        percentage=1.0,
    )

    print(len(dataset))
    data = dataset[0]
    print(data['filename'])
    print(data['feat'].shape)
    print(data['target'].shape)
    print(data['weak_label'])

    # print(np.where(data['target'] == 1.0))
