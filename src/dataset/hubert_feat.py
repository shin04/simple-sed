from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from utils.label_encoder import strong_label_encoding


class HuBERTDataset(Dataset):
    def __init__(
        self, feat_path: Path, metadata_path: Path, weak_label_path: Path,
        class_map: dict = None,
        net_pooling_rate: int = 1,
        transforms: T.Compose = None,
    ) -> None:
        self.feat_path = feat_path

        self.meta_df = pd.read_csv(metadata_path)
        self.filenames = self.meta_df['filename'].unique().tolist()
        self.classes = sorted(self.meta_df['event_label'].unique().tolist())
        if class_map is not None:
            self.class_map = class_map
        else:
            self.class_map = {c: i for i, c in enumerate(self.classes)}

        weak_label_df = pd.read_csv(weak_label_path, index_col=0)
        self.weak_labels = weak_label_df[self.class_map.keys()].values

        self.net_pooling_rate = net_pooling_rate

        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        # p = self.feat_path / f'{filename[:-4]}.npy'
        p = self.feat_path / f'{filename}.npy'

        feat = np.load(p)

        feat = torch.from_numpy(feat).float()
        if self.transforms is not None:
            feat = self.transforms(feat)

        label = strong_label_encoding(
            16000, 16000*10, 1, 320,
            self.meta_df[self.meta_df['filename'] == filename], self.class_map
        )
        label = label[:499, ]

        item = {
            'filename': filename,
            'feat': feat.T,
            'target': torch.from_numpy(label.T).float(),
            'weak_label': torch.from_numpy(self.weak_labels[idx]).float()
        }

        return item


if __name__ == '__main__':
    dataset = HuBERTDataset(
        feat_path=Path(
            '/home/kajiwara21/dataset/hubert_feat/urbansed/sample-train'),
        # '/home/kajiwara21/mrnas02/home/datasets/hubert_feat/sample-train'),
        metadata_path=Path(
            '/home/kajiwara21/work/sed/meta/train_meta_strong.csv'),
        weak_label_path=Path(
            '/home/kajiwara21/work/sed/meta/train_meta_weak.csv'),
        net_pooling_rate=320
    )

    print(len(dataset))
    print(dataset[0]['filename'])
    print(dataset[0]['feat'].shape)
    print(dataset[0]['target'].shape)
    print(dataset[0]['weak_label'])
