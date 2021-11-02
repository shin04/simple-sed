from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from utils.label_encoder import strong_label_encoding


class StrongDataset(Dataset):
    def __init__(
        self, audio_path: Path, metadata_path: Path, weak_label_path: Path,
        class_map: dict = None,
        sr: int = 44100,
        sample_sec: int = 10,
        frame_hop: int = 256,
        net_pooling_rate: int = 1,
        hubert_feat_path: Path = None,
        transforms: T.Compose = None,
    ) -> None:
        self.audio_path = audio_path

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
        self.sample_len = sr*sample_sec
        self.frame_hop = frame_hop
        self.net_pooling_rate = net_pooling_rate

        self.transforms = transforms

        self.hubert_feat_path = hubert_feat_path
        if hubert_feat_path is not None:
            _feat = np.load(hubert_feat_path / f'{self.filenames[0][:-4]}.npy')
            self.feat_shape = _feat.shape

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        p = self.audio_path / filename

        waveform, _ = sf.read(p)

        if self.sample_len > len(waveform):
            pad_width = (self.sample_len-len(waveform), 0)
            waveform = np.pad(waveform, pad_width,
                              'constant', constant_values=0)
        else:
            waveform = waveform[:self.sample_len]

        waveform = torch.from_numpy(waveform).float()
        if self.transforms is not None:
            waveform = self.transforms(waveform)

        if self.hubert_feat_path is not None:
            feat = self.get_hubert_feat()
            feat = feat.T
            waveform = waveform[:, :499]
            waveform = np.concatenate([waveform, feat])

        label = strong_label_encoding(
            self.sr, self.sample_len, self.frame_hop, self.net_pooling_rate,
            self.meta_df[self.meta_df['filename'] == filename], self.class_map
        )
        label = label[:499, ]

        item = {
            'filename': filename,
            'waveform': waveform,
            'target': torch.from_numpy(label.T).float(),
            'weak_label': torch.from_numpy(self.weak_labels[idx]).float()
        }

        return item

    def get_hubert_feat(self):
        feat = np.zeros(self.feat_shape)
        feat = torch.from_numpy(feat).float()

        return feat


if __name__ == '__main__':
    dataset = StrongDataset(
        audio_path=Path(
            '/home/kajiwara21/dataset/URBAN-SED_v2.0.0/audio/train'),
        metadata_path=Path(
            '/home/kajiwara21/work/sed/meta/train_meta_strong.csv'),
        weak_label_path=Path(
            '/home/kajiwara21/work/sed/meta/train_meta_weak.csv'),
        sr=16000,
        sample_sec=10,
        frame_hop=320,
        net_pooling_rate=1,
    )

    print(len(dataset))
    print(dataset[0]['filename'])
    print(dataset[0]['waveform'].shape)
    print(dataset[0]['target'].shape)
    print(dataset[0]['weak_label'])
