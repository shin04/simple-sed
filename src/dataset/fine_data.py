from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from utils.label_encoder import strong_label_encoding


class FineDataset(Dataset):
    def __init__(
        self,
        audio_path: Path,
        metadata_path: Path,
        weak_label_path: Path,
        class_map: dict = None,
        sr: int = 44100,
        sec: int = 10,
        net_pooling_rate: int = 1
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
        self.sec = sec
        self.net_pooling_rate = net_pooling_rate

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        p = self.audio_path / filename

        waveform, _ = sf.read(p)

        if self.sr*self.sec > len(waveform):
            pad_width = (self.sr*self.sec-len(waveform), 0)
            waveform = np.pad(waveform, pad_width,
                              'constant', constant_values=0)
        else:
            waveform = waveform[:self.sr*self.sec]

        waveform = torch.from_numpy(waveform).float()

        label = strong_label_encoding(
            self.sr, self.sr*self.sec, 1, self.net_pooling_rate,
            self.meta_df[self.meta_df['filename'] == filename], self.class_map
        )
        label = label[:499, ]
        label = torch.from_numpy(label.T).float()

        item = {
            'filename': filename,
            'waveform': waveform,
            'target': label,
            'weak_label': torch.from_numpy(self.weak_labels[idx]).float()
        }

        return item


if __name__ == '__main__':
    dataset = FineDataset(
        audio_path=Path(
            '/home/kajiwara21/dataset/URBAN-SED_v2.0.0/audio/train-16k'),
        metadata_path=Path(
            '/home/kajiwara21/work/sed/meta/train_meta_strong.csv'),
        weak_label_path=Path(
            '/home/kajiwara21/work/sed/meta/train_meta_weak.csv'),
        sr=16000,
        sec=10,
        net_pooling_rate=1,
        percentage=0.5,
    )

    print(len(dataset))
    print(dataset[0]['filename'])
    print(dataset[0]['waveform'].shape)
    print(dataset[0]['target'].shape)
    print(dataset[0]['weak_label'])
