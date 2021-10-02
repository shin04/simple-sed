from pathlib import Path
from typing import Union

from tqdm import tqdm
import numpy as np
import pandas as pd
import soundfile as sf


def load_hubert_feature(feat_path: Path) -> np.ndarray:
    print('loading hubert feature ...')
    if not feat_path.exists():
        raise FileNotFoundError(f'{feat_path} is not found')

    features = {}
    feat_pathes = list(feat_path.glob('*.npy'))
    for p in tqdm(feat_pathes):
        feat = np.load(p)
        features[f'{p.stem}.wav'] = feat

    return features


def load_predict(pred_path: Path) -> np.ndarray:
    if not pred_path.exists():
        raise FileNotFoundError(f'{pred_path} is not found')

    pred_dict = np.load(pred_path, allow_pickle=True).item()

    return pred_dict  # (n_preds, n_frames, events)


def load_audio(audio_path: Path) -> Union[np.ndarray, int]:
    if not audio_path.exists():
        raise FileNotFoundError(f'{audio_path} is not found')

    waveform, sr = sf.read(audio_path)

    return waveform, sr


def load_weak_label(weak_path: Path) -> pd.DataFrame:
    if not weak_path.exists():
        raise FileNotFoundError(f'{weak_path} is not found')

    weak_meta_df = pd.read_csv(weak_path)

    return weak_meta_df


def laod_strong_label(strong_path: Path) -> pd.DataFrame:
    if not strong_path.exists():
        raise FileNotFoundError(f'{strong_path} is not found')

    strong_meta_df = pd.read_csv(strong_path)

    return strong_meta_df
