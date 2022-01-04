import numpy as np
import torch
import torchaudio.transforms as T


class GaussianNoise:
    def __init__(
        self, min_snr: float = 5.0, max_snr: float = 20.0
    ) -> None:
        self.min_snr = min_snr
        self.max_snr = max_snr

    def __call__(self, y: np.ndarray) -> np.ndarray:
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        white_noise = np.random.randn(len(y))
        a_white = np.sqrt(white_noise ** 2).max()
        augmented = (y + white_noise * 1 / a_white * a_noise).astype(y.dtype)

        return augmented


class GetMelSpectrogram:
    def __init__(
        self,
        sr: int,
        n_filters: int,
        n_window: int,
        hop_length: int,
        n_mels: int,
        f_min: int,
        f_max: int,
        log_scale: bool
    ) -> None:
        self.mel_spec_trans = T.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_filters,
            win_length=n_window,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels
        )

        self.log_scale = log_scale

        self.amp_to_db = T.AmplitudeToDB(stype='power')
        self.amp_to_db.amin = 1e-5

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        mel_spec = self.mel_spec_trans(y)
        if self.log_scale:
            log_offset = 1e-6
            mel_spec = torch.log(mel_spec + log_offset)
        else:
            mel_spec = self.amp_to_db(mel_spec)

        return mel_spec


class TimeStretch:
    def __init__(self, rate: float) -> None:
        self.rate = rate
        self.strecher = T.TimeStretch()

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        streched = self.strecher(y, self.rate)

        return streched


class TimeMasking:
    def __init__(self, time_mask_param: int, mask_num: int) -> None:
        self.masker = T.TimeMasking(time_mask_param)
        self.mask_num = mask_num

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        masked = y
        for _ in range(self.mask_num):
            masked = self.masker(masked)

        return masked


class FrequencyMasking:
    def __init__(self, freq_mask_param: int, mask_num: int) -> None:
        self.masker = T.FrequencyMasking(freq_mask_param)
        self.mask_num = mask_num

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        masked = y
        for _ in range(self.mask_num):
            masked = self.masker(masked)

        return masked


class Normalize:
    def __init__(self, mean=None, std=None, mode="gcmvn"):
        self.mean = mean
        self.std = std
        self.mode = mode
        self.ref_level_db = 20
        self.min_level_db = -80

    def __call__(self, data):
        if self.mode == "gcmvn":
            return (data - self.mean) / self.std
        elif self.mode == "cmvn":
            return (data - data.mean(axis=0)) / data.std(axis=0)
        elif self.mode == "cmn":
            return data - data.mean(axis=0)
        elif self.mode == "min_max":
            data -= self.ref_level_db
            return np.clip((data - self.min_level_db) / -self.min_level_db, 0, 1)
