import numpy as np
import torch
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB


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
        log_scale: bool
    ) -> None:
        self.mel_spec_trans = MelSpectrogram(
            sample_rate=sr,
            n_fft=n_filters,
            win_length=n_window,
            hop_length=hop_length,
            n_mels=n_mels
        )

        self.log_scale = log_scale

        self.amp_to_db = AmplitudeToDB(stype='power')
        self.amp_to_db.amin = 1e-5

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        mel_spec = self.mel_spec_trans(y)
        if self.log_scale:
            log_offset = 1e-6
            mel_spec = torch.log(mel_spec + log_offset)
        else:
            mel_spec = self.amp_to_db(mel_spec)

        return mel_spec
