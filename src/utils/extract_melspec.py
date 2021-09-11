import numpy as np
import librosa


def extract_melspec(
    x: np.ndarray,
    sr: int = 44100,
    n_filters: int = 2048,
    # n_window: int,
    hop_length: int = 1024,
    n_mels: int = 128,
    f_min: int = 0,
    f_max: int = 22050,
    log_scale: bool = True
) -> np.ndarray:
    fmin = f_min
    fmax = f_max
    ham_win = np.hamming(n_filters)

    spec = librosa.stft(
        x,
        n_fft=n_filters,
        hop_length=hop_length,
        window=ham_win,
        center=True,
        pad_mode="reflect"
    )

    mel_spec = librosa.feature.melspectrogram(
        S=np.abs(spec),
        sr=sr,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        htk=False,
        norm=None,
    )

    if log_scale:
        mel_spec = librosa.amplitude_to_db(mel_spec)

    # mel_spec = mel_spec.T
    mel_spec = mel_spec.astype(np.float32)

    return mel_spec
