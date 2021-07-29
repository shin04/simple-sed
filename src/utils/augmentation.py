import numpy as np


class GaussianNoise:
    def __init__(
        self, min_snr: float = 5.0, max_snr: float = 20.0
    ) -> None:
        self.min_snr = min_snr
        self.max_snr = max_snr

    def __call__(self, y: np.ndarray) -> np.array:
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        white_noise = np.random.randn(len(y))
        a_white = np.sqrt(white_noise ** 2).max()
        augmented = (y + white_noise * 1 / a_white * a_noise).astype(y.dtype)

        return augmented
