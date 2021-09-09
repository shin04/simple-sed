from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import soundfile as sf


def sed_res_visualize(
    pred_path: Path, audio_path: Path, weak_path: Path, out_path: Path, output_name: str
) -> None:

    if not pred_path.exists():
        raise RuntimeError(f'{pred_path} is not found')

    if not audio_path.exists():
        raise RuntimeError(f'{audio_path} is not found')

    if not weak_path.exists():
        raise RuntimeError(f'{weak_path} is not found')

    if not out_path.exists():
        out_path.mkdir(parents=True)

    weak_meta_df = pd.read_csv(weak_path)
    labels = sorted(weak_meta_df.columns[1:].tolist())

    pred_dict = np.load(pred_path, allow_pickle=True).item()
    k = list(pred_dict.keys())[0]
    v = pred_dict[k].T  # (n_frames, events)
    sorted_indexes = np.argsort(np.max(v, axis=0))[::-1]
    top_result_mat = v[:, sorted_indexes[0:]]

    waveform, sr = sf.read(audio_path / k)
    frames_per_second = sr // 1024

    stft = librosa.stft(y=waveform, n_fft=2048,
                        hop_length=1024, window='hann', center=True)
    frames_num = stft.shape[-1]

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 4))
    fig.suptitle(k)
    axs[0].matshow(np.log(np.abs(stft)), origin='lower',
                   aspect='auto', cmap='jet')
    axs[0].set_ylabel('Frequency bins')
    axs[0].set_title('Log spectrogram')
    axs[1].matshow(top_result_mat.T, origin='upper',
                   aspect='auto', cmap='jet', vmin=0, vmax=1)
    axs[1].xaxis.set_ticks(np.arange(0, frames_num, frames_per_second))
    axs[1].xaxis.set_ticklabels(np.arange(0, frames_num / frames_per_second))
    axs[1].yaxis.set_ticks(np.arange(0, 10))
    axs[1].yaxis.set_ticklabels(np.array(labels)[sorted_indexes[0: 10]])
    axs[1].yaxis.grid(color='k', linestyle='solid', linewidth=0.3, alpha=0.3)
    axs[1].set_xlabel('Seconds')
    axs[1].xaxis.set_ticks_position('bottom')

    plt.tight_layout()
    plt.savefig(out_path / output_name)


if __name__ == '__main__':
    '''
    example
    pred_path: /home/kajiwara21/nas02/home/results/sed/baseline-20210823035150-test.npy
    audio_parh: /home/kajiwara21/dataset/URBAN-SED_v2.0.0/audio/test
    weak_path: /home/kajiwara21/work/sed/meta/test_meta_weak.csv
    out_path: /home/kajiwara21/work/sed/visualize_result
    output_name: result.png
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('pred_path', help='to prediction(npy) path')
    parser.add_argument(
        'audio_path', help='to raw audio (used by prediction) path')
    parser.add_argument('weak_path', help='to weak label (tsv) path')
    parser.add_argument(
        'out_path', help='to saving visualize prediction path', default='../visualize_result')
    parser.add_argument(
        'output_name', help='result image name', default='result.png')
    args = parser.parse_args()

    pred_path = Path(args.pred_path)
    audio_path = Path(args.audio_path)
    weak_path = Path(args.weak_path)
    out_path = Path(args.out_path)
    output_name = args.output_name

    sed_res_visualize(pred_path, audio_path, weak_path, out_path, output_name)
