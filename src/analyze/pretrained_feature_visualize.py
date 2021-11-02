import argparse
from pathlib import Path
from typing import Union

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from analyze.reduce_dimention import run_tsne
from analyze.load_data import load_pretrain_hubert_feature


def get_embedded_features(X: np.ndarray, out_path: Path = None) -> np.ndarray:
    embedded_data = run_tsne(X, n_components=2)

    if out_path is not None:
        np.save(out_path/'embedded_features.npy', embedded_data)

    return embedded_data


def generate_visualize_data(
    embedded_data: np.ndarray,
    labels: list,
    out_path: Path = None
) -> Union[dict, list]:

    visualize_data = {}
    for label in labels:
        visualize_data[label] = {'x': [], 'y': []}

    for i, v in enumerate(tqdm(embedded_data)):
        prob = np.random.rand()
        if prob < 0.99:
            continue

        label = labels[i]
        visualize_data[label]['x'].append(v[0])
        visualize_data[label]['y'].append(v[1])

    if out_path is not None:
        np.save(out_path/'visualize_data.npy', visualize_data)

    return visualize_data


def visualize(visualize_data: dict, classes: list, out_path: Path, image_name: str):
    print('==========')
    tot = 0
    for label in classes:
        sample = len(visualize_data[label]['x'])
        print(f'{label}: {sample} samples')
        tot += sample
    print(f'total: {tot} samples')
    print('==========')

    # visualise
    plt.figure(figsize=(20, 20))
    for i, label in enumerate(reversed(classes)):
        data = visualize_data[label]
        plt.scatter(data['x'], data['y'], label=label)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(out_path/image_name)
    plt.clf()


def run_feature_visualize(
    feat_path: Path,
    km_path: Path,
    out_path: Path,
    image_name: str
) -> None:
    if not out_path.exists():
        out_path.mkdir(parents=True)

    # prepare input data
    print('prepare input data and label')
    X, labels = load_pretrain_hubert_feature(feat_path, km_path)

    # running T-SNE
    print('runnning T-SNE')
    embedded_data = get_embedded_features(X, out_path=None)
    # embedded_data = np.load(out_path/'result.npy')

    visualize_data = generate_visualize_data(
        embedded_data, labels, out_path=None)
    # visualize_data = np.load(out_path/'visualize_data.npy', allow_pickle=True).item()

    classes = list(visualize_data.keys())

    # visualize
    visualize(visualize_data, classes, out_path, image_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('feat_path')
    parser.add_argument('km_path')
    parser.add_argument('--out_path', default='.')
    parser.add_argument('--image_name', default='result.png')
    args = parser.parse_args()

    run_feature_visualize(
        feat_path=Path(args.feat_path),
        km_path=Path(args.km_path),
        out_path=Path(args.out_path),
        image_name=args.image_name
    )
