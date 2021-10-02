import argparse
from pathlib import Path

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from analyze.reduce_dimention import run_tsne
from analyze.load_data import load_hubert_feature, laod_strong_label
from utils.label_encoder import strong_label_encoding


def run_feature_visualize(
    feat_path: Path,
    strong_path: Path,
    out_path: Path,
    sr: int,
    sample_len: int,
    frame_hop: int,
    net_pooling_rate: int
) -> None:
    if not out_path.exists():
        out_path.mkdir(parents=True)

    # feats = load_hubert_feature(feat_path)
    # strong_meta = laod_strong_label(strong_path)

    # # prepare input data
    # print('prepare input data ...')
    # filenames = list(feats.keys())
    # n_feats = len(feats)
    # n_frames, n_dim = feats[filenames[0]].shape
    # X_shape = (n_feats*n_frames, n_dim)
    # X = np.zeros(X_shape)

    # s_idx, g_idx = 0, 0
    # lengthes_by_files = []
    # for filename in filenames:
    #     feat = feats[filename]
    #     g_idx += len(feat)
    #     lengthes_by_files.append(len(feat))
    #     X[s_idx:g_idx] = feat
    #     s_idx = g_idx

    # # prepare label
    # print('prepare label ...')
    # classes = sorted(strong_meta['event_label'].unique().tolist())
    # class_map = {c: i for i, c in enumerate(classes)}

    # labels = []
    # for i, filename in enumerate(tqdm(filenames)):
    #     strong_label = strong_label_encoding(
    #         sr,
    #         sample_len,
    #         frame_hop,
    #         net_pooling_rate,
    #         strong_meta[strong_meta['filename'] == filename],
    #         class_map
    #     )
    #     strong_label = strong_label[:499, ]

    #     for label_by_frame in strong_label:
    #         event_ids = np.where(label_by_frame == 1.)[0].tolist()
    #         events = []
    #         for event_id in event_ids:
    #             events.append(classes[int(event_id)])
    #         if len(events) == 0:
    #             events.append('silent')

    #         labels.append(events)

    # # running T-SNE
    # print('runnning T-SNE')
    # embedded_data = run_tsne(X, n_components=2)
    # np.save(out_path/'result.npy', embedded_data)
    # # embedded_data = np.load(out_path/'result.npy')

    # visualize_data = {}
    # for event in classes + ['silent']:
    #     visualize_data[event] = {'x': [], 'y': []}

    # for i, v in enumerate(tqdm(embedded_data)):
    #     label = labels[i]
    #     for event in label:
    #         visualize_data[event]['x'].append(v[0])
    #         visualize_data[event]['y'].append(v[1])

    #     # e = ','.join(label)

    #     # if e not in visualize_data:
    #     #     visualize_data[e] = {'x': [v[0]], 'y': [v[1]]}
    #     # else:
    #     #     visualize_data[e]['x'].append(v[0])
    #     #     visualize_data[e]['y'].append(v[1])
    # np.save(out_path/'visualize_data.npy', visualize_data)

    # # classes += ['silent']
    # classes = list(visualize_data.keys())

    # print('==========')
    # tot = 0
    # for label in classes:
    #     sample = len(visualize_data[label]['x'])
    #     print(f'{label}: {sample} samples')
    #     tot += sample
    # print(f'total: {tot} samples')
    # print('==========')

    visualize_data = np.load(out_path/'visualize_data.npy', allow_pickle=True).item()
    classes = list(visualize_data.keys())

    # visualise
    plt.figure(figsize=(20, 20))
    for i, label in enumerate(classes):
        data = visualize_data[label]
        plt.scatter(data['x'], data['y'], label=label)
    plt.legend(fontsize=30)
    plt.savefig(out_path/'image-all.png')
    plt.clf()

    for i, label in enumerate(classes):
        plt.figure(figsize=(10, 10))
        data = visualize_data[label]
        plt.scatter(data['x'], data['y'], label=label)
        plt.title(label)
        plt.legend(fontsize=30)
        plt.savefig(out_path/f'image-{classes[i]}.png')
        plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('feat_path')
    parser.add_argument(
        '--strong_path', default='/home/kajiwara21/work/sed/meta/test_meta_strong.csv')
    parser.add_argument('--out_path', default='.')
    parser.add_argument('--sr', default=16000)
    parser.add_argument('--sample_len', default=160000)
    parser.add_argument('--frame_hop', default=1)
    parser.add_argument('--net_pooling_rate', default=320)
    args = parser.parse_args()

    run_feature_visualize(
        feat_path=Path(args.feat_path),
        strong_path=Path(args.strong_path),
        out_path=Path(args.out_path),
        sr=args.sr,
        sample_len=args.sample_len,
        frame_hop=args.frame_hop,
        net_pooling_rate=args.net_pooling_rate
    )