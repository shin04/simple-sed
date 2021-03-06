from __future__ import annotations

import numpy as np
import pandas as pd
# import scipy
from scipy.signal import medfilt
from dcase_util.data import DecisionEncoder, ProbabilityEncoder


def strong_label_encoding(
    sr: int,
    sample_len: int,
    frame_hop: int,
    net_pooling_rate: int,
    meta_df: pd.DataFrame,
    class_map: dict
) -> np.array:
    """
    strong_label_encoder converts a strong label set to training label.
    Ex:
        {filename,event,onset,offset} -> [0,0,1,1,1,...]

    Parameters
    ----------
    sr: sampling rate
    sample_len: length of a data sample
    frame_hop: hop length (when extract mel spectrogram)
    meta_df: meta data, with {filename,event,onset,offset}
    class_map: {class_name: number}
    """

    frame_len = -(-sample_len // frame_hop)
    frame_len = frame_len // net_pooling_rate

    label = np.zeros((frame_len, len(class_map)))

    for _, row in meta_df.iterrows():
        event_name = row[1]
        i = class_map[event_name]
        onset = int(row[2]*sr / frame_hop / net_pooling_rate)
        offset = int(row[3]*sr / frame_hop / net_pooling_rate)
        label[onset:offset, i] = 1.

    return label


def strong_label_decoding(
    preds: np.ndarray,
    filename: str,
    sr: int,
    frame_hop: int,
    net_pooling_rate: int,
    class_map: dict,
    threshold: float | dict = 0.5,
    median_filter: int = 7
) -> list:
    """
    strong_label_decoding converts a prediction to a strong label.
    Ex:
        [0,0,1,1,1,...] -> [[event,onset, offset], ...]

    Prameters
    ---------
    preds: (class, frame)
    binarization_type:
        global_threshold or class_threshold
    """

    classes = list(class_map.keys())

    # generate threshold list
    # th_list = []
    # if type(threshold) == float:
    #     th_list = [threshold] * len(class_map)
    # else:
    #     for cls in classes:
    #         th_list.append(threshold[cls])

    preds = ProbabilityEncoder().binarization(
        preds,
        threshold=threshold,
    )

    result_label = []
    for i, pred in enumerate(preds):
        event = classes[i]

        # pred = pred > th_list[i]
        # pred = scipy.ndimage.filters.median_filter(pred, median_filter)
        pred = medfilt(pred, median_filter)

        change_indices = DecisionEncoder().find_contiguous_regions(pred)
        change_indices = change_indices / sr * frame_hop * net_pooling_rate
        for indice in change_indices:
            onset = indice[0].clip(0., 10.)
            offset = indice[1].clip(0., 10.)
            # if offset > 10:
            #     offset = 10.0
            result_label.append(
                {'filename': filename, 'event_label': event, 'onset': onset, 'offset': offset})

    return result_label


if __name__ == '__main__':
    strong_meta = pd.read_csv('../meta/train_meta_strong.csv')
    class_list = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling',
                  'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']
    class_map = {c: i for i, c in enumerate(class_list)}
    print(strong_meta[strong_meta['filename'] ==
                      'soundscape_train_uniform1090.wav'])

    label = strong_label_encoding(
        44100,
        10*44100,
        256,
        1,
        strong_meta[strong_meta['filename'] ==
                    'soundscape_train_uniform1090.wav'],
        class_map
    )
    # print(np.where(label == 1.))

    result_label = strong_label_decoding(
        label.T,
        'soundscape_train_uniform1090.wav',
        44100, 256, 1,
        class_map,
        0.5
    )
    result_df = pd.DataFrame(result_label)
    print(result_df)
