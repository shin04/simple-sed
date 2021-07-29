import numpy as np
import pandas as pd
from dcase_util.data import DecisionEncoder


def strong_label_encoding(
    sr: int,
    sample_len: int,
    frame_hop: int,
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

    label = np.zeros((frame_len, len(class_map)))

    for _, row in meta_df.iterrows():
        event_name = row[1]
        i = class_map[event_name]
        onset = int(row[2]*sr / frame_hop)
        offset = int(row[3]*sr / frame_hop)
        label[onset:offset, i] = 1.

    return label


def strong_label_decoding(
    preds: np.array, filename: str,
    sr: int, frame_hop: int, class_map: dict, threshold: float = 0.5,
) -> list:
    """
    strong_label_decoding converts a prediction to a strong label.
    Ex:
        [0,0,1,1,1,...] -> [[event,onset, offset], ...]

        Prameters
        ---------
        preds: (fraes, class)
    """

    result_label = []
    for i, pred in enumerate(preds):
        event = [k for k, v in class_map.items() if v == i][0]

        pred_section = pred > threshold
        change_indices = DecisionEncoder().find_contiguous_regions(pred_section)
        change_indices = change_indices / sr * frame_hop
        for indice in change_indices:
            onset = indice[0]
            offset = indice[1]
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
        strong_meta[strong_meta['filename'] ==
                    'soundscape_train_uniform1090.wav'],
        class_map
    )
    # print(np.where(label == 1.))

    result_label = strong_label_decoding(
        label.T, 'soundscape_train_uniform1090.wav', 44100, 256, class_map)
    print(result_label)
