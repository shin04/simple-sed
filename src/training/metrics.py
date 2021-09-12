import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics
import torch
from sed_eval.sound_event import SegmentBasedMetrics, EventBasedMetrics
from psds_eval import PSDSEval

from utils.label_encoder import strong_label_decoding


def sed_average_precision(
    strong_label: torch.Tensor, pred: torch.Tensor, average: str = 'macro'
) -> float:
    """
    average: macro | micro
    """

    strong_label = strong_label.to('cpu').detach().numpy().copy()
    pred = pred.to('cpu').detach().numpy().copy()

    (batch_num, classes_num, frame_num) = strong_label.shape

    average_precision = metrics.average_precision_score(
        strong_label.reshape((batch_num * frame_num, classes_num)),
        pred.reshape((batch_num * frame_num, classes_num)),
        average=average
    )

    return average_precision


def calc_sed_weak_f1(
    weak_label: torch.Tensor,
    pred: torch.Tensor,
    thr: float,
    average: str = 'macro'
) -> float:
    weak_label = weak_label.to('cpu').detach().numpy().copy()
    pred = pred.to('cpu').detach().numpy().copy()

    pred = pred > thr

    f1_score = metrics.f1_score(weak_label, pred, average=average)

    return f1_score


def get_event_list_current_file(df: pd.DataFrame, fname: str) -> list:
    event_file = df[df["filename"] == fname]
    if len(event_file) == 1:
        if pd.isna(event_file["event_label"].iloc[0]):
            event_list_for_current_file = [{"filename": fname}]
        else:
            event_list_for_current_file = event_file.to_dict("records")
    else:
        event_list_for_current_file = event_file.to_dict("records")

    return event_list_for_current_file


def calc_sed_eval_metrics(
    metadata_path: Path,
    prediction: pd.DataFrame,
    class_map: dict,
    time_resolution: float,
    t_collar: float,
    is_training: bool = True,
) -> dict:
    if len(prediction.columns) == 0:
        return {
            'segment': {'class_wise_f1': 0.0, 'overall_f1': 0.0, },
            'event': {'class_wise_f1': 0.0, 'overall_f1': 0.0, }
        }

    meta_df = pd.read_csv(metadata_path)
    grand_truth = meta_df

    evaluated_files = meta_df["filename"].unique()
    classes = list(class_map.keys())

    segment_based_metrics = SegmentBasedMetrics(
        event_label_list=classes,
        time_resolution=time_resolution
    )

    event_based_metrics = EventBasedMetrics(
        event_label_list=classes,
        t_collar=t_collar
    )

    # for filename in grand_truth.unique_files:
    for filename in evaluated_files:
        grand_truth_files = get_event_list_current_file(grand_truth, filename)
        prediction_files = get_event_list_current_file(prediction, filename)

        segment_based_metrics.evaluate(
            reference_event_list=grand_truth_files,
            estimated_event_list=prediction_files
        )

        event_based_metrics.evaluate(
            reference_event_list=grand_truth_files,
            estimated_event_list=prediction_files
        )

    segment_res = segment_based_metrics.results()
    event_res = event_based_metrics.results()

    if is_training:
        return {
            'segment': {
                'class_wise_f1': segment_res['class_wise_average']['f_measure']['f_measure'],
                'overall_f1': segment_res['overall']['f_measure']['f_measure'],
            },
            'event': {
                'class_wise_f1': event_res['class_wise_average']['f_measure']['f_measure'],
                'overall_f1': event_res['overall']['f_measure']['f_measure'],
            },
        }
    else:
        return segment_based_metrics, event_based_metrics


def calc_psds_eval_metrics(
    gt_path: Path,
    meta_path: Path,
    predictions: dict,
    dtc_threshold: float = 0.5,
    gtc_threshold: float = 0.5,
    cttc_threshold: float = 0.3,
    alpha_ct: float = 0,
    alpha_st: float = 0,
    max_efpr: float = 100,
) -> dict:
    gt_df = pd.read_csv(gt_path)

    meta_df = pd.read_csv(meta_path)  # cols=[filename, duration]

    psds_eval = PSDSEval(
        ground_truth=gt_df,
        metadata=meta_df,
        dtc_threshold=dtc_threshold,
        gtc_threshold=gtc_threshold,
        cttc_threshold=cttc_threshold
    )

    """calculation macro f1 score"""
    psds_macro_f1 = []
    for thr in predictions.keys():
        pred_df = pd.DataFrame(predictions[thr])
        if not pred_df.empty:
            macro_f1, _ = psds_eval.compute_macro_f_score(pred_df)
        else:
            macro_f1 = 0.
        if np.isnan(macro_f1):
            macro_f1 = 0.
        psds_macro_f1.append(macro_f1)
    psds_macro_f1 = np.mean(psds_macro_f1)

    """calculation psds"""
    try:
        for i, k in enumerate(predictions.keys()):
            pred_df = pd.DataFrame(predictions[k])
            det = pred_df
            psds_eval.add_operating_point(
                det, info={"name": f"Op {i + 1:02d}", "threshold": k}
            )

        psds_score = psds_eval.psds(
            alpha_ct=alpha_ct, alpha_st=alpha_st, max_efpr=max_efpr)
    except Exception as e:
        logging.error(e)

        return 0., psds_macro_f1

    return psds_score.value, psds_macro_f1


def search_best_threshold(
    step: float,
    meta_path: Path,
    prediction: list,  # list of np.array
    filenames: list,
    sr: int,
    hop_length: int,
    pooling_rate: int,
    class_map: dict,
) -> float:
    assert 0 < step < 1.0

    labels = class_map.keys()

    best_th = {k: 0.0 for k in labels}
    best_f1 = {k: 0.0 for k in labels}

    for th in np.arange(step, 1.0, step):
        result = []
        for i, pred in enumerate(prediction):
            result += strong_label_decoding(
                pred, filenames[i], sr, hop_length, pooling_rate, class_map
            )

        _, events_metric = calc_sed_eval_metrics(
            meta_path, pd.DataFrame(result), 0.1, 0.2, False
        )

        for i, label in enumerate(labels):
            f1 = events_metric.class_wise_f_measure(
                event_label=label)["f_measure"]

            if f1 > best_f1[label]:
                best_th[label] = th
                best_f1[label] = f1

    return best_th


if __name__ == '__main__':
    meta_path = '/home/kajiwara21/work/sed/meta/valid_meta_strong.csv'
    est_df = pd.read_csv('/home/kajiwara21/work/sed/prediction.csv')

    res = calc_sed_eval_metrics(
        meta_path, est_df, 0.1, 0.2
    )

    print(res['segment'])
    print(res['event'])
