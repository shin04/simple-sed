from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics
import torch
from sed_eval.sound_event import SegmentBasedMetrics, EventBasedMetrics
from psds_eval import PSDSEval
from dcase_util.containers import MetaDataContainer


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


def calc_sed_weak_f1(weak_label: torch.Tensor, pred: torch.Tensor, average: str = 'macro') -> float:
    weak_label = weak_label.to('cpu').detach().numpy().copy()
    pred = pred.to('cpu').detach().numpy().copy()

    pred = pred > 0.5

    f1_score = metrics.f1_score(weak_label, pred, average=average)

    return f1_score


def calc_sed_eval_metrics(
    metadata_path: Path, prediction: MetaDataContainer, time_resolution: float, t_collar: float
) -> dict:
    meta_df = pd.read_csv(metadata_path)
    grand_truth = MetaDataContainer(meta_df.to_dict('records'))

    segment_based_metrics = SegmentBasedMetrics(
        event_label_list=grand_truth.unique_event_labels, time_resolution=time_resolution
    )

    event_based_metrics = EventBasedMetrics(
        event_label_list=grand_truth.unique_event_labels, t_collar=t_collar
    )

    for filename in grand_truth.unique_files:
        grand_truth_files = grand_truth.filter(filename=filename)
        prediction_files = prediction.filter(filename=filename)

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
    for i, k in enumerate(predictions.keys()):
        pred_df = pd.DataFrame(predictions[k])
        det = pred_df
        psds_eval.add_operating_point(
            det, info={"name": f"Op {i + 1:02d}", "threshold": k}
        )

    psds_score = psds_eval.psds(
        alpha_ct=alpha_ct, alpha_st=alpha_st, max_efpr=max_efpr)

    return psds_score.value, psds_macro_f1
