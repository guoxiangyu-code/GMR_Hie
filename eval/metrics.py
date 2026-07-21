# -*- coding: utf-8 -*-
"""
Soccer-GMR metric implementations.

Metrics follow the paper notation: mR@k, mR+@k, mAP, mIoU@k, mIoU+@k,
GMR-CLS (AUROC / Rej-F1), and G-mIoU@k.
Localization metrics are computed on positive queries \\mathcal{Q}^{+}; rejection
and G-mIoU are computed on all queries \\mathcal{Q}.
"""

from __future__ import annotations

import multiprocessing as mp
from collections import OrderedDict, defaultdict
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from .utils import (
        compute_average_precision_detection,
        compute_temporal_iou_batch_cross,
    )
except ImportError:  # Support `python eval/metrics.py` and the legacy CLI layout.
    from utils import (
        compute_average_precision_detection,
        compute_temporal_iou_batch_cross,
    )

DEFAULT_IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def greedy_match(
    preds: List[Sequence[float]],
    gts: List[Sequence[float]],
    iou_thd: float = -1.0,
) -> List[Tuple[int, int, float]]:
    """
    Greedily match predicted windows to GT windows one-to-one.

    Args:
        preds: [[st, ed], ...], ordered as the top-k predictions.
        gts:   [[st, ed], ...]
        iou_thd: Minimum IoU for a valid match; -1 forces matching for mIoU metrics.
    """
    if len(preds) == 0 or len(gts) == 0:
        return []
    preds_arr = np.array(preds, dtype=np.float64).reshape(-1, 2)
    gts_arr = np.array(gts, dtype=np.float64).reshape(-1, 2)
    iou_matrix, _ = compute_temporal_iou_batch_cross(preds_arr, gts_arr)

    matched_gt: set = set()
    matches: List[Tuple[int, int, float]] = []
    for i in range(iou_matrix.shape[0]):
        best_iou, best_j = -1.0, None
        for j in range(iou_matrix.shape[1]):
            if j not in matched_gt and iou_matrix[i, j] > best_iou:
                best_iou = iou_matrix[i, j]
                best_j = j
        if best_iou >= iou_thd and best_j is not None:
            matched_gt.add(best_j)
            matches.append((i, best_j, float(best_iou)))
    return matches


def _ap_worker(triple: Tuple[Any, list, list], tiou_thresholds: np.ndarray) -> Tuple[Any, np.ndarray]:
    qid, gt_list, pred_list = triple
    if len(gt_list) == 0:
        return qid, np.zeros(len(tiou_thresholds))
    scores, _, _ = compute_average_precision_detection(
        gt_list, pred_list, tiou_thresholds=tiou_thresholds
    )
    return qid, scores


def compute_mAP(
    submission: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    iou_thds: np.ndarray = DEFAULT_IOU_THRESHOLDS,
    max_pred_windows: int = 10,
    num_workers: int = 8,
) -> Dict[str, Any]:
    """Compute standard temporal detection mAP on positive queries."""
    iou_thds = np.array([float(f"{e:.2f}") for e in iou_thds])

    pred_by_qid: Dict[Any, list] = defaultdict(list)
    for d in submission:
        qid = d["qid"]
        windows = d.get("pred_relevant_windows", [])
        if max_pred_windows is not None:
            windows = windows[:max_pred_windows]
        for w in windows:
            pred_by_qid[qid].append({
                "video-id": qid,
                "t-start": w[0],
                "t-end": w[1],
                "score": w[2] if len(w) > 2 else 1.0,
            })

    gt_by_qid: Dict[Any, list] = defaultdict(list)
    for d in ground_truth:
        qid = d["qid"]
        for w in d["relevant_windows"]:
            gt_by_qid[qid].append({
                "video-id": qid,
                "t-start": w[0],
                "t-end": w[1],
            })

    triples = [[qid, gt_by_qid[qid], pred_by_qid.get(qid, [])] for qid in gt_by_qid]

    compute_fn = partial(_ap_worker, tiou_thresholds=iou_thds)
    qid2ap: Dict[Any, np.ndarray] = {}
    if num_workers > 1 and len(triples) > 50:
        with mp.Pool(num_workers) as pool:
            for qid, scores in pool.imap_unordered(compute_fn, triples, chunksize=50):
                qid2ap[qid] = scores
    else:
        for t in triples:
            qid, scores = compute_fn(t)
            qid2ap[qid] = scores

    if len(qid2ap) == 0:
        return {"mAP": 0.0, "AP@IoU": {}}

    all_aps = np.array(list(qid2ap.values()))
    ap_per_iou = all_aps.mean(axis=0)
    mAP = float(ap_per_iou.mean())

    return {
        "mAP": round(100 * mAP, 2),
        "AP@IoU": {
            f"{iou_thds[i]:.2f}": round(100 * float(ap_per_iou[i]), 2)
            for i in range(len(iou_thds))
        },
    }


def compute_mR(
    submission: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    k_list: Sequence[int] = (1, 3, 5),
    iou_thds: np.ndarray = DEFAULT_IOU_THRESHOLDS,
) -> Dict[str, Any]:
    """
    Paper mR@k: greedily match predictions for each IoU threshold \\theta on
    positive queries, compute per-query recall |M_k|/|G|, then average over
    queries and thresholds. The output is a percentage.
    """
    iou_thds = [float(f"{e:.2f}") for e in iou_thds]

    pred_map: Dict[Any, List] = {}
    for d in submission:
        pred_map[d["qid"]] = [[w[0], w[1]] for w in d.get("pred_relevant_windows", [])]

    positive_gt = [d for d in ground_truth if len(d["relevant_windows"]) > 0]

    results: Dict[str, Any] = {}
    for k in k_list:
        recall_per_iou: List[float] = []
        for thd in iou_thds:
            recalls: List[float] = []
            for d in positive_gt:
                gts = d["relevant_windows"]
                preds = pred_map.get(d["qid"], [])[:k]
                if len(preds) == 0:
                    recalls.append(0.0)
                    continue
                matches = greedy_match(preds, gts, iou_thd=thd)
                recalls.append(len(matches) / len(gts))
            recall_per_iou.append(float(np.mean(recalls)) if recalls else 0.0)

        m_r = float(np.mean(recall_per_iou))
        results[f"mR@{k}"] = round(100 * m_r, 2)
        results[f"mR@{k}_per_IoU"] = {
            f"{thd:.2f}": round(100 * recall_per_iou[i], 2)
            for i, thd in enumerate(iou_thds)
        }
    return results


def compute_mR_plus(
    submission: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    k_list: Sequence[int] = (1, 3, 5),
    iou_thds: np.ndarray = DEFAULT_IOU_THRESHOLDS,
) -> Dict[str, Any]:
    """Paper mR+@k for queries with at least two GT windows, excluding the first hit."""
    iou_thds = [float(f"{e:.2f}") for e in iou_thds]

    pred_map: Dict[Any, List] = {}
    for d in submission:
        pred_map[d["qid"]] = [[w[0], w[1]] for w in d.get("pred_relevant_windows", [])]

    multi_gt = [d for d in ground_truth if len(d["relevant_windows"]) >= 2]

    results: Dict[str, Any] = {"num_multi_moment_queries": len(multi_gt)}
    for k in k_list:
        ir_plus_per_iou: List[float] = []
        for thd in iou_thds:
            values: List[float] = []
            for d in multi_gt:
                gts = d["relevant_windows"]
                preds = pred_map.get(d["qid"], [])[:k]
                if len(preds) == 0:
                    values.append(0.0)
                    continue
                n_matched = len(greedy_match(preds, gts, iou_thd=thd))
                values.append(
                    max(0, n_matched - 1) / (len(gts) - 1) if n_matched >= 1 else 0.0
                )
            ir_plus_per_iou.append(float(np.mean(values)) if values else 0.0)

        m_r_plus = float(np.mean(ir_plus_per_iou))
        results[f"mR+@{k}"] = round(100 * m_r_plus, 2)
        results[f"mR+@{k}_per_IoU"] = {
            f"{thd:.2f}": round(100 * ir_plus_per_iou[i], 2)
            for i, thd in enumerate(iou_thds)
        }
    return results


def compute_mIoU(
    submission: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    k_list: Sequence[int] = (1, 3, 5),
) -> Dict[str, Any]:
    """mIoU@k: force-match top-k predictions to GT on positive queries and macro-average."""
    pred_map: Dict[Any, List] = {}
    for d in submission:
        pred_map[d["qid"]] = [[w[0], w[1]] for w in d.get("pred_relevant_windows", [])]

    positive_gt = [d for d in ground_truth if len(d["relevant_windows"]) > 0]

    results: Dict[str, Any] = {}
    for k in k_list:
        sample_ious: List[float] = []
        for d in positive_gt:
            gts = d["relevant_windows"]
            preds = pred_map.get(d["qid"], [])[:k]
            if len(preds) == 0:
                sample_ious.append(0.0)
                continue
            matches = greedy_match(preds, gts, iou_thd=-1.0)
            if len(matches) == 0:
                sample_ious.append(0.0)
            else:
                sample_ious.append(float(np.mean([m[2] for m in matches])))
        results[f"mIoU@{k}"] = (
            round(100 * float(np.mean(sample_ious)), 2) if sample_ious else 0.0
        )
    return results


def compute_mIoU_plus(
    submission: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    k_list: Sequence[int] = (1, 3, 5),
) -> Dict[str, Any]:
    """mIoU+@k: average remaining matches after removing the highest-IoU pair."""
    pred_map: Dict[Any, List] = {}
    for d in submission:
        pred_map[d["qid"]] = [[w[0], w[1]] for w in d.get("pred_relevant_windows", [])]

    multi_gt = [d for d in ground_truth if len(d["relevant_windows"]) >= 2]

    results: Dict[str, Any] = {}
    for k in k_list:
        sample_ious: List[float] = []
        for d in multi_gt:
            gts = d["relevant_windows"]
            preds = pred_map.get(d["qid"], [])[:k]
            if len(preds) == 0:
                sample_ious.append(0.0)
                continue
            matches = greedy_match(preds, gts, iou_thd=-1.0)
            if len(matches) <= 1:
                sample_ious.append(0.0)
            else:
                ious = sorted([m[2] for m in matches], reverse=True)
                sample_ious.append(float(np.mean(ious[1:])))
        results[f"mIoU+@{k}"] = (
            round(100 * float(np.mean(sample_ious)), 2) if sample_ious else 0.0
        )
    return results


def _compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUROC by trapezoidal integration; return 0.5 if one class is missing."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5

    desc_idx = np.argsort(y_score, kind="mergesort")[::-1]
    y_true_sorted = y_true[desc_idx]
    y_score_sorted = y_score[desc_idx]

    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)
    tpr = tps / n_pos
    fpr = fps / n_neg

    distinct = np.where(np.diff(y_score_sorted))[0]
    distinct = np.append(distinct, len(y_score_sorted) - 1)
    tpr = np.concatenate([[0], tpr[distinct]])
    fpr = np.concatenate([[0], fpr[distinct]])

    return float(np.trapezoid(tpr, fpr))


def get_existence_score(pred: Dict[str, Any]) -> Tuple[float, str]:
    """
    Use explicit existence probability for s(q), or max window confidence as a proxy.
    Return (score, source_label).
    """
    if "pred_exist_score" in pred:
        try:
            return float(pred["pred_exist_score"]), "pred_exist_score"
        except (TypeError, ValueError):
            pass
    windows = pred.get("pred_relevant_windows", []) or []
    scores: List[float] = []
    for w in windows:
        if isinstance(w, (list, tuple)) and len(w) >= 3:
            try:
                scores.append(float(w[2]))
            except (TypeError, ValueError):
                continue
    if scores:
        return max(scores), "window_score"
    return 0.0, "default"


def compute_gmr_cls(
    submission: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    thresholds: Tuple[float, ...] = (0.4, 0.6),
) -> "OrderedDict[str, Any]":
    """
    GMR-CLS: AUROC plus Rej-F1 / Acc and related scores at each threshold.
    The paper treats s(q)<=\\tau as null; this implementation uses score>\\tau
    as positive, which is the complementary binary decision.
    """
    qid2pred = {d["qid"]: d for d in submission if isinstance(d, dict) and "qid" in d}

    y_true: List[int] = []
    y_score: List[float] = []
    score_sources: Dict[str, int] = defaultdict(int)

    for gt in ground_truth:
        qid = gt["qid"]
        gt_pos = len(gt.get("relevant_windows", [])) > 0
        y_true.append(1 if gt_pos else 0)

        pred = qid2pred.get(qid, {})
        score, source = get_existence_score(pred)
        y_score.append(score)
        score_sources[source] += 1

    y_true_arr = np.array(y_true)
    y_score_arr = np.array(y_score)

    auroc = _compute_auroc(y_true_arr, y_score_arr)

    per_thd: "OrderedDict[str, OrderedDict[str, Any]]" = OrderedDict()
    for thd in sorted(thresholds):
        # Positive class means a non-empty prediction, complementing the paper's null rule.
        pred_pos = y_score_arr > thd
        tp = int(np.sum((y_true_arr == 1) & pred_pos))
        tn = int(np.sum((y_true_arr == 0) & ~pred_pos))
        fp = int(np.sum((y_true_arr == 0) & pred_pos))
        fn = int(np.sum((y_true_arr == 1) & ~pred_pos))

        denom = tp + tn + fp + fn
        acc = (tp + tn) / denom if denom > 0 else 0.0
        rej_p = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        rej_r = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        rej_f1 = 2 * rej_p * rej_r / (rej_p + rej_r) if (rej_p + rej_r) > 0 else 0.0

        per_thd[str(thd)] = OrderedDict([
            ("Rej-F1", round(100 * rej_f1, 2)),
            ("Acc", round(100 * acc, 2)),
            ("Rej-P", round(100 * rej_p, 2)),
            ("Rej-R", round(100 * rej_r, 2)),
            ("TP", tp),
            ("TN", tn),
            ("FP", fp),
            ("FN", fn),
        ])

    return OrderedDict([
        ("AUROC", round(100 * auroc, 2)),
        ("per_threshold", per_thd),
        ("score_sources", dict(score_sources)),
    ])


def _clean_pred_windows(
    windows: Any,
    max_pred_windows: Optional[int] = None,
) -> List[List[float]]:
    """Parse pred_relevant_windows into [st, ed, score] items and skip invalid entries."""
    cleaned: List[List[float]] = []
    for w in windows or []:
        if not isinstance(w, (list, tuple)) or len(w) < 2:
            continue
        try:
            st, ed = float(w[0]), float(w[1])
            score = float(w[2]) if len(w) > 2 else 1.0
        except (TypeError, ValueError):
            continue
        if ed <= st:
            continue
        cleaned.append([st, ed, score])
    if max_pred_windows is not None:
        cleaned = cleaned[:max_pred_windows]
    return cleaned


def prepare_submission_for_gmiou(
    submission: List[Dict[str, Any]],
    cls_threshold: float = 0.4,
    max_pred_windows: Optional[int] = 10,
) -> Tuple[List[Dict[str, Any]], OrderedDict[str, Any]]:
    """
    Gate predicted windows by existence score: keep top predictions when s(q)>\\tau,
    otherwise treat the prediction set as \\emptyset.
    Only used for G-mIoU@k; localization metrics use the original submission.
    """
    processed: List[Dict[str, Any]] = []
    score_sources: Dict[str, int] = defaultdict(int)
    num_pred_pos = 0
    num_pred_neg = 0
    non_empty_before = 0
    non_empty_after = 0

    for pred in submission:
        if not isinstance(pred, dict) or "qid" not in pred:
            continue

        score, source = get_existence_score(pred)
        raw_windows = _clean_pred_windows(
            pred.get("pred_relevant_windows", []),
            max_pred_windows=max_pred_windows,
        )
        pred_positive = score > cls_threshold
        final_windows = raw_windows if pred_positive else []

        processed.append({
            "qid": pred["qid"],
            "pred_relevant_windows": final_windows,
        })

        score_sources[source] += 1
        non_empty_before += int(len(raw_windows) > 0)
        non_empty_after += int(len(final_windows) > 0)
        if pred_positive:
            num_pred_pos += 1
        else:
            num_pred_neg += 1

    stats = OrderedDict([
        ("cls_threshold", cls_threshold),
        ("decision_positive_if", "existence_score > cls_threshold"),
        ("num_submission_samples", len(processed)),
        ("num_predicted_positive", num_pred_pos),
        ("num_predicted_negative", num_pred_neg),
        ("num_non_empty_windows_before_gate", non_empty_before),
        ("num_non_empty_windows_after_gate", non_empty_after),
        ("score_sources", dict(score_sources)),
    ])
    return processed, stats


def _compute_set_iou_score(
    preds: List[Sequence[float]],
    gts: List[Sequence[float]],
) -> float:
    """
    Set-level IoU for one query: both empty gives 1, one empty gives 0, and two
    non-empty sets use sum(matched IoU) / (|pred|+|gt|-|M|) with greedy IoU>0 matches.
    """
    if len(preds) == 0 and len(gts) == 0:
        return 1.0
    if len(preds) == 0 or len(gts) == 0:
        return 0.0

    matches = greedy_match(preds, gts, iou_thd=0.0)
    sum_iou = float(sum(m[2] for m in matches))
    denom = len(preds) + len(gts) - len(matches)
    if denom <= 0:
        return 0.0
    return sum_iou / denom


def compute_G_mIoU(
    gated_submission: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    k_list: Sequence[int] = (1, 3, 5),
) -> OrderedDict[str, Any]:
    """
    Paper G-mIoU@k: compute set IoU between gated top-k predictions and GT, then
    average over all queries including empty-set cases.
    """
    pred_by_qid: Dict[Any, List] = {}
    for d in gated_submission:
        pred_by_qid[d["qid"]] = [[w[0], w[1]] for w in d.get("pred_relevant_windows", [])]

    qid2gt = {d["qid"]: d.get("relevant_windows", []) for d in ground_truth}

    out: "OrderedDict[str, Any]" = OrderedDict()
    for k in k_list:
        sample_scores: List[float] = []
        for qid, gts in qid2gt.items():
            preds = pred_by_qid.get(qid, [])[:k]
            sample_scores.append(_compute_set_iou_score(preds, gts))

        mean_score = (
            float(sum(sample_scores) / len(sample_scores)) if sample_scores else 0.0
        )
        out[f"G-mIoU@{k}"] = round(100 * mean_score, 2)
    return out
