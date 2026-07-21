"""Production event-set and cardinality diagnostics for HieA2M Part 2."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment


def temporal_iou(a: list[float], b: list[float]) -> float:
    inter = max(0.0, min(float(a[1]), float(b[1])) - max(float(a[0]), float(b[0])))
    union = (float(a[1]) - float(a[0])) + (float(b[1]) - float(b[0])) - inter
    return inter / max(union, 1e-9) if inter > 0 else 0.0


def maximum_cardinality_matching(
    pred_set: list[list[float]], gt_set: list[list[float]], theta: float = 0.5
) -> list[tuple[int, int]]:
    """Maximise eligible-pair cardinality, then total tIoU."""
    p_count, g_count = len(pred_set), len(gt_set)
    if p_count == 0 or g_count == 0:
        return []
    ious = np.asarray(
        [[temporal_iou(pred, gt) for gt in gt_set] for pred in pred_set],
        dtype=np.float64,
    )
    eligible = ious >= float(theta)
    # One extra match must dominate every possible change in summed IoU.
    bonus = float(min(p_count, g_count) + 1)
    weights = np.where(eligible, bonus + ious, 0.0)
    rows, cols = linear_sum_assignment(-weights)
    return [
        (int(row), int(col))
        for row, col in zip(rows, cols)
        if eligible[row, col]
    ]


def compute_event_set_metrics(
    pairs: Iterable[tuple[list[list[float]], list[list[float]]]],
    theta: float = 0.5,
) -> dict[str, float | int]:
    total_eligible = 0
    total_duplicate = 0
    full_coverage: list[int] = []
    set_success: list[int] = []
    for pred_set, gt_set in pairs:
        matching = maximum_cardinality_matching(pred_set, gt_set, theta)
        matched = len(matching)
        eligible_predictions = sum(
            any(temporal_iou(pred, gt) >= theta for gt in gt_set)
            for pred in pred_set
        )
        total_eligible += eligible_predictions
        total_duplicate += max(eligible_predictions - matched, 0)
        if len(gt_set) >= 2:
            full_coverage.append(int(matched == len(gt_set)))
        set_success.append(int(len(pred_set) == len(gt_set) == matched))
    return {
        "DuplicateRate": total_duplicate / max(total_eligible, 1),
        "FullCoverage": float(np.mean(full_coverage)) if full_coverage else float("nan"),
        "SetSuccess": float(np.mean(set_success)) if set_success else float("nan"),
        "n_fc_queries": len(full_coverage),
        "n_ss_queries": len(set_success),
    }


def _binary_ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    result = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        include_right = index == bins - 1
        mask = (probabilities >= edges[index]) & (
            probabilities <= edges[index + 1] if include_right else probabilities < edges[index + 1]
        )
        if mask.any():
            result += float(mask.mean()) * abs(float(labels[mask].mean()) - float(probabilities[mask].mean()))
    return result


def compute_submission_diagnostics(submission: list[dict], ground_truth: list[dict]) -> dict:
    """Compute replayable diagnostics and require exact qid coverage."""
    pred_by_qid = {str(row["qid"]): row for row in submission}
    gt_by_qid = {str(row["qid"]): row for row in ground_truth}
    if set(pred_by_qid) != set(gt_by_qid):
        missing = sorted(set(gt_by_qid) - set(pred_by_qid))[:10]
        extra = sorted(set(pred_by_qid) - set(gt_by_qid))[:10]
        raise ValueError(f"QID coverage mismatch: missing={missing}, extra={extra}")

    selected_pairs = []
    oracle_pairs = []
    raw_pairs = []
    gt_counts = []
    pred_classes = []
    selected_counts = []
    nonempty_probabilities = []
    grouped_rows: list[tuple[int, list, list, int]] = []
    for qid in sorted(gt_by_qid):
        pred = pred_by_qid[qid]
        gt_windows = gt_by_qid[qid].get("relevant_windows") or []
        pred_windows = [window[:2] for window in pred.get("pred_relevant_windows", [])]
        selected_pairs.append((pred_windows, gt_windows))
        if "oracle_mode_windows" in pred:
            oracle_pairs.append((pred["oracle_mode_windows"], gt_windows))
        if "raw_proposal_windows" in pred:
            raw_pairs.append((pred["raw_proposal_windows"], gt_windows))
        gt_count = len(gt_windows)
        selected_count = len(pred_windows)
        pred_class = int(pred.get("pred_count", min(selected_count, 4)))
        probabilities = pred.get("pred_count_probs")
        p_nonempty = 1.0 - float(probabilities[0]) if probabilities else float(pred_class > 0)
        gt_counts.append(gt_count)
        selected_counts.append(selected_count)
        pred_classes.append(pred_class)
        nonempty_probabilities.append(p_nonempty)
        grouped_rows.append((gt_count, pred_windows, gt_windows, pred_class))

    selected = compute_event_set_metrics(selected_pairs, theta=0.5)
    oracle = compute_event_set_metrics(oracle_pairs, theta=0.5) if oracle_pairs else None
    raw = compute_event_set_metrics(raw_pairs, theta=0.5) if raw_pairs else None
    gt_array = np.asarray(gt_counts, dtype=np.int64)
    pred_class_array = np.asarray(pred_classes, dtype=np.int64)
    selected_array = np.asarray(selected_counts, dtype=np.int64)
    gt_class = np.minimum(gt_array, 4)
    exist_labels = (gt_array > 0).astype(np.int64)
    exist_probs = np.asarray(nonempty_probabilities, dtype=np.float64)
    exist_pred = (pred_class_array > 0).astype(np.int64)
    tp = int(((exist_labels == 1) & (exist_pred == 1)).sum())
    fp = int(((exist_labels == 0) & (exist_pred == 1)).sum())
    fn = int(((exist_labels == 1) & (exist_pred == 0)).sum())
    tn = int(((exist_labels == 0) & (exist_pred == 0)).sum())
    try:
        from sklearn.metrics import roc_auc_score
        auroc = float(roc_auc_score(exist_labels, exist_probs)) if len(set(exist_labels.tolist())) > 1 else float("nan")
    except ImportError:
        auroc = float("nan")
    rejection_f1 = (2 * tp / max(2 * tp + fp + fn, 1))

    grouped = {}
    for name, predicate in (
        ("null", lambda count: count == 0),
        ("single", lambda count: count == 1),
        ("multi", lambda count: count >= 2),
    ):
        rows = [row for row in grouped_rows if predicate(row[0])]
        pair_rows = [(row[1], row[2]) for row in rows]
        grouped[name] = {
            "query_count": len(rows),
            "Count-Acc-5": float(np.mean([min(row[0], 4) == row[3] for row in rows])) if rows else float("nan"),
            "SetSuccess@0.5": compute_event_set_metrics(pair_rows)["SetSuccess"] if rows else float("nan"),
        }

    return {
        "query_count": len(gt_counts),
        "SetSuccess@0.5": selected["SetSuccess"],
        "DuplicateRate@0.5": selected["DuplicateRate"],
        "Selected-FullCoverage@0.5": selected["FullCoverage"],
        "Oracle-Mode-FullCoverage@0.5": oracle["FullCoverage"] if oracle else float("nan"),
        "Raw-Proposal-Oracle-FullCoverage@0.5": raw["FullCoverage"] if raw else float("nan"),
        "Count-Acc-5": float(np.mean(pred_class_array == gt_class)),
        "Count-Acc-Exact-Selected": float(np.mean(selected_array == gt_array)),
        "Count-MAE-Selected": float(np.mean(np.abs(selected_array - gt_array))),
        "OverPredictionRate": float(np.mean(selected_array > gt_array)),
        "UnderPredictionRate": float(np.mean(selected_array < gt_array)),
        "AUROC": auroc,
        "Rej-F1": rejection_f1,
        "Null-FPR": fp / max(fp + tn, 1),
        "Nonempty-ECE": _binary_ece(exist_labels, exist_probs),
        "Nonempty-Brier": float(np.mean((exist_probs - exist_labels) ** 2)),
        "grouped_metrics": grouped,
    }
