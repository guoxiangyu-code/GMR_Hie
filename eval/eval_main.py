# -*- coding: utf-8 -*-
"""
Main Soccer-GMR evaluation entry point for prediction and GT JSONL files.

Depends on normalization.py, metrics.py, and utils.py in the same directory.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import OrderedDict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

try:
    from .metrics import (
        DEFAULT_IOU_THRESHOLDS,
        compute_G_mIoU,
        compute_gmr_cls,
        compute_mAP,
        compute_mIoU,
        compute_mIoU_plus,
        compute_mR,
        compute_mR_plus,
        get_existence_score,
        prepare_submission_for_gmiou,
    )
    from .normalization import load_ts_window_cfg, normalize_ground_truth
    from .utils import load_jsonl
except ImportError:  # Support direct execution as `python eval/eval_main.py`.
    from metrics import (
        DEFAULT_IOU_THRESHOLDS,
        compute_G_mIoU,
        compute_gmr_cls,
        compute_mAP,
        compute_mIoU,
        compute_mIoU_plus,
        compute_mR,
        compute_mR_plus,
        get_existence_score,
        prepare_submission_for_gmiou,
    )
    from normalization import load_ts_window_cfg, normalize_ground_truth
    from utils import load_jsonl


def validate_qid_coverage(
    submission: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]
) -> None:
    pred_qids = [item.get("qid") for item in submission]
    gt_qids = [item.get("qid") for item in ground_truth]
    if any(qid is None for qid in pred_qids):
        raise ValueError("Every submission row must contain qid")
    if any(qid is None for qid in gt_qids):
        raise ValueError("Every ground-truth row must contain qid")
    if len(set(pred_qids)) != len(pred_qids):
        raise ValueError("Submission contains duplicate qids")
    if len(set(gt_qids)) != len(gt_qids):
        raise ValueError("Ground truth contains duplicate qids")
    pred_set = set(pred_qids)
    gt_set = set(gt_qids)
    if pred_set != gt_set:
        raise ValueError(
            "Submission/GT qid coverage mismatch: "
            f"missing_predictions={sorted(gt_set - pred_set, key=str)[:20]}, "
            f"unexpected_predictions={sorted(pred_set - gt_set, key=str)[:20]}"
        )


def evaluate_gmr(
    submission: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    *,
    k_list: Sequence[int] = (1, 3, 5),
    max_pred_windows: int = 10,
    cls_thresholds: Tuple[float, ...] = (0.4, 0.6),
    gmiou_cls_threshold: float = 0.4,
    iou_thds: np.ndarray = DEFAULT_IOU_THRESHOLDS,
    map_num_workers: int = 8,
    verbose: bool = True,
) -> "OrderedDict[str, Any]":
    """
    Compute the full GMR metric suite: CLS, G-mIoU@k for k_list, and mAP / mR /
    mR+ / mIoU / mIoU+ on positive queries.
    """
    validate_qid_coverage(submission, ground_truth)
    start = time.time()

    n_pos = sum(1 for d in ground_truth if len(d.get("relevant_windows", [])) > 0)
    n_multi = sum(1 for d in ground_truth if len(d.get("relevant_windows", [])) >= 2)
    n_neg = len(ground_truth) - n_pos

    results: "OrderedDict[str, Any]" = OrderedDict()
    brief: "OrderedDict[str, Any]" = OrderedDict()

    cls = compute_gmr_cls(submission, ground_truth, thresholds=cls_thresholds)
    brief["AUROC"] = cls["AUROC"]
    for thd_str, metrics in cls["per_threshold"].items():
        brief[f"Rej-F1@{thd_str}"] = metrics["Rej-F1"]
        brief[f"Acc@{thd_str}"] = metrics["Acc"]
    results["GMR-CLS"] = cls

    gated_sub, gmiou_gate = prepare_submission_for_gmiou(
        submission,
        cls_threshold=gmiou_cls_threshold,
        max_pred_windows=max_pred_windows,
    )
    gmiou_res = compute_G_mIoU(gated_sub, ground_truth, k_list=k_list)
    brief.update(gmiou_res)
    results["G-mIoU_gate"] = gmiou_gate
    results["G-mIoU_detail"] = gmiou_res

    pred_by_qid = {item["qid"]: item for item in submission}
    gated_by_qid = {item["qid"]: item for item in gated_sub}
    grouped: "OrderedDict[str, Any]" = OrderedDict()
    grouped_gt = {
        "null": [item for item in ground_truth if len(item.get("relevant_windows", [])) == 0],
        "single": [item for item in ground_truth if len(item.get("relevant_windows", [])) == 1],
        "multi": [item for item in ground_truth if len(item.get("relevant_windows", [])) >= 2],
    }
    null_metrics: "OrderedDict[str, Any]" = OrderedDict(
        [("count", len(grouped_gt["null"]))]
    )
    for threshold in sorted(cls_thresholds):
        false_positives = sum(
            get_existence_score(pred_by_qid[item["qid"]])[0] > threshold
            for item in grouped_gt["null"]
        )
        rate = false_positives / len(grouped_gt["null"]) if grouped_gt["null"] else 0.0
        key = f"FPR@{threshold:.2f}"
        null_metrics[key] = round(100 * rate, 2)
        brief[f"Null-{key}"] = null_metrics[key]
    grouped["null"] = null_metrics
    for group_name in ("single", "multi"):
        gt_group = grouped_gt[group_name]
        gated_group = [gated_by_qid[item["qid"]] for item in gt_group]
        metrics_group = compute_G_mIoU(gated_group, gt_group, k_list=k_list)
        grouped[group_name] = {"count": len(gt_group), **metrics_group}
        for key, value in metrics_group.items():
            brief[f"{group_name.title()}-{key}"] = value
    results["grouped"] = grouped

    pos_qids = {d["qid"] for d in ground_truth if len(d.get("relevant_windows", [])) > 0}
    gt_pos = [d for d in ground_truth if d["qid"] in pos_qids]
    sub_pos = [d for d in submission if d.get("qid") in pos_qids]

    if len(gt_pos) == 0:
        raise ValueError("No positive GT samples; localization metrics cannot be computed.")

    map_res = compute_mAP(
        sub_pos,
        gt_pos,
        iou_thds=iou_thds,
        max_pred_windows=max_pred_windows,
        num_workers=map_num_workers,
    )
    m_r_res = compute_mR(sub_pos, gt_pos, k_list=k_list, iou_thds=iou_thds)
    m_r_plus_res = compute_mR_plus(sub_pos, gt_pos, k_list=k_list, iou_thds=iou_thds)
    miou_res = compute_mIoU(sub_pos, gt_pos, k_list=k_list)
    miou_plus_res = compute_mIoU_plus(sub_pos, gt_pos, k_list=k_list)

    brief["mAP"] = map_res["mAP"]
    for k in k_list:
        brief[f"mR@{k}"] = m_r_res[f"mR@{k}"]
    for k in k_list:
        brief[f"mR+@{k}"] = m_r_plus_res.get(f"mR+@{k}", 0.0)
    for k in k_list:
        brief[f"mIoU@{k}"] = miou_res[f"mIoU@{k}"]
    for k in k_list:
        brief[f"mIoU+@{k}"] = miou_plus_res.get(f"mIoU+@{k}", 0.0)

    results["brief"] = brief
    results["mAP_detail"] = map_res
    results["mR_detail"] = m_r_res
    results["mR+_detail"] = m_r_plus_res
    results["mIoU_detail"] = miou_res
    results["mIoU+_detail"] = miou_plus_res
    results["stats"] = {
        "num_total": len(ground_truth),
        "num_positive": n_pos,
        "num_negative": n_neg,
        "num_multi_instance": n_multi,
        "num_single_instance": n_pos - n_multi,
        "k_list": list(k_list),
        "cls_thresholds": list(cls_thresholds),
        "gmiou_cls_threshold": gmiou_cls_threshold,
        "eval_time_sec": round(time.time() - start, 2),
    }

    if verbose:
        print(
            f"[eval_main] {n_pos} positive ({n_pos - n_multi} single + {n_multi} multi), "
            f"{n_neg} negative, time={time.time() - start:.1f}s"
        )
        print(json.dumps(brief, indent=2, ensure_ascii=False))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Soccer-GMR Evaluation (full GMR metrics)")
    parser.add_argument("--submission_path", type=str, required=True, help="Prediction JSONL")
    parser.add_argument("--gt_path", type=str, required=True, help="GT JSONL")
    parser.add_argument("--save_path", type=str, required=True, help="Output metrics JSON")
    parser.add_argument(
        "--gt_ts_window_cfg",
        type=str,
        default=None,
        help="Timestamp-window expansion config JSON, required when GT uses timestamp moments",
    )
    parser.add_argument(
        "--k_list",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="k values for mR / mR+ / mIoU / mIoU+ / G-mIoU (default: 1 3 5)",
    )
    parser.add_argument(
        "--max_pred_windows",
        type=int,
        default=10,
        help="Maximum retained prediction windows for mAP and G-mIoU gating (default: 10)",
    )
    parser.add_argument(
        "--cls_thresholds",
        type=float,
        nargs="+",
        default=[0.4, 0.6],
        help="Thresholds for reporting GMR-CLS Rej-F1 / Acc",
    )
    parser.add_argument(
        "--gmiou_cls_threshold",
        type=float,
        default=0.4,
        help="Existence-score threshold \\tau used for G-mIoU@k gating (default: 0.4)",
    )
    parser.add_argument(
        "--map_num_workers",
        type=int,
        default=8,
        help="Number of mAP worker processes; <=1 or small inputs use single-thread mode",
    )
    parser.add_argument("--not_verbose", action="store_true", help="Run quietly")
    args = parser.parse_args()

    verbose = not args.not_verbose

    submission = load_jsonl(args.submission_path)
    gt_raw = load_jsonl(args.gt_path)
    ts_cfg = load_ts_window_cfg(args.gt_ts_window_cfg)

    # Keep empty-set GT samples for CLS and G-mIoU.
    gt, gt_stats = normalize_ground_truth(gt_raw, ts_cfg, drop_empty_gt=False)

    pred_qids = {e["qid"] for e in submission if isinstance(e, dict) and "qid" in e}
    gt_qids = {e["qid"] for e in gt}

    if verbose:
        print(f"[eval_main] GT: {json.dumps(gt_stats, ensure_ascii=False)}")
        print(
            f"[eval_main] submission={len(pred_qids)}, gt={len(gt_qids)}, "
            f"gt_only={len(gt_qids - pred_qids)}, "
            f"pred_only={len(pred_qids - gt_qids)}"
        )

    results = evaluate_gmr(
        submission,
        gt,
        k_list=tuple(args.k_list),
        max_pred_windows=args.max_pred_windows,
        cls_thresholds=tuple(args.cls_thresholds),
        gmiou_cls_threshold=args.gmiou_cls_threshold,
        iou_thds=DEFAULT_IOU_THRESHOLDS,
        map_num_workers=args.map_num_workers,
        verbose=verbose,
    )

    save_dir = os.path.dirname(args.save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(args.save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    if verbose:
        print(f"[eval_main] Saved -> {args.save_path}")


if __name__ == "__main__":
    main()
