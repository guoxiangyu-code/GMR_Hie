"""Replayable Part 2 diagnostics over a complete prediction artifact."""

from __future__ import annotations

import argparse
import json
import os

from eval.eval_main import evaluate_gmr
from eval.event_set_metrics import compute_submission_diagnostics
from eval.normalization import load_ts_window_cfg, normalize_ground_truth
from eval.utils import load_jsonl
from training.flash_vtg_gmr.contracts import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--ground_truth", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gt_ts_window_cfg")
    parser.add_argument("--map_num_workers", type=int, default=8)
    args = parser.parse_args()

    submission = load_jsonl(args.submission)
    raw_ground_truth = load_jsonl(args.ground_truth)
    ground_truth, normalization_stats = normalize_ground_truth(
        raw_ground_truth,
        load_ts_window_cfg(args.gt_ts_window_cfg),
        drop_empty_gt=False,
    )
    set_and_count = compute_submission_diagnostics(submission, ground_truth)
    maintained = evaluate_gmr(
        submission,
        ground_truth,
        map_num_workers=max(1, args.map_num_workers),
        verbose=False,
    )
    result = {
        "schema_version": "hiea2m.diagnostics.v1",
        "submission_sha256": sha256_file(args.submission),
        "ground_truth_sha256": sha256_file(args.ground_truth),
        "normalization_stats": normalization_stats,
        "diagnostics": set_and_count,
        "maintained_gmr_metrics": maintained,
    }
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    print(json.dumps({**maintained["brief"], **set_and_count}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
