"""Register one verified Part 2 run into the cardinality/public-P0 indexes."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

from training.flash_vtg_gmr.contracts import sha256_file


def _record(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def _read(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", required=True, choices=["G0-Threshold", "G0", "G0-Con", "P0", "C1", "C2"])
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--baseline_index", type=Path, required=True)
    parser.add_argument("--cardinality_root", type=Path, default=Path("artifacts/cardinality"))
    parser.add_argument("--adapter_index", type=Path, default=Path("artifacts/adapters/public_adapter_index.json"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--validation_predictions", type=Path)
    parser.add_argument("--validation_metrics", type=Path)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--opt", type=Path, required=True)
    args = parser.parse_args()

    baseline_index = _read(args.baseline_index)
    baseline_run = baseline_index.get("runs", {}).get(str(args.seed))
    if baseline_run is None:
        raise ValueError(f"Baseline index has no seed {args.seed}")
    baseline_sha = baseline_run["checkpoint"]["sha256"]
    feature_sha = baseline_index.get("feature_manifest", {}).get("sha256")
    checkpoint_record = None
    if args.variant == "G0-Threshold":
        if args.checkpoint is not None:
            raise ValueError("G0-Threshold must not register a trained checkpoint")
    else:
        if args.checkpoint is None:
            raise ValueError(f"{args.variant} requires --checkpoint")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if int(checkpoint.get("seed", -1)) != args.seed or checkpoint.get("variant") != args.variant:
            raise ValueError("Checkpoint seed/variant metadata mismatch")
        if checkpoint.get("baseline_checkpoint_sha256") != baseline_sha:
            raise ValueError("Checkpoint B0 hash mismatch")
        if checkpoint.get("feature_manifest_sha256") != feature_sha:
            raise ValueError("Checkpoint feature manifest hash mismatch")
        if checkpoint.get("selection_split") != "val":
            raise ValueError("Checkpoint was not selected on validation")
        if not checkpoint.get("training_command") or not checkpoint.get("selection_metric_names"):
            raise ValueError("Checkpoint lacks training command/selection metadata")
        if args.validation_predictions is None or args.validation_metrics is None:
            raise ValueError(f"{args.variant} requires validation predictions and metrics")
        validation_prediction_record = _record(args.validation_predictions)
        validation_metric_record = _record(args.validation_metrics)
        embedded = checkpoint.get("validation_artifacts", {})
        if embedded.get("predictions", {}).get("sha256") != validation_prediction_record["sha256"]:
            raise ValueError("Checkpoint validation prediction hash mismatch")
        if embedded.get("metrics", {}).get("sha256") != validation_metric_record["sha256"]:
            raise ValueError("Checkpoint validation metric hash mismatch")
        checkpoint_record = _record(args.checkpoint)
    if args.variant == "G0-Threshold":
        validation_prediction_record = None
        validation_metric_record = None

    if args.variant == "P0":
        if args.calibration is not None:
            raise ValueError("P0 must register calibration=null")
        calibration_record = None
    else:
        if args.calibration is None:
            raise ValueError(f"{args.variant} requires --calibration")
        calibration = _read(args.calibration)
        if calibration.get("variant") != args.variant or calibration.get("split") != "val" or int(calibration.get("seed", -1)) != args.seed:
            raise ValueError("Calibration seed/variant/split mismatch")
        if calibration.get("baseline_checkpoint_sha256") != baseline_sha:
            raise ValueError("Calibration B0 hash mismatch")
        if calibration.get("feature_manifest_sha256") != feature_sha:
            raise ValueError("Calibration feature manifest hash mismatch")
        expected_calibration_checkpoint = baseline_sha if args.variant == "G0-Threshold" else checkpoint_record["sha256"]
        if calibration.get("checkpoint_sha256") != expected_calibration_checkpoint:
            raise ValueError("Calibration checkpoint hash mismatch")
        calibration_record = _record(args.calibration)

    prediction_record = _record(args.predictions)
    metrics_record = _record(args.metrics)
    metrics_doc = _read(args.metrics)
    if metrics_doc.get("submission_sha256") != prediction_record["sha256"]:
        raise ValueError("Metrics are not bound to the registered predictions")
    replay = _read(args.replay)
    if replay.get("status") != "PASS" or replay.get("variant") != args.variant:
        raise ValueError("Prediction replay did not PASS for the registered variant")
    if replay.get("predictions_sha256") != prediction_record["sha256"]:
        raise ValueError("Prediction replay is not bound to the registered predictions")
    if args.calibration is None:
        if replay.get("calibration_sha256") is not None:
            raise ValueError("P0 replay must not reference count calibration")
    elif replay.get("calibration_sha256") != calibration_record["sha256"]:
        raise ValueError("Prediction replay is not bound to the registered calibration")
    if metrics_doc.get("ground_truth_sha256") != replay.get("ground_truth_sha256"):
        raise ValueError("Metrics/replay ground-truth binding mismatch")
    if int(replay.get("query_count", 0)) <= 0:
        raise ValueError("Prediction replay has no queries")

    option_doc = _read(args.opt)
    if int(option_doc.get("seed", -1)) != args.seed or option_doc.get("variant") != args.variant:
        raise ValueError("Resolved inference opt seed/variant mismatch")
    if option_doc.get("baseline_variant") != "B0":
        raise ValueError("Resolved inference opt must identify the indexed B0 baseline")
    if option_doc.get("eval_split_name") != "test":
        raise ValueError("Registered inference opt is not for test")
    if option_doc.get("baseline_checkpoint_sha256") != baseline_sha:
        raise ValueError("Resolved inference opt B0 hash mismatch")
    if option_doc.get("feature_manifest_sha256") != feature_sha:
        raise ValueError("Resolved inference opt feature hash mismatch")

    now = datetime.now(timezone.utc).isoformat()
    record = {
        "run_id": args.run_id,
        "seed": args.seed,
        "variant": args.variant,
        "registered_at": now,
        "train_status": "not_applicable" if args.variant == "G0-Threshold" else "complete",
        "baseline_checkpoint_sha256": baseline_sha,
        "checkpoint": checkpoint_record,
        "calibration": calibration_record,
        "selection_threshold": 0.5 if args.variant == "P0" else None,
        "validation_predictions": validation_prediction_record,
        "validation_metrics": validation_metric_record,
        "predictions": prediction_record,
        "prediction_replay": _record(args.replay),
        "metrics": metrics_record,
        "opt": _record(args.opt),
    }
    index_path = args.cardinality_root / "cardinality_index.json"
    index = _read(index_path) or {
        "schema_version": "hiea2m.cardinality-index.v1",
        "runs": {},
        "research_outcome": "MIXED",
    }
    previous = index.setdefault("runs", {}).setdefault(str(args.seed), {}).get(args.variant)
    if previous and previous.get("run_id") != args.run_id:
        index.setdefault("superseded_runs", []).append(previous)
    index["runs"][str(args.seed)][args.variant] = record
    index["updated_at"] = now
    _write(index_path, index)

    if args.variant == "P0":
        adapter_index = _read(args.adapter_index) or {
            "schema_version": "hiea2m.public-adapter-index.v1",
            "runs": {},
        }
        prior = adapter_index.setdefault("runs", {}).get(str(args.seed))
        if prior and prior.get("run_id") != args.run_id:
            adapter_index.setdefault("superseded_runs", []).append(prior)
        adapter_index["runs"][str(args.seed)] = {
            "run_id": args.run_id,
            "seed": args.seed,
            "checkpoint": checkpoint_record,
            "baseline_checkpoint_sha256": baseline_sha,
            "feature_manifest_sha256": baseline_index.get("feature_manifest", {}).get("sha256"),
            "selection_threshold": 0.5,
            "event_interface_schema": "EventInterfaceV1",
            "validation_predictions": validation_prediction_record,
            "validation_metrics": validation_metric_record,
            "predictions": prediction_record,
            "metrics": metrics_record,
            "prediction_replay": _record(args.replay),
        }
        adapter_index["updated_at"] = now
        _write(args.adapter_index, adapter_index)
    print(json.dumps(record, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
