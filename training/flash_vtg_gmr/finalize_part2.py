"""Strict, disk-backed Part 2 completion and Part 3 handoff verifier."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from training.flash_vtg_gmr.contracts import sha256_file


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _verify_record(record: object, label: str, unmet: list[str]) -> dict | None:
    if not isinstance(record, dict) or not record.get("path") or not record.get("sha256"):
        unmet.append(f"{label}: missing path/sha256 record")
        return None
    path = Path(record["path"])
    if not path.is_file():
        unmet.append(f"{label}: missing file {path}")
        return None
    actual = sha256_file(path)
    if actual != record["sha256"]:
        unmet.append(f"{label}: sha256 mismatch {actual} != {record['sha256']}")
        return None
    return {"path": str(path), "sha256": actual, "size_bytes": path.stat().st_size}


def _metric_payload(path: str) -> dict:
    document = _load_json(Path(path))
    merged = dict(document)
    if isinstance(document.get("brief"), dict):
        merged.update(document["brief"])
    if isinstance(document.get("diagnostics"), dict):
        merged.update(document["diagnostics"])
    maintained = document.get("maintained_gmr_metrics")
    if isinstance(maintained, dict) and isinstance(maintained.get("brief"), dict):
        merged.update(maintained["brief"])
    return merged


def _baseline_validation_records(baseline_run: dict, label: str, unmet: list[str]):
    manifest_record = _verify_record(
        baseline_run.get("artifact_manifest"), f"{label}/artifact_manifest", unmet
    )
    if not manifest_record:
        return None, None
    manifest = _load_json(Path(manifest_record["path"]))
    artifacts = manifest.get("artifacts", {})
    predictions = _verify_record(
        artifacts.get("val_predictions_raw"), f"{label}/val_predictions_raw", unmet
    )
    metrics = _verify_record(
        artifacts.get("val_metrics_raw"), f"{label}/val_metrics_raw", unmet
    )
    return predictions, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_manifest", type=Path, required=True)
    parser.add_argument("--data_manifest_index", type=Path, required=True)
    parser.add_argument("--baseline_index", type=Path, required=True)
    parser.add_argument("--adapter_index", type=Path, required=True)
    parser.add_argument("--cardinality_root", type=Path, required=True)
    parser.add_argument("--required_seeds", nargs="+", type=int, default=[2024, 2025])
    parser.add_argument(
        "--required_variants", nargs="+",
        default=["G0-Threshold", "G0", "G0-Con", "P0", "C1", "C2"],
    )
    parser.add_argument("--test_result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    unmet: list[str] = []
    feature_hash = sha256_file(args.feature_manifest) if args.feature_manifest.is_file() else None
    data_hash = sha256_file(args.data_manifest_index) if args.data_manifest_index.is_file() else None
    baseline_hash = sha256_file(args.baseline_index) if args.baseline_index.is_file() else None
    if feature_hash is None:
        unmet.append(f"missing feature manifest: {args.feature_manifest}")
    if data_hash is None:
        unmet.append(f"missing data manifest index: {args.data_manifest_index}")
    if baseline_hash is None:
        unmet.append(f"missing baseline index: {args.baseline_index}")

    baseline_index = _load_json(args.baseline_index) if baseline_hash else {}
    if feature_hash and baseline_index.get("feature_manifest", {}).get("sha256") != feature_hash:
        unmet.append("baseline index feature manifest hash mismatch")
    if data_hash and baseline_index.get("data_manifest_index", {}).get("sha256") != data_hash:
        unmet.append("baseline index data manifest hash mismatch")

    adapter_index = _load_json(args.adapter_index) if args.adapter_index.is_file() else {}
    if not adapter_index:
        unmet.append(f"missing/empty adapter index: {args.adapter_index}")
    cardinality_index_path = args.cardinality_root / "cardinality_index.json"
    cardinality_index = _load_json(cardinality_index_path) if cardinality_index_path.is_file() else {}
    if not cardinality_index:
        unmet.append(f"missing/empty cardinality index: {cardinality_index_path}")

    run_records = cardinality_index.get("runs", {})
    adapter_records = adapter_index.get("runs", {})
    verified_runs: dict[str, dict] = {}
    p0_gate_pass = True
    aec_map_gate_pass = True
    gate_details: dict[str, dict] = {}
    for seed in args.required_seeds:
        seed_key = str(seed)
        verified_runs[seed_key] = {}
        baseline_run = baseline_index.get("runs", {}).get(seed_key)
        if baseline_run is None:
            unmet.append(f"seed {seed}: absent from baseline index")
            continue
        baseline_checkpoint_hash = baseline_run.get("checkpoint", {}).get("sha256")
        _, baseline_val_metrics = _baseline_validation_records(
            baseline_run, f"seed {seed}/B0", unmet
        )
        gate_details[seed_key] = {}
        for variant in args.required_variants:
            label = f"seed {seed}/{variant}"
            record = run_records.get(seed_key, {}).get(variant)
            if not isinstance(record, dict):
                unmet.append(f"{label}: missing run record")
                continue
            if int(record.get("seed", -1)) != seed or record.get("variant") != variant:
                unmet.append(f"{label}: seed/variant metadata mismatch")
            if record.get("baseline_checkpoint_sha256") != baseline_checkpoint_hash:
                unmet.append(f"{label}: baseline checkpoint hash mismatch")
            prediction = _verify_record(record.get("predictions"), f"{label}/predictions", unmet)
            replay = _verify_record(record.get("prediction_replay"), f"{label}/prediction_replay", unmet)
            metrics = _verify_record(record.get("metrics"), f"{label}/metrics", unmet)
            option = _verify_record(record.get("opt"), f"{label}/opt", unmet)
            validation_predictions = None
            validation_metrics = None
            checkpoint = None
            calibration = None
            if variant == "G0-Threshold":
                if record.get("train_status") != "not_applicable":
                    unmet.append(f"{label}: train_status must be not_applicable")
                if record.get("checkpoint") is not None:
                    unmet.append(f"{label}: must reference B0 hash, not a trained checkpoint")
            else:
                if record.get("train_status") != "complete":
                    unmet.append(f"{label}: train_status must be complete")
                checkpoint = _verify_record(record.get("checkpoint"), f"{label}/checkpoint", unmet)
                validation_predictions = _verify_record(
                    record.get("validation_predictions"), f"{label}/validation_predictions", unmet
                )
                validation_metrics = _verify_record(
                    record.get("validation_metrics"), f"{label}/validation_metrics", unmet
                )
            if variant == "P0":
                if record.get("calibration") is not None:
                    unmet.append(f"{label}: P0 calibration must be null")
                if float(record.get("selection_threshold", -1)) != 0.5:
                    unmet.append(f"{label}: P0 selection_threshold must be 0.5")
                public_record = adapter_records.get(seed_key)
                if not isinstance(public_record, dict):
                    unmet.append(f"{label}: missing public adapter record")
                else:
                    if checkpoint and public_record.get("checkpoint", {}).get("sha256") != checkpoint["sha256"]:
                        unmet.append(f"{label}: public adapter checkpoint hash mismatch")
                    if prediction and public_record.get("predictions", {}).get("sha256") != prediction["sha256"]:
                        unmet.append(f"{label}: public adapter predictions hash mismatch")
                    if metrics and public_record.get("metrics", {}).get("sha256") != metrics["sha256"]:
                        unmet.append(f"{label}: public adapter metrics hash mismatch")
                    if replay and public_record.get("prediction_replay", {}).get("sha256") != replay["sha256"]:
                        unmet.append(f"{label}: public adapter replay hash mismatch")
                    if validation_predictions and public_record.get("validation_predictions", {}).get("sha256") != validation_predictions["sha256"]:
                        unmet.append(f"{label}: public adapter validation predictions hash mismatch")
                    if validation_metrics and public_record.get("validation_metrics", {}).get("sha256") != validation_metrics["sha256"]:
                        unmet.append(f"{label}: public adapter validation metrics hash mismatch")
                    if public_record.get("baseline_checkpoint_sha256") != baseline_checkpoint_hash:
                        unmet.append(f"{label}: public adapter B0 hash mismatch")
                    if public_record.get("feature_manifest_sha256") != feature_hash:
                        unmet.append(f"{label}: public adapter feature hash mismatch")
                    if public_record.get("event_interface_schema") != "EventInterfaceV1":
                        unmet.append(f"{label}: public adapter schema mismatch")
                    if float(public_record.get("selection_threshold", -1)) != 0.5:
                        unmet.append(f"{label}: public adapter selection threshold mismatch")
            else:
                calibration = _verify_record(record.get("calibration"), f"{label}/calibration", unmet)
                if calibration:
                    calibration_doc = _load_json(Path(calibration["path"]))
                    if calibration_doc.get("split") != "val" or calibration_doc.get("variant") != variant:
                        unmet.append(f"{label}: calibration provenance is not validation-only/same variant")
                    if variant in {"G0", "G0-Con"} and not calibration_doc.get("raw_threshold_calibration_sha256"):
                        unmet.append(f"{label}: missing frozen G0-Threshold calibration hash")
            verified_runs[seed_key][variant] = {
                "run_id": record.get("run_id"),
                "checkpoint": checkpoint,
                "calibration": calibration,
                "predictions": prediction,
                "prediction_replay": replay,
                "metrics": metrics,
                "opt": option,
                "validation_predictions": validation_predictions,
                "validation_metrics": validation_metrics,
            }
            if replay:
                replay_doc = _load_json(Path(replay["path"]))
                if replay_doc.get("status") != "PASS" or replay_doc.get("variant") != variant:
                    unmet.append(f"{label}: prediction replay status/variant mismatch")
                if prediction and replay_doc.get("predictions_sha256") != prediction["sha256"]:
                    unmet.append(f"{label}: prediction replay hash binding mismatch")
                expected_calibration = calibration["sha256"] if calibration else None
                if replay_doc.get("calibration_sha256") != expected_calibration:
                    unmet.append(f"{label}: prediction replay calibration binding mismatch")
                if metrics and prediction:
                    metrics_doc = _load_json(Path(metrics["path"]))
                    if metrics_doc.get("submission_sha256") != prediction["sha256"]:
                        unmet.append(f"{label}: metrics prediction hash binding mismatch")
                    if metrics_doc.get("ground_truth_sha256") != replay_doc.get("ground_truth_sha256"):
                        unmet.append(f"{label}: metrics/replay ground-truth binding mismatch")
            if option:
                option_doc = _load_json(Path(option["path"]))
                if int(option_doc.get("seed", -1)) != seed or option_doc.get("variant") != variant:
                    unmet.append(f"{label}: resolved opt seed/variant mismatch")
                if option_doc.get("baseline_variant") != "B0" or option_doc.get("eval_split_name") != "test":
                    unmet.append(f"{label}: resolved opt is not indexed-B0 test inference")
                if option_doc.get("baseline_checkpoint_sha256") != baseline_checkpoint_hash:
                    unmet.append(f"{label}: resolved opt B0 hash mismatch")
                if option_doc.get("feature_manifest_sha256") != feature_hash:
                    unmet.append(f"{label}: resolved opt feature hash mismatch")

            if validation_metrics and variant == "P0":
                payload = _metric_payload(validation_metrics["path"])
                raw_fc = float(payload.get("Raw-Proposal-Oracle-FullCoverage@0.5", float("nan")))
                mode_fc = float(payload.get("Oracle-Mode-FullCoverage@0.5", float("nan")))
                p0_map = float(payload.get("mAP", payload.get("MR-full-mAP", float("nan"))))
                b0_payload = _metric_payload(baseline_val_metrics["path"]) if baseline_val_metrics else {}
                b0_map = float(b0_payload.get("mAP", b0_payload.get("MR-full-mAP", float("nan"))))
                gate = all(math.isfinite(value) for value in (raw_fc, mode_fc, p0_map, b0_map))
                gate = gate and (raw_fc - mode_fc <= 0.05) and (b0_map - p0_map <= 0.5)
                p0_gate_pass = p0_gate_pass and gate
                gate_details[seed_key]["P0"] = {
                    "split": "val",
                    "baseline_mAP": b0_map,
                    "p0_mAP": p0_map,
                    "mAP_drop": b0_map - p0_map,
                    "raw_oracle_full_coverage": raw_fc,
                    "mode_oracle_full_coverage": mode_fc,
                    "coverage_drop": raw_fc - mode_fc,
                    "pass": gate,
                }
            elif variant == "P0":
                p0_gate_pass = False
            if metrics and variant in {"C1", "C2"}:
                payload = _metric_payload(metrics["path"])
                p0_record = run_records.get(seed_key, {}).get("P0", {})
                p0_metrics_record = p0_record.get("metrics", {}) if isinstance(p0_record, dict) else {}
                if p0_metrics_record.get("path") and Path(p0_metrics_record["path"]).is_file():
                    p0_payload = _metric_payload(p0_metrics_record["path"])
                    current_map = float(payload.get("mAP", payload.get("MR-full-mAP", float("nan"))))
                    p0_map = float(p0_payload.get("mAP", p0_payload.get("MR-full-mAP", float("nan"))))
                    aec_map_gate_pass = aec_map_gate_pass and math.isfinite(current_map) and math.isfinite(p0_map) and (p0_map - current_map <= 0.5)
                else:
                    aec_map_gate_pass = False

    test_result = _load_json(args.test_result) if args.test_result.is_file() else {}
    if test_result.get("status") != "PASS" or not test_result.get("command"):
        unmet.append("test_result missing or not PASS with a replayable command")
    aggregate_record = cardinality_index.get("aggregate_metrics")
    report_record = cardinality_index.get("report")
    aggregate_verified = _verify_record(aggregate_record, "aggregate_metrics", unmet)
    report_verified = _verify_record(report_record, "part2_report", unmet)

    status = "COMPLETE" if not unmet else "INCOMPLETE"
    part3_handoff = "READY" if status == "COMPLETE" and p0_gate_pass and aec_map_gate_pass else "NOT_READY"
    result = {
        "schema_version": "hiea2m.part2-completion.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "part3_handoff": part3_handoff,
        "research_outcome": cardinality_index.get("research_outcome", "MIXED"),
        "feature_manifest_hash": feature_hash,
        "data_manifest_hash": data_hash,
        "baseline_index_hash": baseline_hash,
        "required_variants": args.required_variants,
        "required_seeds": args.required_seeds,
        "runs": verified_runs,
        "test_command": test_result.get("command"),
        "test_result": test_result,
        "aggregate_metrics_path": aggregate_verified,
        "report_path": report_verified,
        "p0_gate_pass": p0_gate_pass,
        "aec_map_gate_pass": aec_map_gate_pass,
        "gate_details": gate_details,
        "unmet_requirements": unmet,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": status, "part3_handoff": part3_handoff, "unmet_requirements": unmet}, indent=2, ensure_ascii=False))
    if status != "COMPLETE":
        sys.exit(1)


if __name__ == "__main__":
    main()
