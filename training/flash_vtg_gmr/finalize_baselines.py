"""Freeze completed B0 runs into the Part 1 handoff artifact contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch


SCHEMA_VERSION = "hiea2m.baseline-index.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def write_hashed_json(path: Path, payload: Dict[str, Any]) -> None:
    payload = dict(payload)
    payload["payload_sha256"] = canonical_json_sha256(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_content_hashed_json(path: Path) -> Dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    stored = document.get("content_sha256")
    payload = dict(document)
    payload.pop("content_sha256", None)
    if stored != canonical_json_sha256(payload):
        raise ValueError(f"Content SHA256 mismatch: {path}")
    return document


def relative_path(path: Path, root: Path) -> str:
    return os.path.relpath(path.resolve(), root.resolve())


def file_record(path: Path, root: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": relative_path(path, root),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def tensor_state_sha256(state: Dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if not torch.is_tensor(value):
            raise TypeError(f"Non-tensor model state entry: {key}={type(value)}")
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def structured_state_sha256(value: Any) -> str:
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"ndarray")
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(json.dumps(list(array.shape)).encode("ascii"))
            digest.update(array.tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=str):
                update(str(key))
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode("ascii"))
            for child in item:
                update(child)
        elif item is None or isinstance(item, (str, int, float, bool)):
            digest.update(repr(item).encode("utf-8"))
        else:
            digest.update(repr(item).encode("utf-8"))

    update(value)
    return digest.hexdigest()


def ensure_relative_symlink(link: Path, target: Path) -> None:
    if link.exists() or link.is_symlink():
        if not link.is_symlink():
            raise FileExistsError(f"Refusing to replace non-symlink artifact: {link}")
        link.unlink()
    link.symlink_to(os.path.relpath(target, link.parent))


def discover_run(seed_dir: Path) -> Path:
    candidates = sorted(
        path
        for path in seed_dir.glob("hl-video_tef-soccer_gmr-*")
        if path.is_dir() and (path / "model_best.ckpt").is_file()
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one completed production run under {seed_dir}, got {candidates}"
        )
    return candidates[0]


def finalize_seed(
    root: Path,
    baseline_root: Path,
    seed: int,
    variant: str,
    feature_manifest: Path,
    data_manifest_index: Path,
) -> Dict[str, Any]:
    seed_dir = baseline_root / str(seed)
    run_dir = discover_run(seed_dir)
    opt_path = run_dir / "opt.json"
    opt = json.loads(opt_path.read_text(encoding="utf-8"))
    if int(opt.get("seed", -1)) != seed:
        raise ValueError(f"Seed mismatch in {opt_path}: {opt.get('seed')} != {seed}")
    if opt.get("baseline_variant") != variant:
        raise ValueError(
            f"Variant mismatch in {opt_path}: {opt.get('baseline_variant')} != {variant}"
        )
    if Path(opt.get("feature_manifest", "")).resolve() != feature_manifest:
        raise ValueError(f"Feature manifest mismatch in {opt_path}")
    if Path(opt.get("data_manifest_index", "")).resolve() != data_manifest_index:
        raise ValueError(f"Data manifest index mismatch in {opt_path}")

    run_files = {
        "checkpoint": run_dir / "model_best.ckpt",
        "opt": opt_path,
        "command": run_dir / "command.txt",
        "environment": run_dir / "environment.txt",
        "predictions_raw": run_dir / "hl_test_submission.jsonl",
        "predictions_legacy_nms": run_dir / "hl_test_submission_nms_thd_0.7.jsonl",
        "metrics_raw": run_dir / "flash_vtg_gmr_test_results_raw.json",
        "metrics": run_dir / "flash_vtg_gmr_test_results_nms.json",
        "val_predictions_raw": run_dir / "best_hl_val_preds.jsonl",
        "val_predictions_legacy_nms": run_dir / "best_hl_val_preds_nms_thd_0.7.jsonl",
        "val_metrics_raw": run_dir / "best_hl_val_preds_metrics.json",
        "val_metrics_legacy_nms": run_dir / "best_hl_val_preds_nms_thd_0.7_metrics.json",
        "code_archive": run_dir / "code.zip",
    }
    for path in run_files.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    checkpoint = torch.load(
        run_files["checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    model_state_hash = tensor_state_sha256(checkpoint["model"])
    rng_state = checkpoint.get("reproducibility_state")
    if checkpoint.get("checkpoint_boundary") != "epoch":
        raise ValueError(f"Checkpoint is not an epoch-boundary snapshot: {run_files['checkpoint']}")
    if rng_state is None:
        raise ValueError(f"Checkpoint lacks reproducibility_state: {run_files['checkpoint']}")
    reproducibility = {
        "schema_version": "hiea2m.reproducibility.v1",
        "seed": seed,
        "variant": variant,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_boundary": checkpoint.get("checkpoint_boundary"),
        "global_step": None if rng_state is None else rng_state.get("global_step"),
        "dataset_epoch": None if rng_state is None else rng_state.get("dataset_epoch"),
        "rng_state_sha256": structured_state_sha256(rng_state),
        "canonical_tensor_state_sha256": model_state_hash,
        "checkpoint_file_sha256": sha256_file(run_files["checkpoint"]),
    }
    reproducibility_path = seed_dir / "reproducibility.json"
    write_hashed_json(reproducibility_path, reproducibility)

    aliases = {
        "model_best.ckpt": run_files["checkpoint"],
        "opt.json": run_files["opt"],
        "command.txt": run_files["command"],
        "environment.txt": run_files["environment"],
        "predictions_raw.jsonl": run_files["predictions_raw"],
        "predictions_legacy_nms.jsonl": run_files["predictions_legacy_nms"],
        "metrics.json": run_files["metrics"],
        "metrics_raw.json": run_files["metrics_raw"],
    }
    for name, target in aliases.items():
        ensure_relative_symlink(seed_dir / name, target)

    artifact_records = {
        name: file_record(path, root) for name, path in run_files.items()
    }
    artifact_records["reproducibility"] = file_record(reproducibility_path, root)
    metrics = json.loads(run_files["metrics"].read_text(encoding="utf-8"))
    artifact_manifest = {
        "schema_version": "hiea2m.baseline-artifact.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "variant": variant,
        "run_dir": relative_path(run_dir, root),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "canonical_tensor_state_sha256": model_state_hash,
        "artifacts": artifact_records,
        "test_metrics_brief": metrics["brief"],
    }
    artifact_manifest_path = seed_dir / "artifact_manifest.json"
    write_hashed_json(artifact_manifest_path, artifact_manifest)

    return {
        "seed": seed,
        "variant": variant,
        "run_dir": relative_path(run_dir, root),
        "checkpoint": file_record(run_files["checkpoint"], root),
        "canonical_tensor_state_sha256": model_state_hash,
        "predictions_raw": file_record(run_files["predictions_raw"], root),
        "predictions_legacy_nms": file_record(
            run_files["predictions_legacy_nms"], root
        ),
        "metrics": file_record(run_files["metrics"], root),
        "reproducibility": file_record(reproducibility_path, root),
        "artifact_manifest": file_record(artifact_manifest_path, root),
        "test_metrics_brief": metrics["brief"],
    }


def aggregate_metrics(runs: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    runs = list(runs)
    keys = set(runs[0]["test_metrics_brief"])
    for run in runs[1:]:
        keys &= set(run["test_metrics_brief"])
    aggregate: Dict[str, Any] = {}
    for key in sorted(keys):
        values = [run["test_metrics_brief"][key] for run in runs]
        if all(isinstance(value, (int, float)) for value in values):
            array = np.asarray(values, dtype=np.float64)
            aggregate[key] = {
                "values": values,
                "mean": round(float(array.mean()), 6),
                "std": round(float(array.std(ddof=1)), 6) if len(array) > 1 else 0.0,
            }
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_root", type=Path, default=Path("artifacts/baselines"))
    parser.add_argument("--feature_manifest", type=Path, required=True)
    parser.add_argument("--data_manifest_index", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2024, 2025, 2026])
    parser.add_argument("--variant", default="B0", choices=["B0"])
    args = parser.parse_args()

    root = Path.cwd().resolve()
    baseline_root = args.baseline_root.resolve()
    feature_manifest = args.feature_manifest.resolve()
    data_manifest_index = args.data_manifest_index.resolve()
    feature_document = validate_content_hashed_json(feature_manifest)
    data_document = validate_content_hashed_json(data_manifest_index)
    if feature_document.get("setting") != "f-lighthouse":
        raise ValueError("B0 finalization requires setting=f-lighthouse")
    if data_document.get("feature_manifest", {}).get("content_sha256") != (
        feature_document.get("content_sha256")
    ):
        raise ValueError("Data manifest index was built from another feature manifest")
    runs = [
        finalize_seed(
            root,
            baseline_root,
            seed,
            args.variant,
            feature_manifest,
            data_manifest_index,
        )
        for seed in args.seeds
    ]
    index = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "variant": args.variant,
        "seeds": list(args.seeds),
        "feature_manifest": file_record(feature_manifest, root),
        "data_manifest_index": file_record(data_manifest_index, root),
        "runs": {str(run["seed"]): run for run in runs},
        "test_metrics_aggregate": aggregate_metrics(runs),
    }
    output = baseline_root / "baseline_index.json"
    write_hashed_json(output, index)
    print(output)


if __name__ == "__main__":
    main()
