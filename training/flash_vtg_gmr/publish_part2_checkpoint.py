"""Publish a hash-complete validation-selected Part 2 checkpoint.

This is primarily a compatibility finalizer for checkpoints produced before
validation artifact hashes became part of the checkpoint payload.  It never
overwrites the training snapshot; the output becomes the immutable public/run
checkpoint consumed by registration and downstream variants.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from training.flash_vtg_gmr.contracts import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation_predictions", type=Path, required=True)
    parser.add_argument("--validation_metrics", type=Path, required=True)
    parser.add_argument("--baseline_index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite published checkpoint: {args.output}")
    for path in (args.checkpoint, args.validation_predictions, args.validation_metrics, args.baseline_index):
        if not path.is_file():
            raise FileNotFoundError(path)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    variant = checkpoint.get("variant")
    if variant not in {"G0", "G0-Con", "P0", "P0-R", "C1", "C2"}:
        raise ValueError(f"Unsupported trainable Part 2 variant: {variant!r}")
    baseline = json.loads(args.baseline_index.read_text(encoding="utf-8"))
    seed = int(checkpoint.get("seed", -1))
    baseline_run = baseline.get("runs", {}).get(str(seed))
    if baseline.get("variant") != "B0" or baseline_run is None:
        raise ValueError("Checkpoint seed is absent from the finalized B0 index")
    if checkpoint.get("baseline_checkpoint_sha256") != baseline_run["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint B0 hash mismatch")
    if checkpoint.get("feature_manifest_sha256") != baseline["feature_manifest"]["sha256"]:
        raise ValueError("Checkpoint feature manifest hash mismatch")

    metrics = json.loads(args.validation_metrics.read_text(encoding="utf-8"))
    brief = metrics.get("brief", {})
    names = ["AdapterScore"] if variant in {"P0", "P0-R"} else [
        "SetSuccess@0.5", "MR-full-mAP", "Count-Acc-5"
    ]
    key = [float(brief[name]) for name in names]
    stored_key = checkpoint.get("training_state", {}).get("prev_best_key")
    if stored_key is not None and [float(value) for value in stored_key] != key:
        raise ValueError(f"Validation metric key {key} does not match checkpoint key {stored_key}")

    opt = checkpoint.get("opt")
    if opt is None:
        raise ValueError("Checkpoint lacks training opt")
    setattr(opt, "baseline_variant", "B0")
    checkpoint.update({
        "opt": opt,
        "selection_split": "val",
        "selection_metric_names": names,
        "selection_key": key,
        "validation_artifacts": {
            "predictions": {
                "path": str(args.validation_predictions),
                "sha256": sha256_file(args.validation_predictions),
            },
            "metrics": {
                "path": str(args.validation_metrics),
                "sha256": sha256_file(args.validation_metrics),
            },
        },
        "event_interface_schema": "EventInterfaceV1" if variant in {"P0", "P0-R", "C1", "C2"} else None,
        "event_interface_metadata": {
            "schema": "EventInterfaceV1",
            "span_source": "greedy_seed_spans",
            "selection_threshold": 0.5,
            "max_modes": int(getattr(opt, "event_max_modes", 10)),
        } if variant in {"P0", "P0-R"} else checkpoint.get("event_interface_metadata"),
    })
    command_path = args.checkpoint.parent / "command.txt"
    if command_path.is_file():
        checkpoint["training_command"] = command_path.read_text(encoding="utf-8").strip()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, args.output)
    print(json.dumps({
        "output": str(args.output),
        "sha256": sha256_file(args.output),
        "seed": seed,
        "variant": variant,
        "selection_key": key,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
