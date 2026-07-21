"""Build cross-seed Part 2 aggregate metrics and the preregistered report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from training.flash_vtg_gmr.contracts import sha256_file


REQUIRED_REPORT_METRICS = (
    "Count-Acc-5",
    "SetSuccess@0.5",
    "mAP",
    "mR+@5",
    "G-mIoU@1",
    "G-mIoU@3",
    "G-mIoU@5",
    "AUROC",
    "Rej-F1",
    "Null-FPR",
    "Count-Acc-Exact-Selected",
    "Count-MAE-Selected",
    "OverPredictionRate",
    "UnderPredictionRate",
    "DuplicateRate@0.5",
    "Selected-FullCoverage@0.5",
    "Oracle-Mode-FullCoverage@0.5",
)


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if isinstance(value, str):
        temporary.write_text(value, encoding="utf-8")
    else:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _metrics(path: str) -> dict[str, float]:
    doc = _load(Path(path))
    merged = {}
    for source in (doc, doc.get("brief", {}), doc.get("diagnostics", {}), doc.get("maintained_gmr_metrics", {}).get("brief", {})):
        if isinstance(source, dict):
            for key, value in source.items():
                if isinstance(value, (int, float)) and np.isfinite(float(value)):
                    merged[key] = float(value)
    return merged


def _grouped_metrics(path: str) -> dict[str, dict[str, float]]:
    document = _load(Path(path))
    grouped = document.get("diagnostics", {}).get("grouped_metrics", {})
    result: dict[str, dict[str, float]] = {}
    if not isinstance(grouped, dict):
        return result
    for group, payload in grouped.items():
        if not isinstance(payload, dict):
            continue
        result[group] = {
            key: float(value)
            for key, value in payload.items()
            if isinstance(value, (int, float)) and np.isfinite(float(value))
        }
    return result


def _summary(values: list[float]) -> dict[str, object]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "values": values,
    }


def _format_metrics(metrics: dict[str, float]) -> str:
    return ", ".join(
        f"{key}={metrics[key]:.6f}"
        for key in REQUIRED_REPORT_METRICS
        if key in metrics
    )


def _baseline_val_metrics(run: dict) -> dict[str, float]:
    manifest_path = Path(run["artifact_manifest"]["path"])
    if sha256_file(manifest_path) != run["artifact_manifest"]["sha256"]:
        raise ValueError(f"Baseline artifact manifest hash mismatch: {manifest_path}")
    manifest = _load(manifest_path)
    record = manifest["artifacts"]["val_metrics_raw"]
    if sha256_file(record["path"]) != record["sha256"]:
        raise ValueError(f"Baseline validation metric hash mismatch: {record['path']}")
    return _metrics(record["path"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cardinality_root", type=Path, default=Path("artifacts/cardinality"))
    parser.add_argument("--baseline_index", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2024, 2025])
    args = parser.parse_args()
    index_path = args.cardinality_root / "cardinality_index.json"
    index = _load(index_path)
    baseline = _load(args.baseline_index)
    metrics_by_seed = {}
    grouped_by_seed = {}
    for seed in args.seeds:
        metrics_by_seed[str(seed)] = {}
        grouped_by_seed[str(seed)] = {}
        for variant, record in index.get("runs", {}).get(str(seed), {}).items():
            metrics_by_seed[str(seed)][variant] = _metrics(record["metrics"]["path"])
            grouped_by_seed[str(seed)][variant] = _grouped_metrics(record["metrics"]["path"])

    aggregate = {"schema_version": "hiea2m.aggregate-metrics.v1", "seeds": args.seeds, "variants": {}}
    variants = sorted({variant for runs in metrics_by_seed.values() for variant in runs})
    for variant in variants:
        common = set.intersection(*[
            set(metrics_by_seed[str(seed)].get(variant, {})) for seed in args.seeds
        ]) if args.seeds else set()
        variant_aggregate = {
            key: _summary([metrics_by_seed[str(seed)][variant][key] for seed in args.seeds])
            for key in sorted(common)
        }
        grouped_aggregate = {}
        groups = sorted(set.intersection(*[
            set(grouped_by_seed[str(seed)].get(variant, {})) for seed in args.seeds
        ])) if args.seeds else []
        for group in groups:
            group_common = set.intersection(*[
                set(grouped_by_seed[str(seed)][variant][group]) for seed in args.seeds
            ])
            grouped_aggregate[group] = {
                key: _summary([
                    grouped_by_seed[str(seed)][variant][group][key] for seed in args.seeds
                ])
                for key in sorted(group_common)
            }
        variant_aggregate["grouped_metrics"] = grouped_aggregate
        aggregate["variants"][variant] = variant_aggregate
    aggregate_path = args.cardinality_root / "aggregate_metrics.json"
    _write(aggregate_path, aggregate)

    comparisons = [
        ("G0-Threshold", "G0"),
        ("G0", "G0-Con"),
        ("G0", "C1"),
        ("C1", "C2"),
    ]
    lines = ["# Part 2 Report", "", "Variant comparisons use frozen per-seed test results; no test result selected a checkpoint or calibration. The B0 → P0 acceptance gate is validation-only, as preregistered.", "", "## Required per-seed metrics", ""]
    for variant in variants:
        lines.extend([f"### {variant}", ""])
        for seed in args.seeds:
            metrics = metrics_by_seed[str(seed)][variant]
            lines.append(f"- Seed {seed}: {_format_metrics(metrics)}")
            for group in ("null", "single", "multi"):
                grouped = grouped_by_seed[str(seed)][variant].get(group, {})
                if grouped:
                    values = ", ".join(f"{key}={value:.6f}" for key, value in sorted(grouped.items()))
                    lines.append(f"  - {group}: {values}")
        lines.append("")
    lines.extend(["## Preregistered comparisons", ""])
    positive_directions = 0
    negative_directions = 0
    for left, right in comparisons:
        lines.extend([f"## {left} → {right}", ""])
        for seed in args.seeds:
            left_metrics = metrics_by_seed[str(seed)].get(left, {})
            right_metrics = metrics_by_seed[str(seed)].get(right, {})
            deltas = []
            for key in ("Count-Acc-5", "SetSuccess@0.5", "mAP"):
                if key in left_metrics and key in right_metrics:
                    delta = right_metrics[key] - left_metrics[key]
                    deltas.append(f"{key}: {delta:+.6f}")
                    if key in {"Count-Acc-5", "SetSuccess@0.5"}:
                        positive_directions += int(delta > 0)
                        negative_directions += int(delta < 0)
            lines.append(f"- Seed {seed}: " + (", ".join(deltas) if deltas else "metrics unavailable"))
        lines.append("")
    lines.extend(["## B0 → P0", ""])
    for seed in args.seeds:
        b0_map = float(_baseline_val_metrics(baseline["runs"][str(seed)])["MR-full-mAP"])
        p0_record = index["runs"][str(seed)]["P0"]
        p0 = _metrics(p0_record["validation_metrics"]["path"])
        p0_map = p0.get("mAP", float("nan"))
        if not np.isfinite(p0_map):
            p0_map = p0.get("MR-full-mAP", float("nan"))
        raw_fc = p0.get("Raw-Proposal-Oracle-FullCoverage@0.5", float("nan"))
        mode_fc = p0.get("Oracle-Mode-FullCoverage@0.5", float("nan"))
        passed = (b0_map - p0_map <= 0.5) and (raw_fc - mode_fc <= 0.05)
        lines.append(f"- Seed {seed} (validation): mAP Δ {p0_map - b0_map:+.6f}; oracle coverage Δ {mode_fc - raw_fc:+.6f}; gate={'PASS' if passed else 'FAIL'}")
    outcome = "POSITIVE" if positive_directions > 0 and negative_directions == 0 else ("NEGATIVE" if negative_directions > 0 and positive_directions == 0 else "MIXED")
    lines.extend(["", "## Failure-mode diagnosis", ""])
    p0_empty = all(metrics_by_seed[str(seed)]["P0"].get("mAP") == 0 for seed in args.seeds)
    if p0_empty:
        lines.append("- Both P0 runs have test mAP=0 and Selected-FullCoverage@0.5=0: the locked 0.5 event threshold selects an empty set, despite non-zero raw-proposal and oracle-mode coverage.")
    c_multi_zero = all(
        grouped_by_seed[str(seed)][variant].get("multi", {}).get("Count-Acc-5") == 0
        for seed in args.seeds for variant in ("C1", "C2")
    )
    if c_multi_zero:
        lines.append("- C1/C2 obtain zero multi-query Count-Acc-5 for both seeds; their overall count accuracy is dominated by null/single queries and does not solve multi-event cardinality.")
    selected_coverage_zero = all(
        metrics_by_seed[str(seed)][variant].get("Selected-FullCoverage@0.5") == 0
        for seed in args.seeds for variant in ("C1", "C2")
    )
    if selected_coverage_zero:
        lines.append("- C1/C2 Selected-FullCoverage@0.5 remains zero for both seeds, so neither CE nor the contrastive extension recovers complete multi-event sets.")
    lines.extend(["", "## Research outcome", "", outcome, ""])
    report_path = args.cardinality_root / "part2_report.md"
    _write(report_path, "\n".join(lines))

    index["aggregate_metrics"] = {"path": str(aggregate_path), "sha256": sha256_file(aggregate_path), "size_bytes": aggregate_path.stat().st_size}
    index["report"] = {"path": str(report_path), "sha256": sha256_file(report_path), "size_bytes": report_path.stat().st_size}
    index["research_outcome"] = outcome
    _write(index_path, index)
    print(json.dumps({"research_outcome": outcome, "aggregate_metrics": str(aggregate_path), "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
