"""Replay and validate the locked Part 2 set-selection contract from JSONL."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from training.flash_vtg_gmr.contracts import sha256_file


COUNT_VARIANTS = {"G0", "G0-Con", "C1", "C2"}


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _postprocess(windows: list[list[float]], decode_duration: float) -> list[list[float]]:
    result = []
    for start, end, score in windows:
        # Mirrors the HL PostProcessorDETR: clamp to 150, round to the 2-second
        # feature grid, then clamp to the per-video decode duration.
        start = round(max(0.0, min(float(start), 150.0)) / 2.0) * 2.0
        end = round(max(0.0, min(float(end), 150.0)) / 2.0) * 2.0
        start = max(0.0, min(start, decode_duration))
        end = max(0.0, min(end, decode_duration))
        if end > start:
            result.append([start, end, round(float(score), 3)])
    return result


def replay_prediction(row: dict, calibration: dict, decode_duration: float) -> list[list[float]]:
    variant = row["variant"]
    tau_raw = float(calibration.get("tau_raw", 0.5))
    tau_mode = float(calibration.get("tau_mode", 0.5))
    pred_count = int(row.get("pred_count", -1))

    if variant == "G0-Threshold":
        chosen = [
            [candidate[0], candidate[1], _sigmoid(float(candidate[2]))]
            for candidate in row["raw_proposal_windows"]
            if _sigmoid(float(candidate[2])) >= tau_raw
        ]
        chosen.sort(key=lambda item: item[2], reverse=True)
    elif variant in {"G0", "G0-Con"}:
        if pred_count == 0:
            chosen = []
        else:
            ranked = sorted(
                ([candidate[0], candidate[1], _sigmoid(float(candidate[2]))]
                 for candidate in row["raw_proposal_windows"]),
                key=lambda item: item[2], reverse=True,
            )
            if pred_count <= 3:
                chosen = ranked[:pred_count]
            else:
                above = [item for item in ranked if item[2] >= tau_raw]
                chosen = ranked[:4] if len(above) < 4 else above[:10]
    elif variant in {"P0", "P0-R", "C1", "C2"}:
        ranked = sorted(
            ([mode[0], mode[1], _sigmoid(float(mode[2])) * _sigmoid(float(mode[3])),
              _sigmoid(float(mode[2]))]
             for mode in row["oracle_mode_windows"]),
            key=lambda item: item[2], reverse=True,
        )
        if variant in {"P0", "P0-R"}:
            chosen = [item[:3] for item in ranked if item[3] >= 0.5]
        elif pred_count == 0:
            chosen = []
        elif pred_count <= 3:
            chosen = [item[:3] for item in ranked[:pred_count]]
        else:
            above = [item for item in ranked if item[3] >= tau_mode]
            chosen = [item[:3] for item in (ranked[:4] if len(above) < 4 else above[:10])]
    else:
        raise ValueError(f"Unsupported variant {variant!r}")
    return _postprocess(chosen, decode_duration)


def validate_predictions(predictions: list[dict], ground_truth: list[dict], calibration: dict) -> dict:
    pred_by_qid = {str(row["qid"]): row for row in predictions}
    gt_by_qid = {str(row["qid"]): row for row in ground_truth}
    if len(pred_by_qid) != len(predictions):
        raise ValueError("Duplicate qid in predictions")
    if len(gt_by_qid) != len(ground_truth):
        raise ValueError("Duplicate qid in ground truth")
    if set(pred_by_qid) != set(gt_by_qid):
        raise ValueError("Prediction qids do not exactly cover the requested split")

    variants = {row.get("variant") for row in predictions}
    if len(variants) != 1:
        raise ValueError(f"Predictions contain multiple variants: {sorted(variants)}")
    variant = next(iter(variants))
    if calibration and calibration.get("variant") != variant:
        raise ValueError("Calibration/prediction variant mismatch")
    if variant not in {"P0", "P0-R"} and not calibration:
        raise ValueError(f"{variant} replay requires validation calibration")

    mismatches = []
    for qid, gt in gt_by_qid.items():
        row = pred_by_qid[qid]
        probabilities = row.get("pred_count_probs")
        if variant in COUNT_VARIANTS:
            if not isinstance(probabilities, list) or len(probabilities) != 5:
                raise ValueError(f"qid {qid}: missing five-class probabilities")
            if abs(sum(map(float, probabilities)) - 1.0) > 1e-5:
                raise ValueError(f"qid {qid}: count probabilities do not sum to one")
            argmax = max(range(5), key=lambda index: float(probabilities[index]))
            if int(row.get("pred_count", -1)) != argmax:
                raise ValueError(f"qid {qid}: pred_count is not probability argmax")
            if abs(float(row.get("pred_exist_score", -1)) - (1.0 - float(probabilities[0]))) > 1e-6:
                raise ValueError(f"qid {qid}: pred_exist_score is not 1-P(count=0)")

        expected = replay_prediction(row, calibration, float(gt["D_decode"]))
        actual = row.get("pred_relevant_windows", [])
        same = len(expected) == len(actual)
        if same:
            for expected_window, actual_window in zip(expected, actual):
                if len(actual_window) < 3 or any(
                    abs(float(left) - float(right)) > 1.1e-3
                    for left, right in zip(expected_window, actual_window[:3])
                ):
                    same = False
                    break
        if not same:
            mismatches.append({"qid": row["qid"], "expected": expected, "actual": actual})
            if len(mismatches) >= 10:
                break

    if mismatches:
        raise ValueError(f"Selection replay mismatch: {mismatches}")
    return {
        "status": "PASS",
        "variant": variant,
        "query_count": len(predictions),
        "selection_replay_mismatches": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ground_truth", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    calibration = json.load(args.calibration.open(encoding="utf-8")) if args.calibration else {}
    result = validate_predictions(
        _load_jsonl(args.predictions), _load_jsonl(args.ground_truth), calibration
    )
    result.update({
        "predictions_sha256": sha256_file(args.predictions),
        "ground_truth_sha256": sha256_file(args.ground_truth),
        "calibration_sha256": sha256_file(args.calibration) if args.calibration else None,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
