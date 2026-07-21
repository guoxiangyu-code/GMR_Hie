import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from training.flash_vtg_gmr.contracts import sha256_file


class Part2CompletionContractTest(unittest.TestCase):
    def _write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def _record(self, path):
        return {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}

    def test_complete_requires_hash_verified_disk_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            feature = root / "feature.json"
            data = root / "data.json"
            checkpoint = root / "checkpoint.bin"
            prediction = root / "prediction.jsonl"
            ground_truth = root / "test.jsonl"
            report = root / "report.md"
            aggregate = root / "aggregate.json"
            for path in (feature, data, checkpoint, prediction, ground_truth, report, aggregate):
                path.write_text("{}\n", encoding="utf-8")
            baseline = root / "baseline.json"
            baseline_value = {
                "feature_manifest": self._record(feature),
                "data_manifest_index": self._record(data),
                "runs": {},
            }
            for seed in (2024, 2025):
                baseline_val_predictions = root / f"baseline-val-predictions-{seed}.jsonl"
                baseline_val_metrics = root / f"baseline-val-metrics-{seed}.json"
                baseline_val_predictions.write_text("{}\n", encoding="utf-8")
                self._write_json(baseline_val_metrics, {"brief": {"MR-full-mAP": 20.0}})
                baseline_artifact_manifest = root / f"baseline-artifact-{seed}.json"
                self._write_json(baseline_artifact_manifest, {
                    "artifacts": {
                        "val_predictions_raw": self._record(baseline_val_predictions),
                        "val_metrics_raw": self._record(baseline_val_metrics),
                    }
                })
                baseline_value["runs"][str(seed)] = {
                    "checkpoint": self._record(checkpoint),
                    "artifact_manifest": self._record(baseline_artifact_manifest),
                    # Deliberately differs from validation.  P0 acceptance must
                    # use the transitive, hash-indexed validation artifact.
                    "test_metrics_brief": {"mAP": 100.0},
                }
            self._write_json(baseline, baseline_value)

            cardinality_root = root / "cardinality"
            runs = {}
            adapter_runs = {}
            for seed in (2024, 2025):
                runs[str(seed)] = {}
                for variant in ("G0-Threshold", "G0", "G0-Con", "P0", "C1", "C2"):
                    option = root / f"opt-{seed}-{variant}.json"
                    self._write_json(option, {
                        "seed": seed,
                        "variant": variant,
                        "baseline_variant": "B0",
                        "eval_split_name": "test",
                        "baseline_checkpoint_sha256": self._record(checkpoint)["sha256"],
                        "feature_manifest_sha256": self._record(feature)["sha256"],
                    })
                    metrics = root / f"metrics-{seed}-{variant}.json"
                    self._write_json(metrics, {
                        "submission_sha256": self._record(prediction)["sha256"],
                        "ground_truth_sha256": self._record(ground_truth)["sha256"],
                        "brief": {"mAP": 20.0},
                        "diagnostics": {
                            "Raw-Proposal-Oracle-FullCoverage@0.5": 0.9,
                            "Oracle-Mode-FullCoverage@0.5": 0.88,
                        },
                    })
                    calibration_record = None
                    if variant != "P0":
                        calibration = root / f"calibration-{seed}-{variant}.json"
                        self._write_json(calibration, {
                            "variant": variant,
                            "seed": seed,
                            "split": "val",
                            "raw_threshold_calibration_sha256": "frozen" if variant in {"G0", "G0-Con"} else None,
                        })
                        calibration_record = self._record(calibration)
                    record = {
                        "run_id": f"{seed}-{variant}",
                        "seed": seed,
                        "variant": variant,
                        "baseline_checkpoint_sha256": self._record(checkpoint)["sha256"],
                        "train_status": "not_applicable" if variant == "G0-Threshold" else "complete",
                        "checkpoint": None if variant == "G0-Threshold" else self._record(checkpoint),
                        "calibration": calibration_record,
                        "selection_threshold": 0.5 if variant == "P0" else None,
                        "validation_predictions": None,
                        "validation_metrics": None,
                        "predictions": self._record(prediction),
                        "prediction_replay": None,
                        "metrics": self._record(metrics),
                        "opt": self._record(option),
                    }
                    replay = root / f"replay-{seed}-{variant}.json"
                    self._write_json(replay, {
                        "status": "PASS",
                        "variant": variant,
                        "predictions_sha256": self._record(prediction)["sha256"],
                        "ground_truth_sha256": self._record(ground_truth)["sha256"],
                        "query_count": 1,
                        "calibration_sha256": calibration_record["sha256"] if calibration_record else None,
                    })
                    if variant != "G0-Threshold":
                        val_prediction = root / f"val-prediction-{seed}-{variant}.jsonl"
                        val_prediction.write_text("{}\n", encoding="utf-8")
                        val_metrics = root / f"val-metrics-{seed}-{variant}.json"
                        self._write_json(val_metrics, {
                            "brief": {"MR-full-mAP": 20.0},
                            "diagnostics": {
                                "Raw-Proposal-Oracle-FullCoverage@0.5": 0.9,
                                "Oracle-Mode-FullCoverage@0.5": 0.88,
                            },
                        })
                        record["validation_predictions"] = self._record(val_prediction)
                        record["validation_metrics"] = self._record(val_metrics)
                    record["prediction_replay"] = self._record(replay)
                    runs[str(seed)][variant] = record
                    if variant == "P0":
                        adapter_runs[str(seed)] = {
                            "checkpoint": self._record(checkpoint),
                            "predictions": self._record(prediction),
                            "metrics": self._record(metrics),
                            "prediction_replay": self._record(replay),
                            "baseline_checkpoint_sha256": self._record(checkpoint)["sha256"],
                            "feature_manifest_sha256": self._record(feature)["sha256"],
                            "event_interface_schema": "EventInterfaceV1",
                            "selection_threshold": 0.5,
                            "validation_predictions": record["validation_predictions"],
                            "validation_metrics": record["validation_metrics"],
                        }
            cardinality_index = cardinality_root / "cardinality_index.json"
            self._write_json(cardinality_index, {
                "runs": runs,
                "aggregate_metrics": self._record(aggregate),
                "report": self._record(report),
                "research_outcome": "MIXED",
            })
            adapter_index = root / "adapter.json"
            self._write_json(adapter_index, {"runs": adapter_runs})
            test_result = root / "test_result.json"
            self._write_json(test_result, {"status": "PASS", "command": "python -m unittest discover -s tests"})
            output = root / "completion.json"
            command = [
                sys.executable, "-m", "training.flash_vtg_gmr.finalize_part2",
                "--feature_manifest", str(feature),
                "--data_manifest_index", str(data),
                "--baseline_index", str(baseline),
                "--adapter_index", str(adapter_index),
                "--cardinality_root", str(cardinality_root),
                "--test_result", str(test_result),
                "--output", str(output),
            ]
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            completion = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(completion["status"], "COMPLETE")
            self.assertEqual(completion["part3_handoff"], "READY")


if __name__ == "__main__":
    unittest.main()
