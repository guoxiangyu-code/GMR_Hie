#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ $# -lt 3 ]]; then
  printf 'Usage: %s {G0-Threshold|G0|G0-Con|P0|C1|C2} seed gpu [run_tag]\n' "$0" >&2
  exit 2
fi

VARIANT="$1"
SEED="$2"
GPU="$3"
RUN_TAG="${4:-strict-v1}"
case "${VARIANT}" in
  G0-Threshold|G0|G0-Con|P0|C1|C2) ;;
  *) printf 'Unsupported required variant: %s\n' "${VARIANT}" >&2; exit 2 ;;
esac

PYTHON_BIN="${HIEA2M_PYTHON:-/home/guoxiangyu/miniconda3/envs/flashvtg/bin/python}"
FEATURE_MANIFEST="${HIEA2M_FEATURE_MANIFEST:-artifacts/features/f-lighthouse/feature_manifest.json}"
DATA_MANIFEST_INDEX="${HIEA2M_DATA_MANIFEST_INDEX:-artifacts/manifests/standard/manifest_index.json}"
BASELINE_INDEX="${HIEA2M_BASELINE_INDEX:-artifacts/baselines/baseline_index.json}"
CARDINALITY_ROOT="${HIEA2M_CARDINALITY_ROOT:-artifacts/cardinality}"
ADAPTER_INDEX="${HIEA2M_ADAPTER_INDEX:-artifacts/adapters/public_adapter_index.json}"
RUN_ROOT="${HIEA2M_RUN_ROOT:-artifacts/part2_runs/${RUN_TAG}}"
RUN_DIR="${RUN_ROOT}/${SEED}/${VARIANT}"
TEST_PATH="$("${PYTHON_BIN}" - "${DATA_MANIFEST_INDEX}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["data_manifests"]["test"]["path"])
PY
)"
B0_CKPT="$("${PYTHON_BIN}" -m training.flash_vtg_gmr.resolve_artifact \
  --index "${BASELINE_INDEX}" --seed "${SEED}")"

mkdir -p "${RUN_DIR}"
CALIBRATION="${RUN_DIR}/calibration.json"
PREDICTIONS="${RUN_DIR}/hl_test_submission.jsonl"
METRICS="${RUN_DIR}/test_metrics.json"
REPLAY="${RUN_DIR}/prediction_replay.json"
RESOLVED_OPT="${RUN_DIR}/resolved_inference_opt.json"

common=(
  --variant "${VARIANT}" --seed "${SEED}"
  --feature_manifest "${FEATURE_MANIFEST}"
  --data_manifest_index "${DATA_MANIFEST_INDEX}"
  --baseline_index "${BASELINE_INDEX}"
  --results_dir "${RUN_DIR}" --device "${GPU}" --num_workers 4
)

if [[ "${VARIANT}" == "G0-Threshold" ]]; then
  bash scripts/run_hiea2m.sh calibrate-threshold "${common[@]}" \
    --init_backbone_ckpt "${B0_CKPT}" --checkpoint "${B0_CKPT}" --output "${CALIBRATION}"
  CHECKPOINT="${B0_CKPT}"
  INFER_EXTRA=(--init_backbone_ckpt "${B0_CKPT}")
elif [[ "${VARIANT}" == "P0" ]]; then
  TRAIN_CHECKPOINT="${RUN_DIR}/model_best.ckpt"
  CHECKPOINT="${RUN_DIR}/model_public.ckpt"
  [[ -f "${TRAIN_CHECKPOINT}" ]] || { printf 'Missing trained checkpoint: %s\n' "${TRAIN_CHECKPOINT}" >&2; exit 2; }
  if [[ ! -f "${CHECKPOINT}" ]]; then
    "${PYTHON_BIN}" -m training.flash_vtg_gmr.publish_part2_checkpoint \
      --checkpoint "${TRAIN_CHECKPOINT}" \
      --validation_predictions "${RUN_DIR}/best_hl_val_preds.jsonl" \
      --validation_metrics "${RUN_DIR}/best_hl_val_preds_metrics.json" \
      --baseline_index "${BASELINE_INDEX}" --output "${CHECKPOINT}"
  fi
  INFER_EXTRA=()
else
  CHECKPOINT="${RUN_DIR}/model_best.ckpt"
  [[ -f "${CHECKPOINT}" ]] || { printf 'Missing trained checkpoint: %s\n' "${CHECKPOINT}" >&2; exit 2; }
  CALIBRATE_EXTRA=()
  if [[ "${VARIANT}" == "G0" || "${VARIANT}" == "G0-Con" ]]; then
    CALIBRATE_EXTRA+=(--raw_threshold_calibration "${RUN_ROOT}/${SEED}/G0-Threshold/calibration.json")
  fi
  bash scripts/run_hiea2m.sh calibrate-count "${common[@]}" \
    --checkpoint "${CHECKPOINT}" --output "${CALIBRATION}" "${CALIBRATE_EXTRA[@]}"
  INFER_EXTRA=()
fi

INFER_CALIBRATION=()
if [[ "${VARIANT}" != "P0" ]]; then
  INFER_CALIBRATION=(--count_calibration "${CALIBRATION}")
fi
bash scripts/run_hiea2m.sh infer "${common[@]}" \
  --checkpoint "${CHECKPOINT}" --split test \
  "${INFER_CALIBRATION[@]}" "${INFER_EXTRA[@]}"

"${PYTHON_BIN}" -m eval.eval_hiea2m_diagnostics \
  --submission "${PREDICTIONS}" --ground_truth "${TEST_PATH}" --output "${METRICS}" --map_num_workers 0

REPLAY_CALIBRATION=()
if [[ "${VARIANT}" != "P0" ]]; then
  REPLAY_CALIBRATION=(--calibration "${CALIBRATION}")
fi
"${PYTHON_BIN}" -m training.flash_vtg_gmr.validate_part2_predictions \
  --predictions "${PREDICTIONS}" --ground_truth "${TEST_PATH}" \
  "${REPLAY_CALIBRATION[@]}" --output "${REPLAY}"

REGISTER_ARGS=(
  --seed "${SEED}" --variant "${VARIANT}" --run_id "${RUN_TAG}-${SEED}-${VARIANT}"
  --baseline_index "${BASELINE_INDEX}" --cardinality_root "${CARDINALITY_ROOT}"
  --adapter_index "${ADAPTER_INDEX}" --predictions "${PREDICTIONS}"
  --replay "${REPLAY}" --metrics "${METRICS}" --opt "${RESOLVED_OPT}"
)
if [[ "${VARIANT}" != "G0-Threshold" ]]; then
  REGISTER_ARGS+=(--checkpoint "${CHECKPOINT}")
  REGISTER_ARGS+=(--validation_predictions "${RUN_DIR}/best_hl_val_preds.jsonl")
  REGISTER_ARGS+=(--validation_metrics "${RUN_DIR}/best_hl_val_preds_metrics.json")
fi
if [[ "${VARIANT}" != "P0" ]]; then
  REGISTER_ARGS+=(--calibration "${CALIBRATION}")
fi
"${PYTHON_BIN}" -m training.flash_vtg_gmr.register_part2_run "${REGISTER_ARGS[@]}"
