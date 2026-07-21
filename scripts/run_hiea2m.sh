#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# The host's default environment currently ships a CUDA 12.8 PyTorch build,
# which is newer than the installed 555 driver.  Prefer the project-compatible
# CUDA 12.1 environment while still allowing an explicit override.
PYTHON_BIN="${HIEA2M_PYTHON:-/home/guoxiangyu/miniconda3/envs/flashvtg/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi
# Required by torch.use_deterministic_algorithms for CUDA GEMM/attention.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

if [[ $# -lt 1 ]]; then
  printf 'Usage: %s {baseline|baseline-repro-check|train|infer|calibrate-threshold|calibrate-count} [options]\n' "$0" >&2
  exit 2
fi

COMMAND="$1"
shift
VARIANT="B0"
SEED_VALUE="2024"
FEATURE_MANIFEST_PATH=""
DATA_MANIFEST_INDEX_PATH=""
BASELINE_INDEX_PATH=""
INIT_BACKBONE_CKPT=""
ADAPTER_CKPT=""
RESUME_CKPT=""
FREEZE_ADAPTER="0"
COUNT_CALIBRATION=""
WORKERS="4"
DEVICE_VALUE="0"
RESULTS_DIR_PATH=""
BATCH_SIZE_VALUE="256"
EVAL_BATCH_SIZE_VALUE="1"
EVAL_EPOCH_VALUE=""
N_EPOCH_VALUE=""
CHECKPOINT_PATH=""
SPLIT_NAME="val"
OUTPUT_PATH=""
RAW_THRESHOLD_CALIBRATION=""
MAX_TRAIN_STEPS_VALUE=""
DEBUG_FLAG="0"
REPRO_FLAG="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant) VARIANT="$2"; shift 2 ;;
    --seed) SEED_VALUE="$2"; shift 2 ;;
    --feature_manifest) FEATURE_MANIFEST_PATH="$2"; shift 2 ;;
    --data_manifest_index) DATA_MANIFEST_INDEX_PATH="$2"; shift 2 ;;
    --baseline_index) BASELINE_INDEX_PATH="$2"; shift 2 ;;
    --init_backbone_ckpt) INIT_BACKBONE_CKPT="$2"; shift 2 ;;
    --adapter_ckpt) ADAPTER_CKPT="$2"; shift 2 ;;
    --resume) RESUME_CKPT="$2"; shift 2 ;;
    --freeze_adapter) FREEZE_ADAPTER="1"; shift 1 ;;
    --count_calibration) COUNT_CALIBRATION="$2"; shift 2 ;;
    --num_workers) WORKERS="$2"; shift 2 ;;
    --device) DEVICE_VALUE="$2"; shift 2 ;;
    --results_dir) RESULTS_DIR_PATH="$2"; shift 2 ;;
    --bsz) BATCH_SIZE_VALUE="$2"; shift 2 ;;
    --eval_bsz) EVAL_BATCH_SIZE_VALUE="$2"; shift 2 ;;
    --eval_epoch) EVAL_EPOCH_VALUE="$2"; shift 2 ;;
    --n_epoch) N_EPOCH_VALUE="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT_PATH="$2"; shift 2 ;;
    --split) SPLIT_NAME="$2"; shift 2 ;;
    --output) OUTPUT_PATH="$2"; shift 2 ;;
    --raw_threshold_calibration) RAW_THRESHOLD_CALIBRATION="$2"; shift 2 ;;
    --max_train_steps) MAX_TRAIN_STEPS_VALUE="$2"; shift 2 ;;
    --debug) DEBUG_FLAG="1"; shift ;;
    --repro_check) REPRO_FLAG="1"; shift ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done

[[ -n "${FEATURE_MANIFEST_PATH}" ]] || { printf '%s\n' '--feature_manifest is required' >&2; exit 2; }
[[ -n "${DATA_MANIFEST_INDEX_PATH}" ]] || { printf '%s\n' '--data_manifest_index is required' >&2; exit 2; }
if [[ "${COMMAND}" != "baseline" && "${COMMAND}" != "baseline-repro-check" ]]; then
  [[ -n "${BASELINE_INDEX_PATH}" ]] || { printf '%s\n' '--baseline_index is required for Part 2' >&2; exit 2; }
fi

read_manifest_value() {
  "${PYTHON_BIN}" - "$1" "$2" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
for part in sys.argv[2].split("."):
    value = value[part]
print(value)
PY
}

export TRAIN_PATH
export VAL_PATH
export SLOWFAST_FEAT_DIR
export CLIP_FEAT_DIR
export TEXT_FEAT_DIR
export FEATURE_MANIFEST="${FEATURE_MANIFEST_PATH}"
export DATA_MANIFEST_INDEX="${DATA_MANIFEST_INDEX_PATH}"
export BASELINE_VARIANT="${VARIANT}"
export SEED="${SEED_VALUE}"
export NUM_WORKERS="${WORKERS}"
export DEVICE="${DEVICE_VALUE}"
if [[ -n "${RESULTS_DIR_PATH}" ]]; then
  RESULTS_DIR_DEFAULT="${RESULTS_DIR_PATH}"
elif [[ "${COMMAND}" == "baseline" || "${COMMAND}" == "baseline-repro-check" ]]; then
  RESULTS_DIR_DEFAULT="artifacts/baselines/${SEED_VALUE}"
elif [[ "${VARIANT}" == "P0" || "${VARIANT}" == "P0-R" || "${VARIANT}" == "P0-AllK" ]]; then
  RESULTS_DIR_DEFAULT="artifacts/adapters/${SEED_VALUE}/${VARIANT}"
else
  RESULTS_DIR_DEFAULT="artifacts/cardinality/${SEED_VALUE}/${VARIANT}"
fi
export RESULTS_DIR="${RESULTS_DIR_DEFAULT}"
export BATCH_SIZE="${BATCH_SIZE_VALUE}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE_VALUE}"

TRAIN_PATH="$(read_manifest_value "${DATA_MANIFEST_INDEX_PATH}" data_manifests.train.path)"
VAL_PATH="$(read_manifest_value "${DATA_MANIFEST_INDEX_PATH}" data_manifests.val.path)"
SPLIT_PATH="$(read_manifest_value "${DATA_MANIFEST_INDEX_PATH}" "data_manifests.${SPLIT_NAME}.path")"
SLOWFAST_FEAT_DIR="$(read_manifest_value "${FEATURE_MANIFEST_PATH}" slowfast_dir)"
CLIP_FEAT_DIR="$(read_manifest_value "${FEATURE_MANIFEST_PATH}" clip_dir)"
TEXT_FEAT_DIR="$(read_manifest_value "${FEATURE_MANIFEST_PATH}" text_dir)"

if [[ "${COMMAND}" == "baseline" || "${COMMAND}" == "baseline-repro-check" ]]; then
  if [[ "${COMMAND}" == "baseline-repro-check" ]]; then
    export REPRO_CHECK=1
    export NUM_WORKERS=0
  else
    export REPRO_CHECK=0
  fi
  exec bash scripts/train_flash_vtg_gmr.sh
fi

# Set up parameters list for training/inference
ARGS=(
  configs/flash_vtg_gmr/model.py
  --dset_name hl
  --ctx_mode video_tef
  --train_path "${TRAIN_PATH}"
  --eval_path "${VAL_PATH}"
  --eval_split_name val
  --v_feat_dirs "${SLOWFAST_FEAT_DIR}" "${CLIP_FEAT_DIR}"
  --t_feat_dir "${TEXT_FEAT_DIR}"
  --v_feat_dim 2816
  --t_feat_dim 512
  --max_q_l 40
  --max_v_l 75
  --clip_length 2
  --max_windows -1
  --lr 3e-5
  --lr_drop 10
  --wd 1e-4
  --n_epoch 15
  --max_es_cnt 5
  --bsz "${BATCH_SIZE}"
  --eval_bsz "${EVAL_BATCH_SIZE}"
  --eval_epoch 1
  --num_workers "${NUM_WORKERS}"
  --device "${DEVICE}"
  --results_root "${RESULTS_DIR}"
  --exp_id soccer_gmr
  --seed "${SEED}"
  --hidden_dim 256
  --dim_feedforward 1024
  --enc_layers 3
  --t2v_layers 6
  --dummy_layers 2
  --nheads 8
  --num_dummies 40
  --total_prompts 10
  --num_prompts 1
  --kernel_size 5
  --num_conv_layers 1
  --num_mlp_layers 5
  --use_SRM
  --input_dropout 0.5
  --dropout 0.1
  --span_loss_type l1
  --lw_reg 1.0
  --lw_cls 5.0
  --lw_sal 0.0
  --lw_saliency 0.0
  --lw_wattn 1.0
  --lw_ms_align 1.0
  --mr_only
  --eval_full_only
  --pred_score_thd_for_cls 0.4
  --nms_thd 0.7
  --no_drop_last
  --strict_data_contract
  --baseline_variant B0
  --feature_manifest "${FEATURE_MANIFEST}"
  --data_manifest_index "${DATA_MANIFEST_INDEX}"
)

# Append Part 2 optional arguments
if [[ -n "${VARIANT}" ]]; then
  ARGS+=(--variant "${VARIANT}")
fi
if [[ -n "${BASELINE_INDEX_PATH}" ]]; then
  ARGS+=(--baseline_index "${BASELINE_INDEX_PATH}")
fi
if [[ -n "${INIT_BACKBONE_CKPT}" ]]; then
  ARGS+=(--init_backbone_ckpt "${INIT_BACKBONE_CKPT}")
fi
if [[ -n "${ADAPTER_CKPT}" ]]; then
  ARGS+=(--adapter_ckpt "${ADAPTER_CKPT}")
fi
if [[ "${FREEZE_ADAPTER}" == "1" ]]; then
  ARGS+=(--freeze_adapter)
fi
if [[ -n "${COUNT_CALIBRATION}" ]]; then
  ARGS+=(--count_calibration "${COUNT_CALIBRATION}")
fi
if [[ -n "${EVAL_EPOCH_VALUE}" ]]; then
  ARGS+=(--eval_epoch "${EVAL_EPOCH_VALUE}")
fi
if [[ -n "${N_EPOCH_VALUE}" ]]; then
  ARGS+=(--n_epoch "${N_EPOCH_VALUE}")
fi
if [[ -n "${MAX_TRAIN_STEPS_VALUE}" ]]; then
  ARGS+=(--max_train_steps "${MAX_TRAIN_STEPS_VALUE}")
fi
if [[ "${DEBUG_FLAG}" == "1" ]]; then
  ARGS+=(--debug)
fi
if [[ "${REPRO_FLAG}" == "1" ]]; then
  ARGS+=(--repro_check)
fi

case "${COMMAND}" in
  train)
    if [[ "${VARIANT}" == "G0-Threshold" ]]; then
      printf '%s\n' 'G0-Threshold is calibration-only and cannot be trained' >&2
      exit 2
    fi
    if [[ -n "${RESUME_CKPT}" ]]; then
      ARGS+=(--resume "${RESUME_CKPT}" --resume_all)
    fi
    exec "${PYTHON_BIN}" -m training.flash_vtg_gmr.train "${ARGS[@]}"
    ;;
  infer)
    CKPT_PATH="${CHECKPOINT_PATH:-${RESUME_CKPT:-${RESULTS_DIR}/model_best.ckpt}}"
    if [[ "${VARIANT}" != "P0" && "${VARIANT}" != "P0-R" && "${VARIANT}" != "P0-AllK" && -z "${COUNT_CALIBRATION}" ]]; then
      printf '%s\n' '--count_calibration is required for Part 2 count/threshold inference' >&2
      exit 2
    fi
    exec "${PYTHON_BIN}" -m training.flash_vtg_gmr.inference "${ARGS[@]}" \
      --resume "${CKPT_PATH}" --eval_split_name "${SPLIT_NAME}" --eval_path "${SPLIT_PATH}" \
      --eval_results_dir "${RESULTS_DIR}"
    ;;
  calibrate-threshold)
    CKPT_PATH="${CHECKPOINT_PATH:-${INIT_BACKBONE_CKPT}}"
    if [[ -z "${CKPT_PATH}" ]]; then
      CKPT_PATH="${RESULTS_DIR}/model_best.ckpt"
    fi
    exec "${PYTHON_BIN}" -m training.flash_vtg_gmr.calibrate_count \
      --checkpoint "${CKPT_PATH}" \
      --data_manifest_index "${DATA_MANIFEST_INDEX_PATH}" \
      --feature_manifest "${FEATURE_MANIFEST_PATH}" \
      --baseline_index "${BASELINE_INDEX_PATH}" \
      --split val \
      --output "${OUTPUT_PATH:-${RESULTS_DIR}/calibration.json}" \
      --device "${DEVICE}" \
      --variant "${VARIANT}"
    ;;
  calibrate-count)
    CKPT_PATH="${CHECKPOINT_PATH:-${RESUME_CKPT:-${RESULTS_DIR}/model_best.ckpt}}"
    CALIBRATION_ARGS=(
      --checkpoint "${CKPT_PATH}"
      --data_manifest_index "${DATA_MANIFEST_INDEX_PATH}"
      --feature_manifest "${FEATURE_MANIFEST_PATH}"
      --baseline_index "${BASELINE_INDEX_PATH}"
      --split val
      --output "${OUTPUT_PATH:-${RESULTS_DIR}/calibration.json}"
      --device "${DEVICE}"
      --variant "${VARIANT}"
    )
    if [[ -n "${RAW_THRESHOLD_CALIBRATION}" ]]; then
      CALIBRATION_ARGS+=(--raw_threshold_calibration "${RAW_THRESHOLD_CALIBRATION}")
    fi
    exec "${PYTHON_BIN}" -m training.flash_vtg_gmr.calibrate_count \
      "${CALIBRATION_ARGS[@]}"
    ;;
  *)
    printf 'Unknown command: %s\n' "${COMMAND}" >&2
    exit 2
    ;;
esac
