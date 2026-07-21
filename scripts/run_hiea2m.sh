#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ $# -lt 1 ]]; then
  printf 'Usage: %s {baseline|baseline-repro-check|train|infer|calibrate-threshold} [options]\n' "$0" >&2
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
FREEZE_ADAPTER="0"
COUNT_CALIBRATION=""
WORKERS="4"
DEVICE_VALUE="0"
RESULTS_DIR_PATH=""
BATCH_SIZE_VALUE="8"
EVAL_BATCH_SIZE_VALUE="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant) VARIANT="$2"; shift 2 ;;
    --seed) SEED_VALUE="$2"; shift 2 ;;
    --feature_manifest) FEATURE_MANIFEST_PATH="$2"; shift 2 ;;
    --data_manifest_index) DATA_MANIFEST_INDEX_PATH="$2"; shift 2 ;;
    --baseline_index) BASELINE_INDEX_PATH="$2"; shift 2 ;;
    --init_backbone_ckpt) INIT_BACKBONE_CKPT="$2"; shift 2 ;;
    --adapter_ckpt) ADAPTER_CKPT="$2"; shift 2 ;;
    --freeze_adapter) FREEZE_ADAPTER="1"; shift 1 ;;
    --count_calibration) COUNT_CALIBRATION="$2"; shift 2 ;;
    --num_workers) WORKERS="$2"; shift 2 ;;
    --device) DEVICE_VALUE="$2"; shift 2 ;;
    --results_dir) RESULTS_DIR_PATH="$2"; shift 2 ;;
    --bsz) BATCH_SIZE_VALUE="$2"; shift 2 ;;
    --eval_bsz) EVAL_BATCH_SIZE_VALUE="$2"; shift 2 ;;
    --repro_check) shift ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done

[[ -n "${FEATURE_MANIFEST_PATH}" ]] || { printf '%s\n' '--feature_manifest is required' >&2; exit 2; }
[[ -n "${DATA_MANIFEST_INDEX_PATH}" ]] || { printf '%s\n' '--data_manifest_index is required' >&2; exit 2; }

read_manifest_value() {
  python - "$1" "$2" <<'PY'
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
export RESULTS_DIR="${RESULTS_DIR_PATH:-${RESULTS_DIR:-artifacts/baselines/${SEED_VALUE}}}"
export BATCH_SIZE="${BATCH_SIZE_VALUE}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE_VALUE}"

TRAIN_PATH="$(read_manifest_value "${DATA_MANIFEST_INDEX_PATH}" data_manifests.train.path)"
VAL_PATH="$(read_manifest_value "${DATA_MANIFEST_INDEX_PATH}" data_manifests.val.path)"
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
  --lr_drop 400
  --wd 1e-4
  --n_epoch 400
  --max_es_cnt 80
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
  --use_exist_head
  --exist_pool mean
  --exist_loss_coef 1.0
  --exist_gate_thd 0.4
  --pred_score_thd_for_cls 0.4
  --nms_thd 0.7
  --no_drop_last
  --strict_data_contract
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

case "${COMMAND}" in
  train)
    exec python3 -m training.flash_vtg_gmr.train "${ARGS[@]}"
    ;;
  infer)
    exec python3 -m training.flash_vtg_gmr.inference "${ARGS[@]}" --resume "${RESULTS_DIR}/soccer_gmr_best.ckpt"
    ;;
  calibrate-threshold)
    CKPT_PATH="${INIT_BACKBONE_CKPT}"
    if [[ -z "${CKPT_PATH}" ]]; then
      CKPT_PATH="${RESULTS_DIR}/soccer_gmr_best.ckpt"
    fi
    exec python3 -m training.flash_vtg_gmr.calibrate_count \
      --checkpoint "${CKPT_PATH}" \
      --data_manifest_index "${DATA_MANIFEST_INDEX_PATH}" \
      --split val \
      --output "${RESULTS_DIR}/calibration.json" \
      --device "${DEVICE}"
    ;;
  *)
    printf 'Unknown command: %s\n' "${COMMAND}" >&2
    exit 2
    ;;
esac
