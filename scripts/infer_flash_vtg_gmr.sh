#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

: "${MODEL_PATH:?Set MODEL_PATH to the downloaded flashvtg_gmr checkpoint}"
: "${TEST_PATH:?Set TEST_PATH to data/label/Standard/test.jsonl}"
: "${SLOWFAST_FEAT_DIR:?Set SLOWFAST_FEAT_DIR to the SlowFast feature directory}"
: "${CLIP_FEAT_DIR:?Set CLIP_FEAT_DIR to the CLIP video feature directory}"
: "${TEXT_FEAT_DIR:?Set TEXT_FEAT_DIR to the CLIP text feature directory}"

RESULTS_DIR="${RESULTS_DIR:-results/flash_vtg_gmr}"
DEVICE="${DEVICE:-0}"
mkdir -p "${RESULTS_DIR}"

python -m training.flash_vtg_gmr.inference \
  configs/flash_vtg_gmr/model.py \
  --resume "${MODEL_PATH}" \
  --opt_path configs/flash_vtg_gmr/soccer_gmr.json \
  --eval_split_name test \
  --eval_path "${TEST_PATH}" \
  --eval_results_dir "${RESULTS_DIR}" \
  --v_feat_dirs "${SLOWFAST_FEAT_DIR}" "${CLIP_FEAT_DIR}" \
  --t_feat_dir "${TEXT_FEAT_DIR}" \
  --v_feat_dim 2816 \
  --t_feat_dim 512 \
  --device "${DEVICE}" \
  --nms_thd 0.7

python eval/eval_main.py \
  --submission_path "${RESULTS_DIR}/hl_test_submission_nms_thd_0.7.jsonl" \
  --gt_path "${TEST_PATH}" \
  --save_path "${RESULTS_DIR}/flash_vtg_gmr_test_results.json" \
  --cls_thresholds 0.4 0.6 \
  --gmiou_cls_threshold 0.4
