#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

: "${MODEL_PATH:?Set MODEL_PATH to the frozen B0 checkpoint}"
: "${TEST_PATH:?Set TEST_PATH to the canonical test JSONL from manifest_index.json}"
: "${SLOWFAST_FEAT_DIR:?Set SLOWFAST_FEAT_DIR to the SlowFast feature directory}"
: "${CLIP_FEAT_DIR:?Set CLIP_FEAT_DIR to the CLIP video feature directory}"
: "${TEXT_FEAT_DIR:?Set TEXT_FEAT_DIR to the CLIP text feature directory}"

MODEL_DIR="$(cd "$(dirname "${MODEL_PATH}")" && pwd)"
RESULTS_DIR="${RESULTS_DIR:-${MODEL_DIR}}"
OPT_PATH="${OPT_PATH:-${MODEL_DIR}/opt.json}"
DEVICE="${DEVICE:-0}"
mkdir -p "${RESULTS_DIR}"

[[ -f "${OPT_PATH}" ]] || { printf 'Missing saved training options: %s\n' "${OPT_PATH}" >&2; exit 2; }

python -m training.flash_vtg_gmr.inference \
  configs/flash_vtg_gmr/model.py \
  --resume "${MODEL_PATH}" \
  --opt_path "${OPT_PATH}" \
  --eval_split_name test \
  --eval_path "${TEST_PATH}" \
  --eval_results_dir "${RESULTS_DIR}" \
  --v_feat_dirs "${SLOWFAST_FEAT_DIR}" "${CLIP_FEAT_DIR}" \
  --t_feat_dir "${TEXT_FEAT_DIR}" \
  --v_feat_dim 2816 \
  --t_feat_dim 512 \
  --device "${DEVICE}" \
  --nms_thd 0.7

python -m eval.eval_main \
  --submission_path "${RESULTS_DIR}/hl_test_submission.jsonl" \
  --gt_path "${TEST_PATH}" \
  --save_path "${RESULTS_DIR}/flash_vtg_gmr_test_results_raw.json" \
  --cls_thresholds 0.4 0.6 \
  --gmiou_cls_threshold 0.4

python -m eval.eval_main \
  --submission_path "${RESULTS_DIR}/hl_test_submission_nms_thd_0.7.jsonl" \
  --gt_path "${TEST_PATH}" \
  --save_path "${RESULTS_DIR}/flash_vtg_gmr_test_results_nms.json" \
  --cls_thresholds 0.4 0.6 \
  --gmiou_cls_threshold 0.4
