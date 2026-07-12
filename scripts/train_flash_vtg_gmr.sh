#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

: "${TRAIN_PATH:?Set TRAIN_PATH to data/label/Standard/train.jsonl}"
: "${VAL_PATH:?Set VAL_PATH to data/label/Standard/val.jsonl}"
: "${SLOWFAST_FEAT_DIR:?Set SLOWFAST_FEAT_DIR to the SlowFast feature directory}"
: "${CLIP_FEAT_DIR:?Set CLIP_FEAT_DIR to the CLIP video feature directory}"
: "${TEXT_FEAT_DIR:?Set TEXT_FEAT_DIR to the CLIP text feature directory}"

RESULTS_DIR="${RESULTS_DIR:-results/flash_vtg_gmr}"
DEVICE="${DEVICE:-0}"

python -m training.flash_vtg_gmr.train \
  configs/flash_vtg_gmr/model.py \
  --dset_name hl \
  --ctx_mode video_tef \
  --train_path "${TRAIN_PATH}" \
  --eval_path "${VAL_PATH}" \
  --eval_split_name val \
  --v_feat_dirs "${SLOWFAST_FEAT_DIR}" "${CLIP_FEAT_DIR}" \
  --t_feat_dir "${TEXT_FEAT_DIR}" \
  --v_feat_dim 2816 \
  --t_feat_dim 512 \
  --max_q_l 40 \
  --max_v_l 75 \
  --clip_length 2 \
  --max_windows 5 \
  --lr 3e-5 \
  --lr_drop 400 \
  --wd 1e-4 \
  --n_epoch 400 \
  --max_es_cnt 80 \
  --bsz 8 \
  --eval_bsz 1 \
  --eval_epoch 1 \
  --num_workers 0 \
  --device "${DEVICE}" \
  --results_root "${RESULTS_DIR}" \
  --exp_id soccer_gmr \
  --seed 2024 \
  --hidden_dim 256 \
  --dim_feedforward 1024 \
  --enc_layers 3 \
  --t2v_layers 6 \
  --dummy_layers 2 \
  --nheads 8 \
  --num_dummies 40 \
  --total_prompts 10 \
  --num_prompts 1 \
  --kernel_size 5 \
  --num_conv_layers 1 \
  --num_mlp_layers 5 \
  --use_SRM \
  --input_dropout 0.5 \
  --dropout 0.1 \
  --span_loss_type l1 \
  --lw_reg 1.0 \
  --lw_cls 5.0 \
  --lw_sal 0.0 \
  --lw_saliency 0.0 \
  --lw_wattn 1.0 \
  --lw_ms_align 1.0 \
  --mr_only \
  --eval_full_only \
  --use_exist_head \
  --exist_pool mean \
  --exist_loss_coef 1.0 \
  --exist_gate_thd 0.5 \
  --nms_thd 0.7
