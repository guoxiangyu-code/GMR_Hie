#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ $# -lt 1 ]]; then
  printf 'Usage: %s {G0|G0-Con|P0|C1|C2} [run_tag]\n' "$0" >&2
  exit 2
fi

VARIANT="$1"
RUN_TAG="${2:-strict-v1}"
case "${VARIANT}" in
  G0|G0-Con|P0|C1|C2) ;;
  *) printf 'Unsupported trainable Part 2 variant: %s\n' "${VARIANT}" >&2; exit 2 ;;
esac

PYTHON_BIN="${HIEA2M_PYTHON:-/home/guoxiangyu/miniconda3/envs/flashvtg/bin/python}"
FEATURE_MANIFEST="${HIEA2M_FEATURE_MANIFEST:-artifacts/features/f-lighthouse/feature_manifest.json}"
DATA_MANIFEST_INDEX="${HIEA2M_DATA_MANIFEST_INDEX:-artifacts/manifests/standard/manifest_index.json}"
BASELINE_INDEX="${HIEA2M_BASELINE_INDEX:-artifacts/baselines/baseline_index.json}"
ADAPTER_INDEX="${HIEA2M_ADAPTER_INDEX:-artifacts/adapters/public_adapter_index.json}"
BSZ="${HIEA2M_BSZ:-256}"
EPOCHS="${HIEA2M_EPOCHS:-15}"
WORKERS="${HIEA2M_NUM_WORKERS:-4}"
RUN_ROOT="${HIEA2M_RUN_ROOT:-artifacts/part2_runs/${RUN_TAG}}"

resolve_checkpoint() {
  "${PYTHON_BIN}" -m training.flash_vtg_gmr.resolve_artifact --index "$1" --seed "$2"
}

pids=()
for assignment in "2024:0" "2025:1"; do
  seed="${assignment%%:*}"
  gpu="${assignment##*:}"
  b0_ckpt="$(resolve_checkpoint "${BASELINE_INDEX}" "${seed}")"
  run_dir="${RUN_ROOT}/${seed}/${VARIANT}"
  if [[ -e "${run_dir}/model_best.ckpt" ]]; then
    printf 'Refusing to overwrite completed checkpoint: %s\n' "${run_dir}/model_best.ckpt" >&2
    exit 2
  fi
  args=(
    train --variant "${VARIANT}" --seed "${seed}"
    --feature_manifest "${FEATURE_MANIFEST}"
    --data_manifest_index "${DATA_MANIFEST_INDEX}"
    --baseline_index "${BASELINE_INDEX}"
    --results_dir "${run_dir}"
    --device "${gpu}" --num_workers "${WORKERS}" --bsz "${BSZ}"
    --n_epoch "${EPOCHS}" --eval_epoch 1 --repro_check
  )
  if [[ "${VARIANT}" == "C1" || "${VARIANT}" == "C2" ]]; then
    p0_ckpt="$(resolve_checkpoint "${ADAPTER_INDEX}" "${seed}")"
    args+=(--adapter_ckpt "${p0_ckpt}" --freeze_adapter)
  else
    args+=(--init_backbone_ckpt "${b0_ckpt}")
  fi
  bash scripts/run_hiea2m.sh "${args[@]}" >"/tmp/hiea2m-${RUN_TAG}-${seed}-${VARIANT}.log" 2>&1 &
  pids+=("$!")
  printf 'Started seed=%s variant=%s gpu=%s pid=%s run=%s\n' \
    "${seed}" "${VARIANT}" "${gpu}" "${pids[-1]}" "${run_dir}"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
