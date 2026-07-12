# FlashVTG-GMR

This directory contains the FlashVTG feature-level baseline adapted to
Generalized Moment Retrieval (GMR). The implementation adds an explicit
existence-estimation branch to FlashVTG while retaining its temporal
localization backbone.

The model returns temporal windows together with `pred_exist_score`, the
probability that the query has at least one relevant moment in the video.
Localization losses supervise positive pairs, while the existence loss uses
both positive and negative pairs.

## Installation

The released checkpoint was verified with Python 3.10.19, PyTorch 2.2.2,
torchtext 0.17.2, and nncore 0.4.7.
Install the FlashVTG-specific environment from the repository root:

```bash
pip install -r requirements-flash-vtg.txt
```

The code expects precomputed 2-second features:

- SlowFast video features with 2304 dimensions;
- CLIP video features with 512 dimensions;
- CLIP text features with 512 dimensions.

The video streams are concatenated into a 2816-dimensional representation.

## Checkpoint

Download `checkpoint/flashvtg_gmr` from the
[Soccer-GMR Hugging Face repository](https://huggingface.co/datasets/diiiA22B9S/Soccer-GMR/tree/main/checkpoint).
The checkpoint is loaded together with the sanitized release options in
`configs/flash_vtg_gmr/soccer_gmr.json`; an experiment-machine `opt.json` is
not required.

## Inference

Set the paths and run the repository script:

```bash
export MODEL_PATH=/path/to/flashvtg_gmr
export TEST_PATH=data/label/Standard/test.jsonl
export SLOWFAST_FEAT_DIR=/path/to/features/slowfast
export CLIP_FEAT_DIR=/path/to/features/clip
export TEXT_FEAT_DIR=/path/to/features/clip_text
export RESULTS_DIR=results/flash_vtg_gmr

bash scripts/infer_flash_vtg_gmr.sh
```

`eval_bsz=1` is required by the released boundary decoder. The script writes
the raw and NMS submissions and evaluates the NMS output with the GMR evaluator.

The paper checkpoint uses:

| Setting | Value |
| --- | ---: |
| Existence pooling | `mean` |
| NMS threshold | `0.7` |
| Rejection / G-mIoU threshold | `0.4` |
| Maximum evaluated windows | `10` |

## Training

Training uses the same feature layout:

```bash
export TRAIN_PATH=data/label/Standard/train.jsonl
export VAL_PATH=data/label/Standard/val.jsonl
export SLOWFAST_FEAT_DIR=/path/to/features/slowfast
export CLIP_FEAT_DIR=/path/to/features/clip
export TEXT_FEAT_DIR=/path/to/features/clip_text
export RESULTS_DIR=results/flash_vtg_gmr

bash scripts/train_flash_vtg_gmr.sh
```

## Acknowledgement

This implementation is based on
[FlashVTG](https://github.com/Zhuo-Cao/FlashVTG), the official implementation
of *FlashVTG: Feature Layering and Adaptive Score Handling Network for Video
Temporal Grounding* (WACV 2025). Upstream copyright and license notices are
retained in the corresponding source files. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for provenance details, and
cite the FlashVTG paper when using this baseline.
