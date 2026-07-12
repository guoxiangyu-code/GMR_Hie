# Training and Inference

This directory contains the released feature-level training and inference code for **Moment-DETR-GMR** and **FlashVTG-GMR**.

## Input Format

Training and evaluation labels are JSONL files. Each line should contain:

```json
{
  "qid": 580,
  "query": "a corner kick",
  "vid": "match_0001_clip_0003.mp4",
  "duration": 150.0,
  "relevant_windows": [[26.0, 34.0], [104.0, 112.0]]
}
```

Use an empty list for null-set samples:

```json
{"qid": 581, "query": "a red card", "vid": "match_0001_clip_0003.mp4", "duration": 150.0, "relevant_windows": []}
```

## Feature Layout

The released code expects precomputed feature files:

```text
features/soccer_gmr/
|-- clip/
|   `-- <video_id_without_extension>.npz
|-- slowfast/
|   `-- <video_id_without_extension>.npz
`-- clip_text/
    `-- qid<qid>.npz
```

Video feature files should contain a `features` array. Text feature files should contain a `last_hidden_state` array. The default model concatenates CLIP and SlowFast video features and appends temporal endpoint features.

## Train

```bash
bash scripts/train_moment_detr_gmr.sh
```

Equivalent direct command:

```bash
python training/moment_detr_gmr/train.py \
  --dataset soccer_gmr \
  --feature clip_slowfast \
  --train_path data/label/Standard/train.jsonl \
  --eval_path data/label/Standard/val.jsonl \
  --t_feat_dir features/soccer_gmr/clip_text \
  --v_feat_dirs features/soccer_gmr/clip features/soccer_gmr/slowfast \
  --results_dir results/moment_detr_gmr
```

The training split keeps null-set samples when `use_exist_head: true`, so the GMR Adapter receives binary existence supervision from `relevant_windows`.

## Inference

```bash
bash scripts/infer_moment_detr_gmr.sh
```

Equivalent direct command:

```bash
python training/moment_detr_gmr/evaluate.py \
  --dataset soccer_gmr \
  --feature clip_slowfast \
  --model_path results/moment_detr_gmr/best.ckpt \
  --split test \
  --eval_path data/label/Standard/test.jsonl \
  --t_feat_dir features/soccer_gmr/clip_text \
  --v_feat_dirs features/soccer_gmr/clip features/soccer_gmr/slowfast \
  --results_dir results/moment_detr_gmr/test
```

The generated submission contains `pred_relevant_windows` and, when the checkpoint has the adapter head, `pred_exist_score`.
## Available Methods

- [`moment_detr_gmr/`](moment_detr_gmr/): Moment-DETR-GMR training and inference.
- [`flash_vtg_gmr/`](flash_vtg_gmr/): FlashVTG-GMR training and inference. See the [method documentation](../models/flash_vtg_gmr/README.md).
