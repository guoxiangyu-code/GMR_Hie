# GMR Evaluation Toolkit

This directory provides the official evaluation toolkit for **Generalized Moment Retrieval (GMR)**, including rejection, localization, and end-to-end GMR metrics.

## Overview

The toolkit evaluates GMR from three complementary perspectives:

- **Null-set rejection**
- **Temporal localization on positive queries**
- **End-to-end generalized moment retrieval**

It supports official benchmark evaluation on JSONL-formatted predictions and ground-truth annotations.

## Installation

Requirements:

- Python 3.9+
- `numpy`
- `scikit-learn`

Install dependencies with:

```bash
pip install -r ../requirements.txt
```

## Quick Start

Run evaluation with:

```bash
python eval_main.py \
  --submission_path /path/to/predictions.jsonl \
  --gt_path /path/to/ground_truth.jsonl \
  --save_path /path/to/results.json
```

You can also run a full benchmark-style example provided in this repository:

```bash
python eval_main.py \
  --submission_path example/example_test_submission.jsonl \
  --gt_path ../data/label/Standard/test.jsonl \
  --save_path example/example_test_results.json
```

If your ground truth is stored in intermediate timestamp form and needs timestamp-to-window expansion, add `--gt_ts_window_cfg /path/to/config.json`.

## Input Format

### Submission

Each prediction record should contain:

- `qid`: query identifier
- `pred_relevant_windows`: `[[start, end, score?], ...]`
- `pred_exist_score` (optional): explicit existence score for null-set rejection

If `pred_exist_score` is not provided, the toolkit uses the maximum window score as the existence score.

### Ground Truth

Each ground-truth record should contain:

- `qid`
- `relevant_windows`: `[[start, end], ...]`

Empty-set queries should use:

- `relevant_windows = []`

The toolkit also supports intermediate annotation forms such as `moment.type = "clips"` or `moment.type = "timestamps"`, which can be normalized into `relevant_windows`.

## Output Format

The evaluation result is saved as a JSON file containing:

- `brief`: a compact summary of the main metrics
- `GMR-CLS`: rejection-related details
- `G-mIoU_gate` and `G-mIoU_detail`: end-to-end GMR metrics and gating details
- `mAP_detail`, `mR_detail`, `mR+_detail`, `mIoU_detail`, `mIoU+_detail`
- `stats`: sample counts, thresholds, and evaluation metadata

Localization metrics are computed only on positive queries, while rejection and G-mIoU metrics are computed on all shared query IDs.

## Metrics

### Null-set Rejection

- `AUROC`
- `Rej-F1`
- `Acc`

These metrics evaluate whether the model can correctly reject queries with no relevant moment.

### Temporal Localization

- `mAP`
- `mR@k`
- `mR+@k`
- `mIoU@k`
- `mIoU+@k`

These metrics are computed on positive queries and evaluate temporal grounding quality.

### End-to-End GMR

- `G-mIoU@k`

This metric jointly evaluates rejection and localization in a unified end-to-end setting.

## Command-Line Arguments

Common options include:

- `--submission_path`: prediction JSONL file
- `--gt_path`: ground-truth JSONL file
- `--save_path`: output result JSON file
- `--k_list`: top-k values used for mR / mR+ / mIoU / G-mIoU
- `--gmiou_cls_threshold`: classification threshold used for G-mIoU gating
- `--max_pred_windows`: maximum number of prediction windows retained
- `--cls_thresholds`: thresholds reported for rejection-related metrics
- `--gt_ts_window_cfg`: optional config for timestamp-to-window expansion
- `--not_verbose`: disable brief console logging

Evaluation is performed on the intersection of submission and ground-truth `qid` values.

## Code Structure

- `eval_main.py`: CLI entry and evaluation pipeline
- `metrics.py`: metric implementations
- `normalization.py`: ground-truth normalization utilities
- `utils.py`: JSONL loading and numerical utilities
