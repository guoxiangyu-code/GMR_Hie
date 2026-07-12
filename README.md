# Generalized Moment Retrieval

Official repository for **Retrieving Any Relevant Moments: Benchmark and Models for Generalized Moment Retrieval**.

**Generalized Moment Retrieval (GMR)** extends video moment retrieval to a unified setting where a query may correspond to **no moment**, **one moment**, or **multiple moments** in a video. A GMR system must retrieve the complete set of relevant temporal moments, or correctly return an empty set when the queried event is absent.

<div align="center">
  <a href="https://arxiv.org/abs/2605.02623"><img src="https://img.shields.io/badge/Paper-arXiv%3A2605.02623-b31b1b?style=for-the-badge" alt="Paper" /></a>
  <a href="https://dymm9977.github.io/generalized-moment-retrieval/"><img src="https://img.shields.io/badge/Project_Page-Visit-2563eb?style=for-the-badge" alt="Project Page" /></a>
  <a href="https://huggingface.co/datasets/diiiA22B9S/Soccer-GMR"><img src="https://img.shields.io/badge/Hugging_Face-Soccer--GMR-f59e0b?style=for-the-badge" alt="Hugging Face" /></a>
</div>

![Three retrieval scenarios in Generalized Moment Retrieval](assets/intro.png)

## Task

Traditional Video Moment Retrieval (VMR) commonly assumes that every query has exactly one matching temporal segment. This assumption is too restrictive for realistic retrieval: an event can be absent, occur once, or occur repeatedly within the same video.

GMR unifies these cases into one retrieval task:

- **Null-set rejection**: return an empty set when the event does not appear.
- **Single-moment retrieval**: retrieve the only relevant temporal segment.
- **Multi-moment retrieval**: retrieve all relevant temporal segments.

The output for each query is a set of temporal windows:

```json
{
  "qid": 580,
  "pred_relevant_windows": [[26.0, 34.0, 0.91], [104.0, 112.0, 0.87]],
  "pred_exist_score": 0.95
}
```

## Pipeline

Soccer-GMR is built through a duration-flexible semi-automated data construction pipeline with human verification. The pipeline converts timestamped soccer event supervision into generalized moment retrieval annotations by constructing natural-language queries, sampling positive and in-domain negative query-video pairs, and normalizing temporal annotations into evaluation-ready windows.

![Soccer-GMR construction pipeline](assets/pipeline222.png)

The duration-flexible design allows the benchmark to scale from fixed 150-second clips to longer video horizons by merging adjacent clips while preserving moment-level supervision.

## Dataset

**Soccer-GMR** is a large-scale GMR benchmark built on challenging soccer videos. It includes realistic positive and negative query-video pairs and covers all three retrieval scenarios in a unified format.


| Split group | Purpose | Files |
| --- | --- | --- |
| `data/label/Standard/` | Benchmark split used for evaluation | `train.jsonl`, `val.jsonl`, `test.jsonl` |
| `data/label/Full/` | Complete dataset used for scaling studies | `train.jsonl`, `val.jsonl`, `test.jsonl`, `full.jsonl` |

Dataset statistics:

| Split group | Train | Val | Test | Total | Videos |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Standard` | 4,138 | 465 | 1,036 | 5,639 | 1,957 |
| `Full` | 16,898 | 2,235 | 2,986 | 22,119 | 5,468 |

For details about label fields and directory organization, see [`data/README.md`](data/README.md). Large assets, including videos, features, and model weights, are hosted on [Hugging Face](https://huggingface.co/datasets/diiiA22B9S/Soccer-GMR). Access is manually reviewed after the requester completes the Soccer-GMR NDA form. Commercial use, redistribution, public hosting, or sharing access links is not permitted.

<p align="center">
  <img src="assets/dataset.png" alt="Statistics of Soccer-GMR" width="720" />
</p>

## Metrics

GMR evaluation requires measuring both rejection and localization. The official evaluation toolkit is provided in [`eval/`](eval/).

The evaluation protocol reports:

- **Null-set rejection**: AUROC, Rej-F1, Acc
- **Temporal localization on positive queries**: mAP, mR@k, mR+@k, mIoU@k, mIoU+@k
- **End-to-end GMR performance**: G-mIoU@k

Run the example evaluation:

```bash
python eval/eval_main.py \
  --submission_path eval/example/example_test_submission.jsonl \
  --gt_path data/label/Standard/test.jsonl \
  --save_path eval/example/example_test_results.json
```

For input formats, metric definitions, and command-line options, see [`eval/README.md`](eval/README.md).

## Methods

The paper studies GMR across two modeling paradigms. This repository releases the feature-level implementations of **Moment-DETR-GMR** and **FlashVTG-GMR**, two discriminative baselines augmented with the GMR Adapter.

**GMR Adapter** is a lightweight plug-and-play module for discriminative VMR backbones. It adds an explicit existence-estimation branch for null-set prediction while preserving the temporal localization backbone. The released Moment-DETR-GMR code pools decoder query representations, predicts `pred_exist_score`, and uses binary existence supervision derived from whether `relevant_windows` is empty.

![Architecture of the GMR Adapter](assets/adapter10.png)

Training and inference entry points are provided in [`training/`](training/), with runnable script templates in [`scripts/`](scripts/). See the [FlashVTG-GMR documentation](models/flash_vtg_gmr/README.md) for its environment, training, and inference details.

**GMR-tailored GRPO Reward** adapts reinforcement learning for generative multimodal large language models by jointly rewarding correct rejection behavior and temporal localization quality.

## Main Results

The main results below are reported on the Soccer-GMR `Standard` benchmark split.

| Model | AUROC | Rej-F1 | mAP | mR@5 | mR+@5 | G-mIoU@1 | G-mIoU@3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Moment-DETR | 69.92 | 0.00 | 6.98 | 10.92 | 0.78 | 5.39 | 2.47 |
| Moment-DETR-GMR | 72.09 | **64.01** | 7.52 | 12.96 | 0.84 | 35.84 | 32.89 |
| EaTR | 70.99 | 0.80 | 18.48 | 25.27 | 11.81 | 12.94 | 6.67 |
| EaTR-GMR | **79.11** | 62.10 | 18.56 | 24.43 | 13.97 | 37.89 | 31.95 |
| FlashVTG | 57.33 | 7.12 | 23.61 | 33.06 | 15.30 | 15.41 | 8.21 |
| FlashVTG-GMR | 74.00 | 61.72 | **24.62** | **33.36** | **19.10** | **39.58** | **33.53** |

## Repository Structure

```text
Generalized_Moment_Retrieval/
|-- README.md
|-- LICENSE
|-- CITATION.cff
|-- requirements.txt
|-- assets/
|   |-- intro.png
|   |-- pipeline222.png
|   |-- dataset.png
|   `-- adapter10.png
|-- data/
|   |-- README.md
|   `-- label/
|       |-- Full/
|       `-- Standard/
|-- eval/
|   |-- README.md
|   |-- eval_main.py
|   |-- metrics.py
|   |-- normalization.py
|   |-- utils.py
|   `-- example/
|-- configs/
|   |-- moment_detr_gmr/
|   `-- flash_vtg_gmr/
|-- models/
|   |-- moment_detr_gmr/
|   `-- flash_vtg_gmr/
|-- training/
|   |-- moment_detr_gmr/
|   `-- flash_vtg_gmr/
|-- scripts/
|   |-- train_moment_detr_gmr.sh
|   |-- infer_moment_detr_gmr.sh
|   |-- train_flash_vtg_gmr.sh
|   `-- infer_flash_vtg_gmr.sh
|-- docs/
|   |-- index.html
|   |-- styles.css
|   `-- script.js
`-- pipeline/
```

## Installation

```bash
pip install -r requirements.txt
```

PyTorch installation can vary by CUDA version. If needed, install the matching PyTorch build from the official PyTorch instructions before installing the rest of the requirements.

## Moment-DETR-GMR

Train with precomputed CLIP text features and CLIP + SlowFast video features:

```bash
bash scripts/train_moment_detr_gmr.sh
```

Run inference with a trained checkpoint:

```bash
bash scripts/infer_moment_detr_gmr.sh
```

The scripts can be configured with `MODEL_PATH`, `TEXT_FEAT_DIR`, `CLIP_FEAT_DIR`, `SLOWFAST_FEAT_DIR`, and `RESULTS_DIR`.

## FlashVTG-GMR

Train or evaluate FlashVTG-GMR with the provided entry points:

```bash
bash scripts/train_flash_vtg_gmr.sh
bash scripts/infer_flash_vtg_gmr.sh
```

See [`models/flash_vtg_gmr/README.md`](models/flash_vtg_gmr/README.md) for setup, required environment variables, and reproduction settings.

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@article{ding2026retrieving,
  title={Retrieving Any Relevant Moments: Benchmark and Models for Generalized Moment Retrieval},
  author={Ding, Yiming and Cao, Siyu and Jiao, Luyuan and Li, Yixuan and Wang, Zitong and Liu, Zhiyong and Zhang, Lu},
  journal={arXiv preprint arXiv:2605.02623},
  year={2026},
  doi={10.48550/arXiv.2605.02623}
}
```
