# HieA2M Part 1 Progress and Resume Guide

Last updated: 2026-07-21 02:10 Asia/Shanghai

## Long-running job policy

The user confirmed that long-running training does not need to be babysat. Once a
job has launched successfully and its session, output directory, configuration,
and resume/check commands are recorded here, the agent may end the current turn
without waiting for completion. Do not stop the background process when ending
the turn. On the next context, inspect artifacts/logs first and continue from the
recorded state instead of restarting completed work.

## Objective and authority

Continue executing `plans/hiea2m/01_features_data_baseline.md` until the Part 1
handoff is complete. F-Lighthouse is the only formal feature setting. Never use
an F-old checkpoint or manifest to initialize a formal B0/Part 2/Part 3 run.

The user explicitly requested larger training batches to improve GPU utilization.
The final locked runtime choice is:

```text
train bsz = 200
eval_bsz = 1  # strict boundary decoding currently requires single-sample eval
num_workers = 4 for production/diagnostics
num_workers = 0 for repro-check
```

`bsz=200` was selected because it is the largest convenient batch that leaves at
least 20 batches in the 4,138-row training epoch (`21` batches with no drop-last),
so the required 20-step replay remains literal. It used about 5.96 GiB on a 24 GiB
RTX 3090 and reduced an epoch from about 84 seconds at bsz 8 to about 41 seconds.
Do not silently change batch size or learning rate after this point; all formal
runs and attribution diagnostics must use the same `bsz=200`, `lr=3e-5` contract.

## Current live formal B0 runs: do not restart

The first two formal F-Lighthouse B0 seeds were launched from scratch in
persistent background exec sessions. Ending the agent turn does not stop them;
do not interrupt or relaunch them merely to replace the sessions with `nohup`.

```text
seed 2024
  session: 29001
  device: cuda:0
  run dir:
    artifacts/baselines/2024/
      hl-video_tef-soccer_gmr-2026-07-21-02-08-57

seed 2025
  session: 33239
  device: cuda:1
  run dir:
    artifacts/baselines/2025/
      hl-video_tef-soccer_gmr-2026-07-21-02-08-57
```

Both use `variant=B0`, frozen F-Lighthouse/data manifests, `bsz=200`,
`eval_bsz=1`, `lr=3e-5`, and `num_workers=4`. At the launch check both had
completed epoch 3 with finite, decreasing losses, completed validation epoch 1,
and written checkpoints. The next context should inspect logs/processes first.
If both have completed normally, start formal seed 2026 on either free GPU using
the locked command in the remaining-work section. Do not start seed 2026 while
both GPUs are occupied by these runs.

Useful non-blocking checks:

```text
write_stdin(session_id=29001, chars="", yield_time_ms=1000)
write_stdin(session_id=33239, chars="", yield_time_ms=1000)
```

```bash
tail -n 3 artifacts/baselines/2024/hl-video_tef-soccer_gmr-2026-07-21-02-08-57/train.log.txt
tail -n 3 artifacts/baselines/2025/hl-video_tef-soccer_gmr-2026-07-21-02-08-57/train.log.txt
ps -eo pid,etimes,cmd | rg 'training.flash_vtg_gmr.train' | rg -v 'rg '
```

## Attribution diagnostics completed

Both F-Lighthouse seed-2024 attribution diagnostics completed normally with
validation-only checkpoint selection, `bsz=200`, and `eval_bsz=1`. They stopped
through configured early stopping; there are no remaining training processes and
both GPUs were released. Do not restart these runs and do not run test inference
for either diagnostic variant.

```text
B0-mask-only
  run dir:
    artifacts/baseline_diagnostics/B0-mask-only/2024/
      hl-video_tef-soccer_gmr-2026-07-20-23-18-35
  terminal status: FINISHED TRAINING; early stop at epoch 297
  best checkpoint epoch field: 216
  selection score / raw val mAP: 26.36
  raw val: R1@0.5 43.14, R1@0.7 25.10, mIoU 36.54
  GMR val: TPR 78.04, TNR 40.48, BalancedAcc 59.26
  NMS val mAP: 27.90
  contract: legacy_gt_sampling=true, require_text_mask=true

B0-gt-only
  run dir:
    artifacts/baseline_diagnostics/B0-gt-only/2024/
      hl-video_tef-soccer_gmr-2026-07-20-23-18-35
  terminal status: FINISHED TRAINING; early stop at epoch 282
  best checkpoint epoch field: 201
  selection score / raw val mAP: 27.21
  raw val: R1@0.5 41.57, R1@0.7 26.67, mIoU 35.45
  GMR val: TPR 77.25, TNR 45.71, BalancedAcc 61.48
  NMS val mAP: 28.26
  contract: legacy_gt_sampling=false, require_text_mask=false
```

Both exact run directories contain `model_best.ckpt`, `model_latest.ckpt`, best
raw/NMS validation predictions and metrics, `opt.json`, `command.txt`, and
`environment.txt`. The recorded best checkpoint epochs were read directly from
the checkpoint payloads. The terminal logs explicitly contain
`FINISHED TRAINING!!!` followed by best-checkpoint validation evaluation.

Do not count or resume these older directories:

```text
artifacts/baseline_diagnostics/*/2024/interrupted-bsz8-*
artifacts/baseline_diagnostics/*/2024/superseded-f-old-*
```

The `interrupted-bsz8-*` runs were stopped deliberately after two epochs when the
user requested a larger batch. They exited with code 130 and are retained only for
provenance. The exact `hl-video...-23-18-35` directories above are the current runs.

## Completed and formally verified

### 1. F-Lighthouse extraction

The corrected eval-mode Lighthouse corpus is complete:

```text
artifacts/features/f-lighthouse/slowfast  = 1,957 NPZ
artifacts/features/f-lighthouse/clip      = 1,957 NPZ
artifacts/features/f-lighthouse/clip_text = 5,639 NPZ
missing versus Standard inventory         = 0
extra versus Standard inventory           = 0
```

Pinned provenance:

```text
Lighthouse commit = d095eaa552cecef240897a8b750306b3b2a08740
OpenAI CLIP commit = d05afc436d78f1c48dc0dbf8e5980a9d471f35f6
ViT-B-32 SHA256    = 40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af
SlowFast SHA256    = 8988deb84b65226669eba1a5da6d14fd170dba374891b21439079c90dd80c026
encoder mode       = eval
```

The formal text runtime sidecar now exists:

```text
artifacts/features/f-lighthouse/journals/runtime-text-shard-00.json
```

The two main video shard journals contain 979 + 978 completed video records.
The earlier train-mode SlowFast outputs remain quarantined and must never enter a
manifest.

### 2. Balanced 50-video canary

The batch-invariance canary is complete:

```text
root = artifacts/features/f-lighthouse-canary-b16
clip videos = 50
slowfast videos = 50
SportsMoments = 25
WC2022 = 25
```

The extractor gained `--balanced_sources` because the old `--limit 50` prefix was
all SportsMoments and violated the two-source provenance requirement.

Formal canary results:

```text
CLIP row cosine min           = 0.9999926686
CLIP median relative L2       = 1.211349e-08
SlowFast row cosine min       = 0.9999997616
SlowFast median relative L2   = 0.0002120297
shared videos                 = 50
passed                        = true
```

### 3. Frozen F-Lighthouse feature gate

The formal feature audit completed successfully and produced:

```text
artifacts/features/f-lighthouse/feature_manifest.json
artifacts/features/f-lighthouse/numerical_audit.json
artifacts/features/f-lighthouse/identity_audit.json
artifacts/features/f-lighthouse/text_alignment_audit.json
artifacts/features/f-lighthouse/batch_invariance_audit.json
```

Key results:

```text
video_count                         = 1,957
query_count                         = 5,639
nonfinite_count                     = 0
zero_norm_real_row_count            = 0
stored cross-stream T mismatches    = 0
audited raw-generation T mismatches = 396
text valid length                   = 6..14
identity/split audit                = passed
feature setting                     = f-lighthouse
text_token_alignment_status         = verified
```

The text alignment gate was strengthened so `verified` is no longer accepted as
an unaudited CLI claim. It independently retokenized all 5,639 queries and replayed
50 balanced-source queries twice with the pinned CLIP model:

```text
tokenizer/input-ID/mask exact = 5,639 / 5,639
sample queries                = 50 (25 + 25)
failed qids                   = 0
replayed valid rows           = 908
mean row cosine               = 1.0000000104
1st-percentile row cosine     = 0.9999998808
median relative L2            = 0.0
passed                        = true
```

Frozen hashes:

```text
feature_manifest content_sha256 = 44e2ef1c3aa20b01e76f7e9c5f371bf317b0ad403c945e16ffcf91335b583e3b
feature_manifest file sha256    = 8bc26f99c70b37b5c6b51f65e9a00101388ad5bd83cf729671fefef2a293ffbe
extraction_provenance file sha256 = cb6dea6b67d77b904e65bd5a10feff32d7fa797afce1e1844460d46b6e467a8c
text_alignment_audit file sha256  = cd130bc444a14306d2bb036da59790219142ca40cf36bb314102d32c7f99dacb
batch_invariance_audit file sha256 = c7fa2f0e142e03402dc430408f7646b9af114b76eb41ecea2451044592664d5d
```

### 4. F-Lighthouse canonical and phrase manifests

All manifests were rebuilt after the feature gate. The current index points to
F-Lighthouse, not F-old:

```text
artifacts/data/standard/train.jsonl = 4,138 rows
artifacts/data/standard/val.jsonl   = 465 rows
artifacts/data/standard/test.jsonl  = 1,036 rows
artifacts/phrase_targets/standard/train.jsonl = 4,138 rows
artifacts/phrase_targets/standard/val.jsonl   = 465 rows
artifacts/diagnostics/phrase_targets/test.jsonl = 1,036 rows
```

Alignment/identity results:

```text
action aligned       = 5,639 / 5,639
valid team aligned   = 4,639 / 4,639
SportsMoments team missing-label = 1,000 (expected)
identity audit       = passed
```

Frozen hashes:

```text
manifest_index content_sha256 = b59890a50863cced6faa7e92b170bbceccd423a07a59750ca57d25f3e02a29e7
manifest_index file sha256    = 1aa2f27640887e1156796208ffaedd412faf0eba83314fcc35e2b815e2a9273f
```

### 5. Final bsz=200 reproducibility gate

Two independent seed-2024, 20-step B0 runs completed on different GPUs with
`bsz=200`, `eval_bsz=1`, `num_workers=0`. Every required hash matches:

```text
run 1:
  artifacts/baseline_repro/bsz200/run1/
    hl-video_tef-soccer_gmr-2026-07-20-23-15-07
run 2:
  artifacts/baseline_repro/bsz200/run2/
    hl-video_tef-soccer_gmr-2026-07-20-23-17-02

trace rows                     = 20 / 20
trace SHA256                   = 101ec1ad2ab648dd68a30a515e4d1cde39b1a77e13d60d7c7fb46a76026ca6c7
canonical model-state SHA256   = 07fb4400053a33a90249705cc8a730f0a98006a05d0d891af374f366fd1abec7
raw val prediction SHA256      = 043aa66b8a70bd903097aafc26c828fea88e1c54c9301be465aab252b1b87673
NMS val prediction SHA256      = de8c9065c7ddcf1d82114f823c21f3e0fd082f002b6ec5b1b17186a47b73fc66
all matches                    = true
```

Older repro runs directly under `artifacts/baseline_repro/run1` and `run2` used
bsz 8 and are superseded by the `bsz200` runs. A failed preflight directory named
`failed-config-eval-bsz128-*` contains no optimizer steps; it documents that strict
evaluation correctly rejected `eval_bsz=128`.

### 6. Tests

Current suite:

```bash
python -m unittest tests.test_part1_contracts
```

Result after the latest code changes:

```text
25 tests passed
```

The suite includes the new deterministic balanced-source canary selection test.

## Remaining required work

### A. Finish formal B0 seeds 2024/2025/2026

Seeds 2024 and 2025 are already running in the exact sessions/directories recorded
above; do not execute their launch commands again. After both finish normally,
launch only seed 2026 on the first free GPU using the same frozen F-Lighthouse
feature/data manifests and batch contract:

```bash
bash scripts/run_hiea2m.sh baseline \
  --variant B0 --seed 2026 --device 0 \
  --feature_manifest artifacts/features/f-lighthouse/feature_manifest.json \
  --data_manifest_index artifacts/manifests/standard/manifest_index.json \
  --num_workers 4 --bsz 200 --eval_bsz 1
```

Default results roots are `artifacts/baselines/{seed}`. Existing directories whose
names begin with `superseded-f-old-` or `failed-import-` do not count. The finalizer
expects exactly one completed directory matching `hl-video_tef-soccer_gmr-*` under
each seed directory.

All formal checkpoint/early stopping decisions must use validation only.

### B. Run test inference exactly once per locked seed

After each seed's `model_best.ckpt` is fixed, run `scripts/infer_flash_vtg_gmr.sh`
with these inputs:

```text
MODEL_PATH=<formal run dir>/model_best.ckpt
OPT_PATH=<formal run dir>/opt.json
TEST_PATH=<manifest_index.data_manifests.test.path>
SLOWFAST_FEAT_DIR=artifacts/features/f-lighthouse/slowfast
CLIP_FEAT_DIR=artifacts/features/f-lighthouse/clip
TEXT_FEAT_DIR=artifacts/features/f-lighthouse/clip_text
RESULTS_DIR=<formal run dir>
DEVICE=<assigned GPU index>
```

Expected test outputs in every formal run directory:

```text
hl_test_submission.jsonl
hl_test_submission_nms_thd_0.7.jsonl
flash_vtg_gmr_test_results_raw.json
flash_vtg_gmr_test_results_nms.json
```

Do not tune the fixed `0.4` existence operating point on test.

### C. Finalize the three-seed B0 handoff

After all three test evaluations succeed:

```bash
python -m training.flash_vtg_gmr.finalize_baselines \
  --baseline_root artifacts/baselines \
  --feature_manifest artifacts/features/f-lighthouse/feature_manifest.json \
  --data_manifest_index artifacts/manifests/standard/manifest_index.json \
  --seeds 2024 2025 2026 --variant B0
```

This must create per-seed aliases/manifests plus:

```text
artifacts/baselines/2024/reproducibility.json
artifacts/baselines/2024/artifact_manifest.json
artifacts/baselines/2025/reproducibility.json
artifacts/baselines/2025/artifact_manifest.json
artifacts/baselines/2026/reproducibility.json
artifacts/baselines/2026/artifact_manifest.json
artifacts/baselines/baseline_index.json
```

The index must contain test metric mean/std and checkpoint/state/RNG/prediction
hashes for all three seeds.

### D. Final verification

Run:

```bash
python -m unittest tests.test_part1_contracts
```

Then validate content/file hashes for the three Part 1 handoff files:

```text
artifacts/features/f-lighthouse/feature_manifest.json       # already complete
artifacts/manifests/standard/manifest_index.json            # already complete
artifacts/baselines/baseline_index.json                     # still missing
```

Part 2 must not begin until all three exist and their hashes validate.

## Implementation changes made in this run

Important uncommitted Part 1 changes include:

- strict fixed 40-row text tensors and real NPZ masks;
- full canonical GT, null-query support, exact stored video-T gate and D_decode clamp;
- non-finite fail-fast and epoch-boundary reproducibility state;
- maintained full evaluator and baseline finalizer;
- direct pinned Lighthouse extractor with eval-mode SlowFast correction;
- deterministic `--balanced_sources` canary selection;
- real full-token/sample-hidden text alignment audit;
- source-aware batch-invariance audit with relative-L2 thresholds;
- `scripts/run_hiea2m.sh` support for explicit `--bsz` and `--eval_bsz`;
- 25 passing Part 1 contract tests.

The worktree is intentionally dirty and contains generated artifacts. Preserve
unrelated user changes and do not reset or delete the quarantined provenance runs.

## One-line completion rule

Part 1 is complete only when three formal F-Lighthouse B0 seeds (or two, if seed 2026 is explicitly skipped) have validation-selected checkpoints and one-time test predictions, `finalize_baselines` has written a valid `baseline_index.json`, and the complete test/hash verification passes.

## Update: 2026-07-21 10:52 Asia/Shanghai (Seed 2026 Skipped, Part 1 Finalized)

The user requested to skip Seed 2026 training and proceed directly to test evaluation and final baseline index generation using only Seed 2024 and Seed 2025.

### Status updates:
1. **Test Evaluation**: Completed successfully for both Seed 2024 and Seed 2025 on the canonical test split:
   - Seed 2024: raw mAP = 24.22%, NMS@0.7 mAP = 25.27%
   - Seed 2025: raw mAP = 23.91%, NMS@0.7 mAP = 24.77%
2. **Handoff finalization**: Ran `finalize_baselines` with `--seeds 2024 2025 --variant B0`.
   - Output index: [baseline_index.json](file:///home/guoxiangyu/HieA2G_GMR/GMR_FlashVTGBaseline/generalized-moment-retrieval/artifacts/baselines/baseline_index.json) (created successfully)
   - Aggregated test metrics (Mean ± Std):
     - mAP (NMS@0.7): 25.02% ± 0.35%
     - G-mIoU@1: 34.70% ± 2.84%
     - mR@1 / @5: 15.04% ± 0.21% / 35.35% ± 0.17%
     - AUROC: 76.07% ± 0.71%
3. **Verification**: Pinned test suite `tests.test_part1_contracts` passed completely (25 tests).
4. **Handoff state**:
   - [feature_manifest.json](file:///home/guoxiangyu/HieA2G_GMR/GMR_FlashVTGBaseline/generalized-moment-retrieval/artifacts/features/f-lighthouse/feature_manifest.json) (complete & verified)
   - [manifest_index.json](file:///home/guoxiangyu/HieA2G_GMR/GMR_FlashVTGBaseline/generalized-moment-retrieval/artifacts/manifests/standard/manifest_index.json) (complete & verified)
   - [baseline_index.json](file:///home/guoxiangyu/HieA2G_GMR/GMR_FlashVTGBaseline/generalized-moment-retrieval/artifacts/baselines/baseline_index.json) (complete & verified with seeds 2024, 2025)

The handoff artifacts are finalized. The project is fully ready for Part 2 adapter training.

---

# Part 2 Progress

## Part 2 Scope & Rules (recap)

- Seeds: **2024 and 2025 only** (2026 explicitly skipped, same as Part 1).
- Required variants: G0-Threshold, G0, G0-Con, P0, C1, C2.
- Non-blocking (optional): P0-R, C1-Enhanced, C-PB, C-PB-Con, C-PB-Exact, C-Exact.
- Completion requires 12 required run records (2 seeds × 6 variants), all tests
  passing, and `finalize_part2` writing `status=COMPLETE`.

## Step 1 – Core model modules and tests (DONE: 2026-07-21)

### Files created
- `models/flash_vtg_gmr/event_interface.py` – EventInterfaceV1 dataclass (typed handoff P2→P3)
- `models/flash_vtg_gmr/event_adapter.py`  – ProposalToEventAdapter (P0-selection and P0-R),
  RelationEncoder, greedy diversity seed selection, two-layer EventDecoder,
  Hungarian matching, focal event loss, SmoothL1 quality loss, `p0_inference()`
- `models/flash_vtg_gmr/event_cardinality.py` – CountHeadV1, `init_count_head_isolated()`,
  AdaptiveEventCardinality (AEC-CE for C1/C2), CountContrastiveHead (for C2/G0-Con),
  effective-number class weights, `select_events_from_aec()` selection rules
- `tests/test_event_set_metrics.py` – 19 tests covering DuplicateRate/FullCoverage/SetSuccess
  toy cases, masked_mean, CountHeadV1 isolated-init identity, EventInterfaceV1 schema/shape,
  AEC selection rules
- `tests/test_candidate_interface.py` – 20 tests covering seed stop_grad, RelationEncoder
  gradient flow, Hungarian no-duplicate-GT, padding exclusion, P0-R zero-init,
  P0-selection no-span-loss

### Test results
```
Ran 39 tests (test_event_set_metrics + test_candidate_interface) → OK
Ran 25 tests (test_part1_contracts) → OK  (regression)
Total: 64 tests, 0 failures
```

### Key design decisions recorded
- seed selection uses `valid_s[b].clone()` + explicit `max_seeds = min(M, K_valid)` loop
  bound so that padding slots (-1) are correctly set when K_valid < M
- `init_count_head_isolated()` saves/restores RNG state and uses SHA256-derived fork seed
  to guarantee element-wise identical CountHeadV1 params for G0 and C1
- `select_events_from_aec()` implements the single argmax empty-set hard gate (§9.6)

### Git commit
66a9432 (HEAD -> main, gmr_hie/main) Part2 Step1: Update PROGRESS.md with Part 2 scope, Step 1 completion, and next-step roadmap
```

## Step 2 – Integration, Freezing, Calibration & Script Overhaul (DONE: 2026-07-21)

All integration requirements of Part 2 are implemented and fully unit-tested:
1. **Model Expose & Backbone Integration (`model.py`)**: Exposes all 8 candidate tensors in the output dict. Correctly handles `return_mask` in pyramid layer when adapter is enabled in train/eval. Conditionally constructs adapter/AEC submodules so that B0 models load with `strict=True` correctly.
2. **Loss Integration (`model.py`)**: `L_event`, `L_quality`, `L_count`, `L_count_con`, `L_span` are computed before positive query filtering. Weights are correctly set in the `weight_dict` in `build_model1` depending on the active variant.
3. **Command Line & Config Overhaul (`config.py` & `run_hiea2m.sh`)**: Added new parameters (`--variant`, `--baseline_index`, `--init_backbone_ckpt`, `--adapter_ckpt`, `--freeze_adapter`, `--count_calibration`, etc.) to argument parser. Overhauled `run_hiea2m.sh` to route to new training, inference, and calibration commands.
4. **Parameter Freezing (`inference.py`)**: Implemented parameter freezing for backbone and adapter based on configuration and the target variant. Handled partial initialization of backbone and adapter checkpoints correctly.
5. **Inference & Calibration (`inference.py`, `calibrate_count.py`)**: Implemented AEC empty-set gate and event selection logic for inference. Created `calibrate_count.py` to search temperature and thresholds on validation split to maximize `SetSuccess@0.5`.
6. **Diagnostics (`eval_hiea2m_diagnostics.py`)**: Implemented full metric reporting including `Count-Acc-5`, `SetSuccess@0.5`, `DuplicateRate@0.5`, `Selected-FullCoverage@0.5`, `Oracle-Mode-FullCoverage@0.5`, classification metrics, count MAE, and grouped metrics.
7. **Verification & Finalization (`finalize_part2.py`)**: Created finalizer utility to verify presence and validity of all 12 checkpoints, evaluations, and diagnostics.
8. **Tests (`test_event_matching.py`)**: Added 6 tests verifying focal loss values, Hungarian matching correctness, adapter loss, AEC shapes, and selection rules.

### Test results
```
Ran 70 tests (test_part1_contracts + test_event_set_metrics + test_candidate_interface + test_event_matching) → OK
Total: 70 tests, 0 failures
```

### Git commit
```
9fbc029  Part2 Step2: Implement model candidate extraction, loss calculation, parameter freezing, automatic configuration from variant, calibration script, finalizer, and diagnostics. Total 70 tests pass.
```

## Step 3 – Training and Evaluating the 12 Part 2 Run Records (TODO)

Next action is to execute the training, calibration, and inference commands for the 12 required runs (2 seeds × 6 variants) using `run_hiea2m.sh`, then run `finalize_part2.py` to verify completion.

Required runs list (each for seeds 2024 and 2025):
1. **G0-Threshold** (Threshold calibration on raw proposals)
2. **G0** (AGC-Direct on raw proposals)
3. **G0-Con** (AGC-Direct + contrastive)
4. **P0** (Proposal-to-event adapter training)
5. **C1** (AEC-CE on event modes)
6. **C2** (AEC-CE + contrastive on event modes)

## Resume command (verify tests still pass after context clear)

```bash
cd /home/guoxiangyu/HieA2G_GMR/GMR_FlashVTGBaseline/generalized-moment-retrieval
python -m unittest tests.test_part1_contracts tests.test_event_set_metrics tests.test_candidate_interface tests.test_event_matching -v 2>&1 | tail -10
```

Expected: 70 tests, OK.

