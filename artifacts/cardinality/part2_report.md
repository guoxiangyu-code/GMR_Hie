# Part 2 Report

Variant comparisons use frozen per-seed test results; no test result selected a checkpoint or calibration. The B0 → P0 acceptance gate is validation-only, as preregistered.

## Required per-seed metrics

### C1

- Seed 2024: Count-Acc-5=0.682432, SetSuccess@0.5=0.528958, mAP=3.990000, mR+@5=0.000000, G-mIoU@1=52.780000, G-mIoU@3=52.780000, G-mIoU@5=52.780000, AUROC=75.570000, Rej-F1=0.568807, Null-FPR=0.004065, Count-Acc-Exact-Selected=0.682432, Count-MAE-Selected=0.516409, OverPredictionRate=0.001931, UnderPredictionRate=0.315637, DuplicateRate@0.5=0.000000, Selected-FullCoverage@0.5=0.000000, Oracle-Mode-FullCoverage@0.5=0.343750
  - null: Count-Acc-5=0.995935, SetSuccess@0.5=0.995935, query_count=492.000000
  - single: Count-Acc-5=0.565104, SetSuccess@0.5=0.151042, query_count=384.000000
  - multi: Count-Acc-5=0.000000, SetSuccess@0.5=0.000000, query_count=160.000000
- Seed 2025: Count-Acc-5=0.684363, SetSuccess@0.5=0.561776, mAP=6.490000, mR+@5=0.000000, G-mIoU@1=54.940000, G-mIoU@3=54.940000, G-mIoU@5=54.940000, AUROC=72.820000, Rej-F1=0.570302, Null-FPR=0.000000, Count-Acc-Exact-Selected=0.684363, Count-MAE-Selected=0.514479, OverPredictionRate=0.000000, UnderPredictionRate=0.315637, DuplicateRate@0.5=0.000000, Selected-FullCoverage@0.5=0.000000, Oracle-Mode-FullCoverage@0.5=0.356250
  - null: Count-Acc-5=1.000000, SetSuccess@0.5=1.000000, query_count=492.000000
  - single: Count-Acc-5=0.565104, SetSuccess@0.5=0.234375, query_count=384.000000
  - multi: Count-Acc-5=0.000000, SetSuccess@0.5=0.000000, query_count=160.000000

### C2

- Seed 2024: Count-Acc-5=0.681467, SetSuccess@0.5=0.527992, mAP=3.990000, mR+@5=0.000000, G-mIoU@1=52.680000, G-mIoU@3=52.680000, G-mIoU@5=52.680000, AUROC=74.400000, Rej-F1=0.568063, Null-FPR=0.006098, Count-Acc-Exact-Selected=0.681467, Count-MAE-Selected=0.517375, OverPredictionRate=0.002896, UnderPredictionRate=0.315637, DuplicateRate@0.5=0.000000, Selected-FullCoverage@0.5=0.000000, Oracle-Mode-FullCoverage@0.5=0.343750
  - null: Count-Acc-5=0.993902, SetSuccess@0.5=0.993902, query_count=492.000000
  - single: Count-Acc-5=0.565104, SetSuccess@0.5=0.151042, query_count=384.000000
  - multi: Count-Acc-5=0.000000, SetSuccess@0.5=0.000000, query_count=160.000000
- Seed 2025: Count-Acc-5=0.683398, SetSuccess@0.5=0.560811, mAP=6.550000, mR+@5=0.000000, G-mIoU@1=54.880000, G-mIoU@3=54.880000, G-mIoU@5=54.880000, AUROC=73.460000, Rej-F1=0.571429, Null-FPR=0.002033, Count-Acc-Exact-Selected=0.683398, Count-MAE-Selected=0.514479, OverPredictionRate=0.000965, UnderPredictionRate=0.315637, DuplicateRate@0.5=0.000000, Selected-FullCoverage@0.5=0.000000, Oracle-Mode-FullCoverage@0.5=0.356250
  - null: Count-Acc-5=0.997967, SetSuccess@0.5=0.997967, query_count=492.000000
  - single: Count-Acc-5=0.565104, SetSuccess@0.5=0.234375, query_count=384.000000
  - multi: Count-Acc-5=0.000000, SetSuccess@0.5=0.000000, query_count=160.000000

### G0

- Seed 2024: Count-Acc-5=0.679537, SetSuccess@0.5=0.556950, mAP=7.430000, mR+@5=0.000000, G-mIoU@1=54.360000, G-mIoU@3=54.360000, G-mIoU@5=54.360000, AUROC=75.680000, Rej-F1=0.575484, Null-FPR=0.016260, Count-Acc-Exact-Selected=0.679537, Count-MAE-Selected=0.516409, OverPredictionRate=0.007722, UnderPredictionRate=0.312741, DuplicateRate@0.5=0.000000, Selected-FullCoverage@0.5=0.000000
  - null: Count-Acc-5=0.983740, SetSuccess@0.5=0.983740, query_count=492.000000
  - single: Count-Acc-5=0.572917, SetSuccess@0.5=0.242188, query_count=384.000000
  - multi: Count-Acc-5=0.000000, SetSuccess@0.5=0.000000, query_count=160.000000
- Seed 2025: Count-Acc-5=0.683398, SetSuccess@0.5=0.560811, mAP=6.590000, mR+@5=0.000000, G-mIoU@1=55.030000, G-mIoU@3=55.030000, G-mIoU@5=55.030000, AUROC=74.150000, Rej-F1=0.572178, Null-FPR=0.000000, Count-Acc-Exact-Selected=0.683398, Count-MAE-Selected=0.513514, OverPredictionRate=0.000000, UnderPredictionRate=0.316602, DuplicateRate@0.5=0.000000, Selected-FullCoverage@0.5=0.000000
  - null: Count-Acc-5=1.000000, SetSuccess@0.5=1.000000, query_count=492.000000
  - single: Count-Acc-5=0.562500, SetSuccess@0.5=0.231771, query_count=384.000000
  - multi: Count-Acc-5=0.000000, SetSuccess@0.5=0.000000, query_count=160.000000

### G0-Con

- Seed 2024: Count-Acc-5=0.682432, SetSuccess@0.5=0.559846, mAP=7.430000, mR+@5=0.000000, G-mIoU@1=54.650000, G-mIoU@3=54.650000, G-mIoU@5=54.650000, AUROC=75.710000, Rej-F1=0.577720, Null-FPR=0.010163, Count-Acc-Exact-Selected=0.682432, Count-MAE-Selected=0.513514, OverPredictionRate=0.004826, UnderPredictionRate=0.312741, DuplicateRate@0.5=0.000000, Selected-FullCoverage@0.5=0.000000
  - null: Count-Acc-5=0.989837, SetSuccess@0.5=0.989837, query_count=492.000000
  - single: Count-Acc-5=0.572917, SetSuccess@0.5=0.242188, query_count=384.000000
  - multi: Count-Acc-5=0.000000, SetSuccess@0.5=0.000000, query_count=160.000000
- Seed 2025: Count-Acc-5=0.683398, SetSuccess@0.5=0.560811, mAP=6.530000, mR+@5=0.000000, G-mIoU@1=54.990000, G-mIoU@3=54.990000, G-mIoU@5=54.990000, AUROC=73.980000, Rej-F1=0.570302, Null-FPR=0.000000, Count-Acc-Exact-Selected=0.683398, Count-MAE-Selected=0.514479, OverPredictionRate=0.000000, UnderPredictionRate=0.316602, DuplicateRate@0.5=0.000000, Selected-FullCoverage@0.5=0.000000
  - null: Count-Acc-5=1.000000, SetSuccess@0.5=1.000000, query_count=492.000000
  - single: Count-Acc-5=0.562500, SetSuccess@0.5=0.231771, query_count=384.000000
  - multi: Count-Acc-5=0.000000, SetSuccess@0.5=0.000000, query_count=160.000000

### G0-Threshold

- Seed 2024: Count-Acc-5=0.477799, SetSuccess@0.5=0.477799, mAP=0.470000, mR+@5=0.000000, G-mIoU@1=47.730000, G-mIoU@3=47.730000, G-mIoU@5=47.730000, AUROC=50.450000, Rej-F1=0.021779, Null-FPR=0.002033, Count-Acc-Exact-Selected=0.477799, Count-MAE-Selected=0.719112, OverPredictionRate=0.000965, UnderPredictionRate=0.521236, DuplicateRate@0.5=0.000000, Selected-FullCoverage@0.5=0.000000
  - null: Count-Acc-5=0.997967, SetSuccess@0.5=0.997967, query_count=492.000000
  - single: Count-Acc-5=0.010417, SetSuccess@0.5=0.010417, query_count=384.000000
  - multi: Count-Acc-5=0.000000, SetSuccess@0.5=0.000000, query_count=160.000000
- Seed 2025: Count-Acc-5=0.476834, SetSuccess@0.5=0.476834, mAP=0.380000, mR+@5=0.000000, G-mIoU@1=47.660000, G-mIoU@3=47.660000, G-mIoU@5=47.660000, AUROC=50.360000, Rej-F1=0.018182, Null-FPR=0.002033, Count-Acc-Exact-Selected=0.476834, Count-MAE-Selected=0.720077, OverPredictionRate=0.000965, UnderPredictionRate=0.522201, DuplicateRate@0.5=0.000000, Selected-FullCoverage@0.5=0.000000
  - null: Count-Acc-5=0.997967, SetSuccess@0.5=0.997967, query_count=492.000000
  - single: Count-Acc-5=0.007812, SetSuccess@0.5=0.007812, query_count=384.000000
  - multi: Count-Acc-5=0.000000, SetSuccess@0.5=0.000000, query_count=160.000000

### P0

- Seed 2024: Count-Acc-5=0.474903, SetSuccess@0.5=0.474903, mAP=0.000000, mR+@5=0.000000, G-mIoU@1=47.490000, G-mIoU@3=47.490000, G-mIoU@5=47.490000, AUROC=50.000000, Rej-F1=0.000000, Null-FPR=0.000000, Count-Acc-Exact-Selected=0.474903, Count-MAE-Selected=0.723938, OverPredictionRate=0.000000, UnderPredictionRate=0.525097, DuplicateRate@0.5=0.000000, Selected-FullCoverage@0.5=0.000000, Oracle-Mode-FullCoverage@0.5=0.343750
  - null: Count-Acc-5=1.000000, SetSuccess@0.5=1.000000, query_count=492.000000
  - single: Count-Acc-5=0.000000, SetSuccess@0.5=0.000000, query_count=384.000000
  - multi: Count-Acc-5=0.000000, SetSuccess@0.5=0.000000, query_count=160.000000
- Seed 2025: Count-Acc-5=0.474903, SetSuccess@0.5=0.474903, mAP=0.000000, mR+@5=0.000000, G-mIoU@1=47.490000, G-mIoU@3=47.490000, G-mIoU@5=47.490000, AUROC=50.000000, Rej-F1=0.000000, Null-FPR=0.000000, Count-Acc-Exact-Selected=0.474903, Count-MAE-Selected=0.723938, OverPredictionRate=0.000000, UnderPredictionRate=0.525097, DuplicateRate@0.5=0.000000, Selected-FullCoverage@0.5=0.000000, Oracle-Mode-FullCoverage@0.5=0.356250
  - null: Count-Acc-5=1.000000, SetSuccess@0.5=1.000000, query_count=492.000000
  - single: Count-Acc-5=0.000000, SetSuccess@0.5=0.000000, query_count=384.000000
  - multi: Count-Acc-5=0.000000, SetSuccess@0.5=0.000000, query_count=160.000000

## Preregistered comparisons

## G0-Threshold → G0

- Seed 2024: Count-Acc-5: +0.201737, SetSuccess@0.5: +0.079151, mAP: +6.960000
- Seed 2025: Count-Acc-5: +0.206564, SetSuccess@0.5: +0.083977, mAP: +6.210000

## G0 → G0-Con

- Seed 2024: Count-Acc-5: +0.002896, SetSuccess@0.5: +0.002896, mAP: +0.000000
- Seed 2025: Count-Acc-5: +0.000000, SetSuccess@0.5: +0.000000, mAP: -0.060000

## G0 → C1

- Seed 2024: Count-Acc-5: +0.002896, SetSuccess@0.5: -0.027992, mAP: -3.440000
- Seed 2025: Count-Acc-5: +0.000965, SetSuccess@0.5: +0.000965, mAP: -0.100000

## C1 → C2

- Seed 2024: Count-Acc-5: -0.000965, SetSuccess@0.5: -0.000965, mAP: +0.000000
- Seed 2025: Count-Acc-5: -0.000965, SetSuccess@0.5: -0.000965, mAP: +0.060000

## B0 → P0

- Seed 2024 (validation): mAP Δ -25.790000; oracle coverage Δ -0.655556; gate=FAIL
- Seed 2025 (validation): mAP Δ -25.430000; oracle coverage Δ -0.588889; gate=FAIL

## Failure-mode diagnosis

- Both P0 runs have test mAP=0 and Selected-FullCoverage@0.5=0: the locked 0.5 event threshold selects an empty set, despite non-zero raw-proposal and oracle-mode coverage.
- C1/C2 obtain zero multi-query Count-Acc-5 for both seeds; their overall count accuracy is dominated by null/single queries and does not solve multi-event cardinality.
- C1/C2 Selected-FullCoverage@0.5 remains zero for both seeds, so neither CE nor the contrastive extension recovers complete multi-event sets.

## Research outcome

MIXED
