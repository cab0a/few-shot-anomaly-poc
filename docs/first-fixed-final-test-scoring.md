# First Fixed Final-Test Scoring

## 日本語概要

本書は、正解クラスを処理へ渡さずに実施した最初の最終評価スコアリングを記録します。両手法の固定閾値による分類、CPU処理時間、上書きできない成果物、使用した入力と実行コマンド、ラベル開示前の境界の詳細は以下の英語本文を参照してください。

---

## English Summary

This checkpoint preserves the first fixed final-test scores, classifications, and CPU latency evidence before per-path classes were joined. It records the exact inputs, source, command, output protections, and evaluation boundary.

## Status

The first fixed final-test scoring run completed successfully. It produced
scores, fixed-threshold classifications, and CPU latency evidence for both
methods without joining per-path final-test classes.

The runner implementation was committed as
`5b142f31c974334545ca2bb63bb7b2c6c514828a`, and GitHub Actions CI #27 passed
before scoring began. That exact commit is recorded as the run source.

## Result

| Method | Items | Score failures | Predicted normal | Predicted anomalous | CPU median | CPU p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ECC residual | 200 | 0 | 170 | 30 | `0.4369505 s` | `1.24699559 s` |
| Patch HOG + One-Class SVM | 200 | 0 | 171 | 29 | `0.4326800675 s` | `0.565766242 s` |

The latency values use all 600 timed observations per method: three passes over
the 200 final-test images after one complete warm-up pass. The ECC p95 is above
the preregistered one-second latency ceiling. This is a fixed observation, not
yet a complete method decision: per-path final-test classes, image-level
metrics, failure cases, and the ordered hard-gate result have not been
generated at this stage.

## Input boundary

The runner accepts:

- the frozen v0.1 configuration and pre-evaluation checkpoint
- the committed local-integrity and normal-only calibration checkpoints
- the hash-checked local fitted state
- the extracted local `pcb1` asset
- a manifest set and `final-test.jsonl` containing paths but no class or mask
  fields

The scoring interface does not accept the official split CSV, per-path classes,
threshold overrides, metric settings, failure-selection settings, or decision
settings.

The manifest loader verifies its recorded SHA-256, exact record keys, fixed
dataset and split identity, item count, unique IDs, unique source rows, unique
paths, and Unicode code-point path order. Unexpected metadata, including a
class or mask field, is rejected before any image is decoded.

## Fixed execution

For each of the 200 final-test paths, the runner:

1. decodes the local image as grayscale `uint8`;
2. creates one score from each frozen fitted method;
3. applies that method's already fixed normal-only threshold;
4. performs one warm-up pass and three timed passes for each method;
5. measures only the `decoded grayscale uint8 -> preprocessing -> image score`
   CPU boundary;
6. retains timings for score failures as required by the preregistered policy.

The implementation does not refit either method or recalibrate either
threshold.

## Non-overwritable outputs

The public directory is:

```text
artifacts/v0.1/scoring/first-fixed-final-test/
├── first-fixed-scoring.json
├── ecc_residual/
│   ├── scores.csv
│   ├── classifications.csv
│   ├── latency.json
│   └── latency-observations.csv
└── patch_hog_one_class_svm/
    ├── scores.csv
    ├── classifications.csv
    ├── latency.json
    └── latency-observations.csv
```

The exact Python score, classification, and latency objects needed by the next
evaluation stage are stored outside Git at:

```text
work/v0.1/final-test/first-fixed-scoring-state.pkl
```

The public checkpoint records local-state SHA-256
`159bc7eeb61f95c5008e1af8f4bb316d102a4f25a1b022cfd7852a7fb029e88b`.
Loading verifies
the digest before deserialization and revalidates the complete score,
classification, path-order, threshold, and latency contracts. The pickle is
only for trusted state generated locally by this repository.

## Recorded command

```bash
uv run --locked --no-sync python \
  scripts/run_first_fixed_final_test_scoring.py \
  --source-commit 5b142f31c974334545ca2bb63bb7b2c6c514828a \
  --dataset-root /path/to/extracted/VisA_20220922
```

The command requires a clean tree at the requested commit and refuses to
overwrite its public checkpoint or local state.

## Evaluation boundary

This stage read and scored final-test image content. It did not:

- read or join a per-path final-test class
- calculate AUROC, AUPRC, normal FPR, or anomaly recall
- select or display failure-case images
- change a threshold or hard gate
- produce an adoption decision
- display or commit raw VisA images

The next stage will introduce per-path classes through the already implemented
label-reveal boundary and will then calculate metrics, select failure cases,
and apply the fixed hard gates.
