# Final Evaluation and Decision

## 日本語概要

本書は、保存済みの最終スコアへ一度だけ正解ラベルを結合し、指標、誤検知・見逃し、処理時間、固定順の合否判定を生成した経緯を記録します。入力の系譜、成果物、実行コマンド、書込み後に判明した表示処理の不具合の詳細は以下の英語本文を参照してください。

---

## English Summary

This record documents the one-time class reveal, metric reconstruction, failure selection, ordered gate evaluation, and immutable final bundle. It preserves source lineage and the post-write CLI summary defect without rerunning final scoring.

## Status

The fixed final evaluation is complete. Official final-test classes were
joined only after the first fixed scores, classifications, and latency evidence
had been preserved.

The evaluator implementation was committed as
`c6b4e5e164cc8788ff0428361406ada3e116543b`, and GitHub Actions CI #29 passed
before class reveal. That exact commit is recorded in the artifact manifest.

## Result

| Method | AUROC | AUPRC | Normal FPR | Anomaly recall | FP | FN | CPU p95 | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ECC residual | `0.8141` | `0.751334028468922` | `0.09` | `0.21` | 9 | 79 | `1.24699559 s` | `REJECT` |
| Patch HOG + One-Class SVM | `0.7837999999999999` | `0.7241747321517565` | `0.10` | `0.19` | 10 | 81 | `0.565766242 s` | `REJECT` |

The final test contains 100 normal and 100 anomaly images. Neither method had a
score-generation failure.

The ranking metrics are descriptive and do not override the operating-point
gates. Both methods failed the normal-FPR and anomaly-recall gates. ECC also
failed the CPU p95 gate. The reference-count, anomaly-training-label, and
reproducibility gates passed for both methods.

The first failed gate is `final_test_normal_fpr` for both methods because gates
are applied in their preregistered order. No weighted aggregate score or waiver
was used.

## Fixed lineage

Before class reveal, the evaluator hash-checks:

- the pre-evaluation freeze
- the v0.1 configuration
- the committed local-integrity record
- the normal-only calibration checkpoint and local fitted state
- the first fixed scoring checkpoint and local scoring state
- the label-free manifest set and final-test manifest
- the pinned official split CSV

It refuses changed, missing, extra, duplicated, or out-of-order final-test
paths. It does not rescore an image, refit a method, recalibrate a threshold, or
remeasure latency.

## Evaluation sequence

For each method, the evaluator performs the already implemented sequence:

1. join the fixed batch classifications to official per-path classes;
2. calculate image-level AUROC and AUPRC;
3. calculate fixed-threshold normal FPR, anomaly recall, FP count, and FN
   count;
4. mechanically select up to five highest-score false positives and five
   lowest-score false negatives;
5. apply the six preregistered hard gates in their fixed order;
6. record `ADOPT`, `ADOPT WITH CONDITIONS`, or `REJECT`.

No weighted aggregate score or hard-gate waiver is available.

## Failure-review disposition fixed before class reveal

The qualitative disposition is fixed as `guardrail_required` before VisA
classes or metrics are observed.

Rationale:

> Mechanical false-positive and false-negative selection is complete, but
> image content has not been reviewed at this decision stage; causal or
> intended-use boundary claims would be premature.

Condition if every hard gate passes:

> Review the selected false-positive and false-negative image content and
> define an operating guardrail before any follow-up trial.

This disposition cannot rescue a failed hard gate. Any hard-gate failure still
produces `REJECT`.

## Generated output

The fixed evaluation-artifact contract created:

```text
artifacts/v0.1/evaluation/visa-pcb1-v0-1-final/
├── artifact-manifest.json
├── ecc_residual/
│   ├── scores.csv
│   ├── classifications.csv
│   ├── revealed-labels.csv
│   ├── metrics.json
│   ├── latency.json
│   ├── latency-observations.csv
│   ├── failure-cases.csv
│   └── decision.json
└── patch_hog_one_class_svm/
    └── ...
```

The writer re-derives calibration, classification, class reveal, metrics,
failure selection, and decision from the preserved primitive objects before
atomically creating the non-overwritable directory.

## Recorded command

```bash
uv run --locked --no-sync python scripts/run_final_evaluation.py \
  --source-commit c6b4e5e164cc8788ff0428361406ada3e116543b
```

No raw VisA image or pixel mask will be copied into the evaluation bundle.

## Post-write CLI summary defect

The non-overwritable evaluation bundle was successfully validated and
atomically written. The CLI then exited with an error while printing its
human-readable summary because it requested `auroc` and `auprc` instead of the
artifact keys `image_level_auroc` and `image_level_auprc`.

This defect occurred after artifact creation. It did not alter class reveal,
metrics, failure selection, hard gates, decisions, or any bundle byte. The
original bundle remains the only fixed final evaluation. The CLI key names were
corrected without rerunning scoring or replacing the bundle.
