# Final Evaluation and Decision

## Status

The final evaluator is implemented and covered by synthetic tests. It has not
yet revealed VisA final-test classes or produced a VisA metric, failure-case
selection, or adoption decision at this commit.

The implementation must pass CI before class reveal. The exact CI-passed source
commit will then be supplied to the non-overwritable evaluator.

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

## Planned output

The fixed evaluation-artifact contract will create:

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

## Planned command

After the evaluator commit passes CI:

```bash
uv run --locked --no-sync python scripts/run_final_evaluation.py \
  --source-commit <full-ci-passed-commit>
```

No raw VisA image or pixel mask will be copied into the evaluation bundle.
