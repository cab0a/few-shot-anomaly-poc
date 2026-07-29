# Normal-Reference Fitting and Threshold Calibration

## 日本語概要

本書は、正常参照画像だけで両手法を学習し、正常校正画像だけから固定閾値を算出した実行記録です。入力境界、閾値、校正時の誤検知率、学習済み状態と成果物の識別情報、実行コマンド、最終評価を未使用とした範囲の詳細は以下の英語本文を参照してください。

---

## English Summary

This record preserves normal-reference fitting and normal-only threshold calibration for both methods. It fixes inputs, thresholds, calibration outcomes, fitted-state identities, generated artifacts, command lineage, and the boundary that excluded final-test information.

## Status

The fixed normal-reference fitting and normal-only threshold calibration run
completed successfully.

The runner implementation was committed as
`4fef91c1d1e339aa507cad80d51127e01046ae0b`, and GitHub Actions CI #25 passed
before the VisA run began. That exact commit is recorded as the run source. No
runner, method, configuration, partition, threshold rule, or acceptance gate
was changed after the thresholds were observed.

## Result

| Method | Fit result | Calibration score failures | Rank | Fixed threshold | Scores above threshold |
| --- | --- | ---: | ---: | ---: | ---: |
| ECC residual | 20 / 20 references accepted | 0 | 840 | `0.688464437424507` | 44 / 884 |
| Patch HOG + One-Class SVM | 225 / 225 scalers and 225 / 225 models fitted | 0 | 840 | `0.17611826509314352` | 44 / 884 |

The ECC template support fraction is `0.9028167724609375`. The Patch HOG model
collection contains 4,307 support vectors across its 225 position-wise models,
with 13 to 20 support vectors per position.

The realized normal calibration tail rate is `44 / 884 =
0.049773755656108594` for both methods. This is a consequence of selecting the
fixed rank and then applying a strict `score > threshold` rule. It is not the
final-test normal FPR, does not use an anomaly label, and provides no evidence
about anomaly recall.

The threshold source paths are:

- ECC residual: `pcb1/Data/Images/Normal/0691.JPG`
- Patch HOG + One-Class SVM: `pcb1/Data/Images/Normal/0660.JPG`

## Fixed input boundary

The run accepts only:

- the committed pre-evaluation freeze checkpoint
- the fixed VisA `pcb1` archive and extraction that match the local integrity
  checkpoint
- the fixed 20 normal reference paths
- the fixed 884 normal calibration paths
- the frozen v0.1 configuration and dependency lock

The runner rejects a changed frozen file, mismatched local asset, changed path
set, duplicate path, non-contiguous selection rank, dirty Git tree, partial
fitted state, or existing output.

No anomaly image or final-test path is an input to method fitting or threshold
calibration.

## Procedure

The runner performs one bounded sequence:

1. Verify all frozen files against
   `artifacts/v0.1/freeze/pre-evaluation-freeze.json`.
2. Re-run the complete local archive/extraction/split integrity comparison and
   require exact equality with `pcb1-local-integrity.json`.
3. Load the fixed normal partition manifest and require exactly 20 reference
   and 884 calibration paths.
4. Decode and preprocess the 20 reference images with the frozen shared
   preprocessing implementation.
5. Fit the ECC normal template.
6. Extract Patch HOG reference features, fit 225 position-wise
   `StandardScaler` instances, and fit 225 position-wise `OneClassSVM`
   instances.
7. Decode each normal calibration image once and score it with both fitted
   methods.
8. Apply the fixed normal-only nearest-rank 95th-percentile threshold rule
   independently to each method.
9. Validate that each threshold can be regenerated from the preserved scores.
10. Write an immutable public checkpoint and hash-checked local fitted state.

The fixed rank for 884 calibration images is:

```text
ceil(0.95 * 884) = 840
```

The threshold is the score at zero-based index 839 after sorting by anomaly
score and then relative path. A score is classified as anomalous only when
scoring failed or the score is strictly greater than the threshold.

## Generated artifacts

The public, non-overwritable directory is:

```text
artifacts/v0.1/calibration/normal-only/
├── normal-only-calibration.json
├── ecc_residual/
│   └── scores.csv
└── patch_hog_one_class_svm/
    └── scores.csv
```

The JSON checkpoint records the source commit, frozen input hashes, fit
summaries, threshold evidence, score-artifact hashes, and evaluation-boundary
state. Each score CSV contains all 884 label-free normal calibration
records and bounded method diagnostics.

Fitted arrays and scikit-learn estimator objects are needed by the first fixed
final-test run but are not suitable public Git artifacts. They are stored
locally at:

```text
work/v0.1/calibration/normal-only-state.pkl
```

The checkpoint records SHA-256
`d0056a52225d5600e5db9d0c11076a1fbd919f273b66f1e1b65cd5895e883cb4`.
Loading verifies that digest before
deserialization and revalidates the fitted objects and thresholds. The pickle
is only for state generated locally by this repository; it must not be replaced
with or loaded from an untrusted source.

## Command

After the implementation commit has passed CI and the standard data layout
exists:

```bash
uv run --locked --no-sync python scripts/run_normal_calibration.py
```

An existing local archive and extracted root may be supplied explicitly:

```bash
uv run --locked --no-sync python scripts/run_normal_calibration.py \
  --archive /path/to/VisA_20220922.tar \
  --dataset-root /path/to/extracted/VisA_20220922
```

For the recorded run, the command used the explicit CI-passed source commit and
pre-existing verified local asset:

```bash
uv run --locked --no-sync python scripts/run_normal_calibration.py \
  --source-commit 4fef91c1d1e339aa507cad80d51127e01046ae0b \
  --archive /path/to/VisA_20220922.tar \
  --dataset-root /path/to/extracted/VisA_20220922
```

The command requires the requested source commit to equal the checked-out Git
`HEAD`, requires a clean tree, and refuses to overwrite outputs.

## Evaluation boundary

This stage produced VisA-derived normal calibration scores and thresholds. It
did not:

- read or score a final-test image
- read or export a per-path final-test label
- join a label to a score
- calculate AUROC, AUPRC, final-test FPR, or anomaly recall
- measure the preregistered CPU final-test latency
- select a failure case
- apply a hard gate
- issue `ADOPT`, `ADOPT WITH CONDITIONS`, or `REJECT`
- change the threshold rule or any acceptance gate

## License boundary

Raw VisA images and fitted state remain outside Git. VisA-derived relative
paths and numerical calibration records remain attributable dataset-derived
material governed separately from the repository's PolyForm-licensed original
code and documentation. The PolyForm license does not replace or restrict
VisA's CC BY 4.0 terms.
