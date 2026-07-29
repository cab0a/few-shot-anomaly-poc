# Normal-Reference Fitting and Threshold Calibration

## Status

The runner is implemented and covered by synthetic tests. It has not yet been
executed on VisA at this commit.

The implementation must pass CI before the real normal-only run. That CI-passed
commit will then be recorded as the run source. This prevents the runner from
being changed after its VisA thresholds are observed.

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

## Planned artifacts

The public, non-overwritable directory will be:

```text
artifacts/v0.1/calibration/normal-only/
├── normal-only-calibration.json
├── ecc_residual/
│   └── scores.csv
└── patch_hog_one_class_svm/
    └── scores.csv
```

The JSON checkpoint will record the source commit, frozen input hashes, fit
summaries, threshold evidence, score-artifact hashes, and evaluation-boundary
state. Each score CSV will contain all 884 label-free normal calibration
records and bounded method diagnostics.

Fitted arrays and scikit-learn estimator objects are needed by the first fixed
final-test run but are not suitable public Git artifacts. They will be stored
locally at:

```text
work/v0.1/calibration/normal-only-state.pkl
```

The checkpoint will record its SHA-256. Loading verifies that digest before
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

The command requires the requested source commit to equal the checked-out Git
`HEAD`, requires a clean tree, and refuses to overwrite outputs.

## Evaluation boundary

This stage may produce VisA-derived normal calibration scores and thresholds.
It does not:

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
