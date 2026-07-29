# Evaluation Plan

## Status and Change Control

This is the preregistered v0.1 evaluation plan. It is written before algorithm implementation, data download, experiment execution, or result inspection.

The acceptance gates and decision rules in this document must not be changed after final-test results are inspected. If an implementation defect requires a protocol correction, the correction must be documented before rerunning the final test, and both the defect and the affected run must remain in the project history.

## Dataset

v0.1 uses only the `pcb1` category from the Visual Anomaly (VisA) dataset.

The official sources are:

- Amazon Science project: <https://github.com/amazon-science/spot-diff>
- AWS Open Data entry: <https://registry.opendata.aws/visa/>
- Official archive identifier: `VisA_20220922.tar`
- Dataset license: CC BY 4.0

The official archive was acquired outside Git on 2026-07-28. Its source,
observed identity, license boundary, and metadata-only structure checks are
recorded in [the acquisition record](data-acquisition-record.md). Raw VisA data
remains outside Git.

## Licensing Boundary

Original project code and documentation are source-available under the PolyForm Noncommercial License 1.0.0. Commercial use of those original materials requires a separate written license from the copyright holder.

VisA data and VisA content reproduced in future evaluation artifacts remain separately licensed under CC BY 4.0. The PolyForm license does not apply to VisA, and this repository must not impose additional restrictions on VisA data. Third-party dependencies remain governed by their respective licenses.

## Dataset Split

The official VisA one-class split will define the initial train and test boundary.

Within the official `pcb1` normal training partition:

1. Twenty images will form the normal reference partition.
2. Every other eligible normal training image will form the threshold-calibration partition.
3. No image may appear in both partitions.

The official one-class `pcb1` test partition will remain untouched until final evaluation. It will contain the final-test normal and anomaly examples defined by the official split.

The implementation must verify that relative paths are unique and that reference, calibration, and final-test path sets have zero intersection.

## Reference Selection

The fixed v0.1 seed is:

```text
42
```

Reference selection will use a version-independent deterministic ranking:

1. Normalize every eligible relative path to POSIX form.
2. Create the UTF-8 string:

   ```text
   few-shot-anomaly-poc:v0.1:42:<relative-path>
   ```

3. Calculate its SHA-256 digest.
4. Sort by digest and then by relative path as a deterministic tie-break.
5. Select the first 20 paths as the normal reference set.
6. Assign all remaining official normal training paths to threshold calibration.

The selected reference IDs, calibration IDs, official split source, archive checksum, and selection-procedure version must be committed before final-test scoring begins.

The fixed 20-image reference partition and 884-image calibration partition are
listed in
[`artifacts/v0.1/data/pcb1-normal-partitions.csv`](../artifacts/v0.1/data/pcb1-normal-partitions.csv).
Their acquisition context, selection procedure, and manifest checksum are
recorded in [the acquisition record](data-acquisition-record.md). The final-test
paths and class labels are not copied into that normal-only partition artifact.

## Method Fitting Boundary

Both methods may use only the 20 reference images to build their normal representation.

- The ECC residual method may build its normal template and any reference-derived normalization only from the reference partition.
- Patch HOG + One-Class SVM may fit feature scaling and the one-class estimator only from reference-derived samples.
- The calibration partition may be scored only to select the operating threshold.
- Calibration scores must not select the method, feature definition, SVM kernel, image size, residual aggregation, or another implementation parameter.

All method parameters and preprocessing rules are preregistered in the [v0.1 method specification](method-specification.md) and must be committed before implementation and the first final-test run.

## Threshold Calibration

Each method receives its own threshold because the score scales are not assumed to be comparable.

Higher scores must consistently mean "more anomalous."

For `n` calibration scores from one method:

1. Score every normal calibration image.
2. Sort all preregistered finite scores, including fixed failure scores, by
   score ascending and relative path ascending for deterministic ties.
3. Set `rank = ceil(0.95 * n)`.
4. Choose `sorted_scores[rank - 1]` as the threshold.
5. Classify an image as anomalous when its score status is `failed` or its score is strictly greater than the threshold.
6. Record the calibration sample count, threshold, number classified as anomalous, and realized calibration false-positive rate.

The calibration primitive accepts one method identifier and one non-empty
mapping from relative paths to that method's concrete image-score records. ECC
residual and Patch HOG score record types cannot be mixed. The interface accepts
no class labels. It validates successful score bounds, exact fixed failure
scores, and score status before selecting a threshold.

Successful calibration output records the nearest rank, threshold, threshold
source path, full score order, failed-score paths, predicted-anomalous paths,
and realized normal false-positive rate. Invalid input produces no partial
threshold result and uses one of:

- `CALIBRATION_METHOD_INVALID`
- `CALIBRATION_EMPTY`
- `CALIBRATION_PATH_INVALID`
- `CALIBRATION_SCORE_TYPE_MISMATCH`
- `CALIBRATION_SCORE_RECORD_INVALID`
- `CALIBRATION_RESULT_INVALID`

The fixed-threshold classification primitive accepts one relative path, one
concrete method score, and one successful calibration result. It does not
accept a method override, threshold override, or observed class label. The
method and threshold come only from the validated calibration result.

A successful classification record contains the method, input score status,
source scoring failure code when present, threshold and its calibration source,
predicted class, decision reason, and signed `score - threshold` margin.
Invalid input produces no partial class decision and uses one of:

- `CLASSIFICATION_PATH_INVALID`
- `CLASSIFICATION_CALIBRATION_INVALID`
- `CLASSIFICATION_SCORE_TYPE_MISMATCH`
- `CLASSIFICATION_SCORE_RECORD_INVALID`
- `CLASSIFICATION_RESULT_INVALID`

The label-free batch classification primitive accepts one non-empty mapping
from relative paths to concrete score records and one successful calibration
result. It accepts no labels, method override, or threshold override. All paths
are validated before classification, the calibration state is validated once,
and paths are processed in Unicode code-point order. Every score must belong to
the calibration method and satisfy the same record contract as single-image
classification.

A successful batch contains one decision for every input path, in the
documented order, plus normal, anomalous, and scoring-failure counts and path
groups. A scoring failure remains an anomalous decision. If any item or
internally produced result is invalid, the batch returns no partial decision
collection. It records the failed path when available, the number of valid
decisions completed before that path, and the underlying classification
failure code. Batch failures use one of:

- `BATCH_CLASSIFICATION_EMPTY`
- `BATCH_CLASSIFICATION_PATH_INVALID`
- `BATCH_CLASSIFICATION_CALIBRATION_INVALID`
- `BATCH_CLASSIFICATION_ITEM_FAILED`
- `BATCH_CLASSIFICATION_RESULT_INVALID`

No anomaly image and no final-test image may influence this threshold.

The threshold remains fixed during final-test evaluation, even if the realized final-test normal false-positive rate exceeds 5%.

## Scoring Failure Policy

The fixed finite failure scores and method-level fitting-failure rules are defined in the [v0.1 method specification](method-specification.md).

- A failed image remains in calibration, ranking metrics, fixed-threshold metrics, failure counts, and latency evidence when a timing exists.
- A failed image is operationally classified as anomalous regardless of the numeric threshold.
- A method with `FIT_FAILED` status cannot pass and cannot be adopted.
- Failure handling must not be changed after calibration or final-test inspection.

## Test Leakage Prevention

The following controls are mandatory:

- Preserve the official train/test boundary.
- Verify no path overlap across reference, calibration, and test partitions.
- Do not inspect final-test anomaly labels while developing method parameters.
- Do not select a threshold from a test ROC curve.
- Do not choose a method from final-test AUROC or AUPRC and then rerun it with revised settings.
- Do not exclude difficult images, failed registrations, non-finite scores, or slow images from the primary result.
- Convert a score-generation failure to the preregistered finite failure score and retain its failed status.
- Commit method configuration and partition manifests before final-test scoring.
- Record the first valid final-test run used for the decision.

Anomaly labels and image-level class labels are revealed to the evaluation code only to compute the final metrics and failure analysis.

## Metrics

Anomaly is the positive class.

### Image-level AUROC

AUROC summarizes score ranking across all final-test normal and anomaly images. It is descriptive and threshold-independent. It is not an acceptance gate.

### Image-level AUPRC

AUPRC summarizes the precision-recall curve with anomaly as positive. It must be reported with final-test class counts because prevalence affects its interpretation. It is descriptive and is not an acceptance gate.

### Fixed-threshold normal false-positive rate

```text
normal FPR = false-positive normal images / final-test normal images
```

The prediction threshold comes only from normal calibration images.

### Fixed-threshold anomaly recall

```text
anomaly recall = detected anomaly images / final-test anomaly images
```

This uses the same calibration-only threshold as the final-test normal FPR.

### Failure counts

Report:

- Number of final-test normal images
- Number of final-test anomaly images
- True positives
- False negatives
- True negatives
- False positives
- Score-generation failures

No failed score may be silently removed from the denominator.

## CPU Measurement Method

The primary latency measurement covers per-image preprocessing and anomaly scoring after the method has been fitted and the image has been decoded into memory.

It excludes:

- Dataset download
- File decoding and disk I/O
- One-time reference fitting
- Report and figure writing

It includes:

- Resize or color conversion required by the fixed method
- ECC alignment for the residual method
- HOG extraction
- Feature scaling
- One-Class SVM scoring
- Residual or patch-score aggregation into the image score

Measurement requirements:

1. Use CPU execution only.
2. Record CPU model, logical and physical core counts, RAM, operating system, Python version, dependency versions, and relevant thread environment variables.
3. Use one warm-up pass over the fixed final-test sequence.
4. Run three timed passes over the same sequence in the same deterministic order.
5. Measure each image with a monotonic high-resolution timer.
6. Report the median and nearest-rank p95 over all valid per-image timings from the three timed passes.
7. Keep failed or exceptionally slow cases in the latency sample when a timing exists.
8. Do not run unrelated benchmark workloads concurrently.

The one-second p95 gate applies to this defined scoring boundary. End-to-end latency including file I/O may be reported separately but cannot replace it.

## Failure Case Selection

Failure cases are selected mechanically at the fixed threshold.

### Highest-scoring false positives

- Filter final-test normal images predicted as anomalous.
- Sort by anomaly score descending, then relative path ascending.
- Report up to five cases, or all cases when fewer than five exist.

### Lowest-scoring false negatives

- Filter final-test anomaly images predicted as normal.
- Sort by anomaly score ascending, then relative path ascending.
- Report up to five cases, or all cases when fewer than five exist.

For each selected case, record:

- Relative path
- True class
- Predicted class
- Score
- Threshold
- Score margin from threshold
- Registration status when applicable
- A bounded technical observation

Observations must describe visible or measured conditions without claiming an unverified cause.

If VisA images appear in a public failure figure, the caption or adjacent attribution must identify the VisA dataset, cite Zou et al., retain the Amazon copyright notice, link the official source and CC BY 4.0 license, and state all crops, resizing, overlays, or annotations.

## Acceptance Gates

A method passes only when every method-level and process-level gate passes.

| Gate | Pass condition |
| --- | --- |
| Final-test normal FPR | `<= 0.05` at the calibration-only threshold |
| Final-test anomaly recall | `>= 0.90` at the same threshold |
| CPU p95 scoring latency | `<= 1.0 second per image` |
| Normal reference budget | `<= 20 images` |
| Anomaly training labels | None used |
| Reproducibility | Same recorded assets and configuration reproduce the decision metrics within documented numeric tolerances |

The gates are hypothetical case-study assumptions. They are not production requirements or general recommendations.

## Decision Rules

Hard gates are applied before method comparison. No weighted aggregate score is allowed.

### ADOPT

Record `ADOPT` when at least one method passes every gate and the recorded failure cases do not contradict the stated hypothetical use conditions.

If both methods pass, the recommended method is selected lexicographically:

1. Higher fixed-threshold anomaly recall
2. Lower fixed-threshold normal FPR
3. Lower CPU p95 latency
4. Lower CPU median latency
5. ECC-aligned residual if all preceding values are equal, because it has the simpler fitted model

AUROC and AUPRC remain supporting evidence and do not override the hard-gate order.

### ADOPT WITH CONDITIONS

Record `ADOPT WITH CONDITIONS` only when at least one method passes every hard gate but the mechanically selected failure cases reveal a repeatable boundary that requires a narrower input condition, a manual-review step, or another explicit guardrail.

This decision may narrow intended use but may not waive a failed hard gate.

### REJECT

Record `REJECT` when:

- Neither method passes every hard gate, or
- Test leakage invalidates the evaluation, or
- Required assets or configuration cannot reproduce the decision evidence.

A rejection must still identify the first next validation that could reduce the observed uncertainty. DINOv2 may be proposed for v0.2, but a v0.1 rejection does not automatically authorize its implementation.

## Reproducibility Record

Before v0.1 completion, the repository must record:

- VisA access date and official source URL
- Official archive identifier and SHA-256
- Dataset license and attribution text
- Official split file identity and checksum
- Reference and calibration relative-path manifests
- Fixed seed and selection-procedure version
- Method configuration
- Exact Python and direct dependency versions
- The fully resolved dependency lock, distribution hashes, and license inventory
- Environment and CPU information
- Thresholds and calibration counts
- Per-image scores, labels, predictions, and latency samples
- Aggregate metrics
- A verification command and documented numeric tolerances

Raw VisA files remain external and are not committed.

## Known Limitations

- Only VisA `pcb1` is evaluated.
- One fixed reference set is used.
- The normal reference budget is not swept.
- The official split is treated as given rather than independently resampled.
- Image-level labels do not assess anomaly localization.
- No pixel-level metric is reported.
- No DINOv2 or other pretrained representation is compared.
- No confidence interval or statistical significance claim is planned.
- A CPU result is specific to the recorded environment and measurement boundary.
- A public benchmark cannot establish performance on a real inspection process.
- Passing v0.1 justifies only a next validation step, not deployment.
