# Few-Shot Anomaly PoC

Evaluate whether two CPU-only, normal-only visual anomaly detection methods justify a follow-up prototype for one VisA category.

> **Status: v0.1 complete — `REJECT`**
>
> Both CPU-only, normal-only baselines failed the preregistered operating-point
> gates on VisA `pcb1`. The thresholds and gates were not changed after the
> result, and the negative outcome remains the project evidence.

This is a source-available, noncommercially licensed public portfolio project.

| Method | AUROC | Normal FPR | Anomaly recall | CPU p95 | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| ECC residual | `0.8141` | `0.09` | `0.21` | `1.2470 s` | `REJECT` |
| Patch HOG + One-Class SVM | `0.7838` | `0.10` | `0.19` | `0.5658 s` | `REJECT` |

Read the [public evaluation report](docs/v0.1-evaluation-report.md) for the
complete metrics, hard-gate rationale, limitations, and next validation. The
[completion review](docs/v0.1-completion-review.md) records reproducibility,
content, licensing, CI, and public-claim checks.

## Problem

This repository is a public, hypothetical case study. It does not describe a real company, customer, production system, or private dataset.

The hypothetical stakeholder has a small set of normal reference images and wants to decide whether a CPU-only visual anomaly detection prototype is worth a next round of validation. The primary objective is not maximum benchmark accuracy. It is to preserve a reviewable decision process:

```text
Requirements
    -> Research
    -> Longlist
    -> Shortlist
    -> Prototype
    -> Evaluation
    -> Failure Analysis
    -> Decision
```

The final v0.1 decision must be one of:

- `ADOPT`
- `ADOPT WITH CONDITIONS`
- `REJECT`

## Constraints

- Dataset scope: the `pcb1` category from the Visual Anomaly (VisA) dataset
- Normal reference budget: at most 20 images
- Compute: CPU only
- Methods: exactly two v0.1 prototypes
- Data separation: normal reference, threshold calibration, and final test partitions
- Label policy: anomaly labels are used only for final evaluation
- Evaluation scope: image-level metrics only
- No threshold or acceptance-gate changes after final-test results are inspected
- The same assets and configuration must reproduce the reported result

## Compared Methods

### ECC-aligned normal-template residual

This method aligns an input to a template built from the fixed normal references and derives an image-level anomaly score from the residual. It is the low-complexity, interpretable baseline.

### Patch HOG + One-Class SVM

This method describes local appearance with Histogram of Oriented Gradients features and fits position-specific one-class decision functions using only the fixed normal references. It is the classical learned comparator.

DINOv2 patch nearest-neighbor methods are not part of v0.1. They remain a v0.2 research candidate because their CPU cost, model-asset handling, and added implementation scope must be justified first.

## Evaluation Evidence

The preregistered primary evidence is reported without changing the gates:

- Image-level AUROC
- Image-level AUPRC
- Anomaly recall at a threshold calibrated only from normal calibration images
- Realized normal false-positive rate at that threshold
- CPU median per-image latency
- CPU p95 per-image latency
- False-positive and false-negative counts
- Highest-scoring false positives
- Lowest-scoring false negatives

AUROC and AUPRC are descriptive comparison metrics. They do not replace the hard acceptance gates.

## Decision Rule

A method passes v0.1 only if all of the following hold:

1. Final-test normal false-positive rate is no greater than 5%.
2. Final-test anomaly recall is at least 90% at the calibration-only threshold.
3. CPU p95 scoring latency is no greater than 1 second per image.
4. No more than 20 normal reference images are used.
5. No anomaly training label is used.
6. The reported result is reproducible from the same assets and configuration.

The gates are assumptions for this public case study, not real customer requirements or general deployment recommendations. They are fixed before results are produced.

The decision process applies the hard gates first and does not combine unrelated measurements into a weighted score. Detailed rules are preregistered in [the evaluation plan](docs/evaluation-plan.md).

## Dataset and License Status

VisA is distributed by Amazon Science through the official
[`amazon-science/spot-diff`](https://github.com/amazon-science/spot-diff)
repository and the
[AWS Registry of Open Data](https://registry.opendata.aws/visa/).
The dataset is released under the
[Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/).

CC BY 4.0 permits sharing and adaptation, including public display, when its conditions are followed. Required attribution includes appropriate creator and source credit, a license link, and an indication of modifications. Attribution must not imply endorsement.

Repository policy for v0.1 is stricter than the license permits:

- Raw VisA files will not be committed to Git.
- The official archive version, source URL, and observed checksum are fixed in [the acquisition record](docs/data-acquisition-record.md).
- The official one-class split is pinned to repository revision `2a692ab575001cbde74d402d897a7286086c6199` and SHA-256 `a48557e6033318cb90556f706196bc9d247a776a23ea51aecee5a80dd0332995`.
- The fixed 20-image reference and 884-image calibration partitions are recorded in [a metadata-only manifest](artifacts/v0.1/data/pcb1-normal-partitions.csv).
- Any future result figure containing a VisA image must identify the VisA dataset, cite Zou et al., retain the Amazon copyright notice, link the official source and CC BY 4.0 license, and describe crops, resizing, annotations, or other changes.
- No VisA image or derived result image exists in this repository at the current stage.

## License and permitted use

This repository is source-available under the PolyForm Noncommercial License 1.0.0.

You may use, study, modify, and experiment with the original code and documentation only for permitted noncommercial purposes under that license.

Commercial use is not permitted without a separate written license from the copyright holder.

Commercial use includes, but is not limited to:

- incorporation into a paid product or service
- use in commercial client deliverables
- deployment in a revenue-generating system
- internal business use intended to support commercial operations
- resale, sublicensing, or paid redistribution

For commercial licensing inquiries, contact the copyright holder.

See [`LICENSE`](LICENSE) for the controlling terms. This section is an informational summary and does not replace or override the license text. If this section conflicts with the PolyForm Noncommercial License 1.0.0, the terms in `LICENSE` control.

VisA and third-party dependencies are not licensed under PolyForm. Their separate licensing boundaries are recorded in [`NOTICE.md`](NOTICE.md).

## Reproducible environment

The v0.1 runtime uses CPython `3.13.14`. Exact runtime and development
distributions, source artifacts, and SHA-256 hashes are recorded in `uv.lock`;
the build-backend version is pinned in `pyproject.toml`.

With [`uv`](https://docs.astral.sh/uv/) installed:

```bash
uv sync --locked --no-dev
uv run --locked --no-sync python scripts/verify_environment.py
```

The dependency-only smoke test checks exact versions and the required OpenCV ECC, scikit-image HOG, StandardScaler, and One-Class SVM API paths using synthetic arrays. Passing it is not an algorithm result, dataset result, performance result, or acceptance-gate result. Package and bundled-binary license boundaries are documented in [the dependency inventory](docs/dependencies-and-licenses.md).

## Reproducible data foundation

Milestone 1 adds a small Python package and the `few-shot-data` command. Its data
path uses only the Python standard library; the existing numerical dependencies
remain reserved for later preregistered algorithm work.

Run the workflow inside Ubuntu 24.04 on WSL:

```bash
uv sync --locked
uv run --locked --no-sync few-shot-data fetch-split
uv run --locked --no-sync few-shot-data download-archive
uv run --locked --no-sync few-shot-data extract-archive
uv run --locked --no-sync few-shot-data build-manifests
uv run --locked --no-sync few-shot-data validate-manifests
```

The archive downloader streams to a temporary file and records provenance. The
tar extractor validates every archive member, rejects unsafe paths and
non-regular content before writing, and extracts only the `pcb1/` subtree.
Manifest generation applies the fixed SHA-256 path ranking from the evaluation
plan and verifies that reference, calibration, and final-test paths do not
overlap.

Final-test handling is metadata-only in this milestone. The generator reads the
pinned official CSV but receives no dataset root, and final-test records omit
class labels and pixel-mask paths. Neither command opens, displays, scores, or
summarizes image content. See [the data preparation guide](data/README.md) for
the exact boundary and provenance fields.

## Local VisA `pcb1` integrity verification

The fixed local asset has passed a deterministic integrity check before any
normal-reference fitting or final-test scoring. The verifier:

- matches the 1,929,840,640-byte archive to the previously recorded observed
  SHA-256, without presenting that value as an independently published upstream
  checksum
- validates all 12,122 archive members with the safe-extraction rules
- compares all 1,205 selected `pcb1` files byte-for-byte against their archive
  members
- requires the extracted image and mask path sets to match the pinned official
  split exactly
- checks the fixed counts and `image_anno.csv` identity
- emits no image, per-path final-test label, anomaly score, metric, or threshold
  change

The aggregate record is
[`artifacts/v0.1/data/pcb1-local-integrity.json`](artifacts/v0.1/data/pcb1-local-integrity.json).
It contains no raw VisA bytes or machine-specific absolute path. The verifier,
scope, and regeneration commands are documented in
[the local integrity record](docs/local-data-integrity.md).

## Normal-reference fitting and calibration

The fixed runner was committed and passed CI before it was executed. Source
commit `4fef91c1d1e339aa507cad80d51127e01046ae0b` then:

- revalidated the pre-evaluation freeze and current local dataset identity
- fitted the ECC template with 20 of 20 successful references
- fitted all 225 Patch HOG scalers and all 225 One-Class SVMs
- scored exactly the 884 normal calibration images with both methods
- fixed each method's nearest-rank 95th-percentile threshold at rank 840
- produced zero calibration score failures for either method
- preserved hash-checked fitted state outside Git for the first final-test run

| Method | Fixed threshold | Normal calibration scores above threshold |
| --- | ---: | ---: |
| ECC residual | `0.688464437424507` | `44 / 884` |
| Patch HOG + One-Class SVM | `0.17611826509314352` | `44 / 884` |

The `44 / 884` rate is the mechanical result of the fixed normal-only
calibration rule. It is not the final-test normal FPR and is not evidence of
anomaly recall or overall method quality.

The public checkpoint and all label-free calibration scores are in
[`artifacts/v0.1/calibration/normal-only/`](artifacts/v0.1/calibration/normal-only/).
The procedure, fitted-state boundary, exact results, and regeneration command
are documented in
[the normal-only calibration record](docs/normal-only-calibration.md).

## First fixed final-test scoring

The runner was committed and passed CI before its single fixed run. Source
commit `5b142f31c974334545ca2bb63bb7b2c6c514828a`:

- accepted only a hash-bound final-test manifest with no class or mask fields
- decoded and scored exactly 200 final-test assets with both fitted methods
- applied the already fixed per-method thresholds
- ran the fixed CPU warm-up and three-pass timing protocol
- preserved score, classification, and latency records for final evaluation
- produced zero score failures for either method

| Method | Predicted anomalous | CPU median | CPU p95 |
| --- | ---: | ---: | ---: |
| ECC residual | `30 / 200` | `0.4369505 s` | `1.24699559 s` |
| Patch HOG + One-Class SVM | `29 / 200` | `0.4326800675 s` | `0.565766242 s` |

The ECC p95 exceeds the preregistered one-second latency ceiling. This is not
yet the complete hard-gate decision because final-test classes and image-level
metrics remain outside this scoring stage.

The public score, classification, and latency artifacts are in
[`artifacts/v0.1/scoring/first-fixed-final-test/`](artifacts/v0.1/scoring/first-fixed-final-test/).
The input boundary, artifact layout, local-state policy, and recorded command
are documented in
[the first fixed final-test scoring record](docs/first-fixed-final-test-scoring.md).

## Final evaluation and decision

The evaluator was committed and passed CI before class reveal. Source commit
`c6b4e5e164cc8788ff0428361406ada3e116543b`:

- verified the complete freeze, calibration, scoring, manifest, and split lineage
- revealed official per-path final-test classes for the first time
- calculated image-level AUROC, AUPRC, normal FPR, anomaly recall, FP, and FN
- selected the fixed high-confidence false positives and low-confidence false
  negatives
- applied all six hard gates in their preregistered order
- wrote the non-overwritable final evaluation bundle

| Method | AUROC | AUPRC | Normal FPR | Anomaly recall | CPU p95 | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ECC residual | `0.8141` | `0.7513` | `0.09` | `0.21` | `1.2470 s` | `REJECT` |
| Patch HOG + One-Class SVM | `0.7838` | `0.7242` | `0.10` | `0.19` | `0.5658 s` | `REJECT` |

Both methods failed the fixed normal-FPR and anomaly-recall gates. ECC also
failed the CPU p95 gate. Ranking metrics do not override those failures.

The failure-review disposition is fixed as `guardrail_required` before class
reveal because the mechanically selected cases do not yet have an
image-content review. It did not waive the failed hard gates. The exact
boundary, metrics, gate outcomes, CLI-summary defect record, and artifacts are
documented in
[the final evaluation and decision record](docs/final-evaluation-and-decision.md).
The complete machine-readable bundle is in
[`artifacts/v0.1/evaluation/visa-pcb1-v0-1-final/`](artifacts/v0.1/evaluation/visa-pcb1-v0-1-final/).
The [v0.1 public evaluation report](docs/v0.1-evaluation-report.md) explains
the decision, claim boundaries, known limitations, and next validation without
reproducing VisA images.

## Shared input preprocessing

Both shortlisted methods use one fixed implementation that:

- decodes an image as 8-bit grayscale while ignoring orientation metadata
- rejects missing, unreadable, empty, undecodable, or structurally invalid input
- resizes directly to `512 x 512` with area interpolation
- converts to `float32` and divides by `255.0`
- rejects an unexpected shape, dtype, or non-finite output

Failures carry stable codes for later score records. The implementation does
not retry, repair, normalize, crop, preserve aspect ratio, or convert a failure
into a method score. Tests use generated arrays and temporary synthetic image
files; no VisA image is read by the test suite.

## ECC registration primitive

The first method now has a bounded registration component that:

- starts from a `2 x 3` identity warp
- uses OpenCV ECC with the Euclidean rotation-and-translation model
- fixes the iteration, epsilon, Gaussian-filter, interpolation, and border rules
- rejects non-convergence and invalid or non-finite ECC results
- enforces the preregistered `10 degree` and `64 pixel` plausibility limits
- warps a binary validity mask and requires at least `80%` valid area
- returns structured diagnostics and stable failure codes

The registration primitive uses inverse-map warping to bring the moving image
into template coordinates. By itself, it does not calculate a residual, produce
an anomaly score, or use a VisA image.

## ECC normal-template fitting

The ECC method now has a deterministic fitting component that:

- requires exactly the fixed 20 normal references
- sorts reference paths and uses the first path as the identity anchor
- registers the remaining references with the fixed ECC primitive
- records every accepted and rejected reference with bounded diagnostics
- requires at least 16 successfully aligned references, including the anchor
- intersects their validity masks and applies the fixed `5 x 5` erosion
- requires the eroded support to cover at least `75%` of the resized image
- computes a full-size pixel-wise median from valid aligned values

The output retains the fitted template, residual-scoring support mask, support
fraction, reference counts, and stable method-level failure status. Tests use
only generated arrays. This component does not use calibration images, score an
input image, select a threshold, or read a VisA image.

## ECC residual image scoring

The first shortlisted method now has a fixed image-level scoring component
that:

- requires a successful fitted template state instead of converting
  method-level `FIT_FAILED` into an image score
- registers one preprocessed image with the bounded ECC primitive
- erodes the warped validity mask once with a constant-zero `5 x 5` kernel
- intersects it with the fitted template support
- requires at least `95%` of template-support pixels to remain effective
- applies a constant-zero `5 x 5` Gaussian filter with automatic sigma to the
  absolute grayscale residual
- averages the largest `max(1, ceil(0.01 * N))` effective residual values
- preserves registration diagnostics, support counts, score status, and stable
  failure codes

A valid score is in `[0, 1]`, with higher values indicating greater anomaly
evidence. An image-level failure receives the preregistered finite score `1.0`
and remains distinguishable through `score_status=failed` and its failure code.
The implementation does not calibrate a threshold, classify an image, measure
latency, read a VisA image, or make a performance claim.

## Patch HOG feature extraction

The second shortlisted method now has a deterministic feature component that:

- requires the same validated `512 x 512` preprocessed grayscale input
- extracts `64 x 64` patches with stride 32 at the fixed `15 x 15` grid
- preserves row-major positions from `(0, 0)` through `(448, 448)`
- calls scikit-image HOG with the preregistered orientation, cell, block,
  normalization, square-root transform, and channel settings
- returns one complete `float32` matrix with shape `(225, 324)`
- rejects wrong-shaped, wrong-dtype, or non-finite descriptors
- records the failed patch index and returns no partial matrix after a failure

Tests use generated arrays and verify the complete grid, fixed HOG call,
repeatability, and input immutability. This component does not fit or apply a
`StandardScaler`, fit a One-Class SVM, produce an anomaly score, read a VisA
image, or claim method performance.

## Position-wise Patch HOG scaling

The second method now has a deterministic reference-only scaling component
that:

- requires exactly 20 complete Patch HOG reference results
- sorts reference paths in ascending Unicode code-point order
- rejects incomplete, mismatched, wrong-shaped, wrong-dtype, or non-finite
  reference features before fitting
- fits one `StandardScaler` per row-major patch position from its `(20, 324)`
  reference sample matrix
- uses the fixed `copy=True`, `with_mean=True`, and `with_std=True` settings
- validates every fitted mean, variance, scale, feature count, and sample count
- accepts constant feature dimensions only when their fitted variance is zero
  and their scaler-provided scale is one
- preserves the failed reference or position and returns no partial scaler
  collection after a failure

Tests use generated feature matrices and one generated-image HOG extraction.
This component does not use calibration features, transform scoring images, fit
a One-Class SVM, produce an anomaly score, read a VisA image, or claim method
performance.

## Position-wise Patch HOG One-Class SVM fitting

The second method now has a deterministic reference-only model component that:

- requires the same 20 reference paths used by the validated scaler collection
- transforms each position-specific `(20, 324)` sample matrix without updating
  its scaler
- validates every transformed matrix before model fitting
- fits 225 independent RBF One-Class SVMs with the fixed `gamma`, `nu`,
  tolerance, shrinking, cache, iteration, and verbosity settings
- requires successful solver status, the fixed fit shape and feature count, a
  finite positive fitted gamma, and finite support-vector, coefficient,
  intercept, and offset state
- preserves feature, scaler, transform, solver, and fitted-state failure
  boundaries
- records the failed reference or position and returns no partial model
  collection after a failure

Tests use generated feature matrices and one generated-image HOG extraction.
This component does not use calibration features or anomaly labels, calculate
decision-function values for scoring images, aggregate patch scores, read a
VisA image, or claim method performance.

## Patch HOG image scoring

The second method now has a deterministic single-image scoring component that:

- requires complete, validated scaler and model collections fitted from the
  same 20 normal reference paths
- extracts the fixed 225 descriptors and applies only the scaler and model for
  each corresponding patch position
- negates each One-Class SVM decision value so that a higher patch score means
  more anomalous
- rejects non-finite patch scores and scores whose absolute value is at least
  `1e12`
- averages the 12 highest patch scores, with patch index ascending as the
  deterministic tie-breaker
- returns all 225 patch scores and the 12 contributing patch indices after a
  successful score
- converts image-level feature, transform, decision, and aggregation failures
  to the fixed `1e12` failure score without returning partial patch scores

The contributing patch indices are diagnostic evidence, not a pixel-level
localization result. Tests use generated images and generated normal reference
features. This component does not select a threshold, use anomaly labels, read
a VisA image, or claim method performance.

## Normal-only threshold calibration

The evaluation foundation now includes a method-specific calibration primitive
that:

- accepts a non-empty mapping from relative calibration paths to concrete score
  results from exactly one shortlisted method
- exposes no class-label input and therefore cannot consume anomaly labels
- validates each method's successful score range and fixed failure score
- retains failed score records and always predicts them as anomalous
- sorts by anomaly score ascending and relative path ascending for deterministic
  ties
- selects `rank = ceil(0.95 * n)` and
  `threshold = sorted_scores[rank - 1]`
- classifies a successful score as anomalous only when it is strictly greater
  than the threshold
- records the threshold source, ordered paths, failure paths, predicted
  anomalous paths, and realized calibration false-positive rate

Tests use generated score records only, including the fixed v0.1 calibration
sample count of 884. This component does not load the calibration manifest,
score dataset images, inspect final-test data, select a method, apply acceptance
gates, or report a dataset-derived threshold.

## Fixed-threshold image classification

The evaluation foundation now includes a single-score classification primitive
that:

- requires a complete, internally consistent successful calibration result
- inherits the method and threshold from that result instead of accepting a
  second method or threshold argument
- rejects an ECC score paired with a Patch HOG calibration result, and vice
  versa
- exposes no observed class-label input
- classifies `score_status=failed` as `anomalous` regardless of the numeric
  comparison
- classifies a successful score as `anomalous` only when
  `score > threshold`; equality remains `normal`
- records the source path, method, score status, source failure code, threshold
  provenance, predicted class, decision reason, and signed score margin
- produces no partial decision after an invalid path, calibration state, score
  type, or score record

Tests use generated calibration and score records only. This component does not
load a partition, score an image, accept a ground-truth label, calculate an
evaluation metric, inspect final-test data, or report a dataset prediction.

## Label-free batch classification

The fixed-threshold boundary now also accepts a non-empty mapping from relative
paths to score records from one shortlisted method and:

- exposes no observed class-label input
- validates every relative path before producing a decision
- validates the shared calibration result once for the complete batch
- processes paths in Unicode code-point order for deterministic output
- applies the same method, score-record, failure-score, and strict-threshold
  rules as the single-image primitive
- returns the complete ordered decision tuple only when every input produces a
  valid decision
- returns no partial decision collection if an item or internally produced
  result is invalid
- records normal, anomalous, and scoring-failure counts and paths after a
  successful batch
- records the failed path, completed-item count, and underlying classification
  failure code when a batch fails after processing starts

Tests use generated calibration and score records only. This component does not
load a manifest or image, reveal a ground-truth label, compute a metric, inspect
final-test data, or report a dataset prediction.

## Final-test label reveal boundary

The evaluation foundation now includes a one-way boundary that pairs labels
with an already completed label-free classification batch. It:

- accepts only a successful, internally consistent batch, an ordered label
  sequence, and the fixed project configuration
- permits only `normal` and `anomaly` final-test labels
- validates batch paths, threshold provenance, per-image decisions, failure
  handling, and batch summaries before revealing labels
- requires exactly one label for every classified path
- rejects malformed label records, duplicate paths, missing paths, extra paths,
  and order mismatches with stable failure codes
- reports the first order mismatch and complete missing and extra path
  diagnostics without returning partial labeled records
- retains each original classification object unchanged inside the paired
  record

Tests use generated batches and labels only. This boundary does not read the
official split, load a VisA image, calculate AUROC, AUPRC, false-positive or
false-negative counts, apply an acceptance gate, or make a decision.

## Image-level metrics

The evaluation foundation now calculates the preregistered metrics from a
complete, validated label reveal result:

- anomaly is fixed as the positive class
- the existing image anomaly score is used for ranking without transformation
- image-level AUROC uses scikit-learn `roc_auc_score`
- image-level AUPRC uses scikit-learn `average_precision_score`
- TP, FN, TN, and FP counts use the existing fixed-threshold decisions
- normal FPR is `FP / final-test normal count`
- anomaly recall is `TP / final-test anomaly count`
- score-generation failures remain in ranking metrics, fixed-threshold counts,
  and denominators
- an absent normal or anomaly class, invalid revealed input, expected metric
  library failure, or invalid output produces no partial metric result

Tests use generated labeled records with known expected measurements. This
component does not read final-test files, choose or change a threshold, measure
latency, apply hard gates, select failure examples, compare methods, or make an
adoption decision.

## CPU latency measurement

The evaluation foundation now implements the fixed preprocessing-and-scoring
latency protocol:

- starts from a decoded grayscale `uint8` image and times the callback that
  performs fixed preprocessing and method scoring
- processes paths in Unicode code-point order
- performs one complete untimed warm-up pass
- performs three complete timed passes with `time.perf_counter_ns`
- retains one timing for every image in every timed pass
- includes valid `score_status=failed` invocations in the latency sample and
  records their paths and source failure codes
- reports the standard median and nearest-rank p95 over all timed observations
- records the CPU model, logical and physical cores, RAM, OS, architecture,
  Python version, dependency versions, OpenCV thread count, and relevant thread
  environment variables
- rejects invalid paths, method-score mismatches, scorer exceptions, malformed
  scores, invalid timers, and invalid summaries without returning partial
  observations

The protocol is pinned in `configs/v0.1.yaml`. Tests use generated inputs,
generated method score records, and deterministic synthetic timers. No VisA
image is read, and no measured result in this milestone is project performance
evidence. This component does not apply the one-second hard gate or make an
adoption decision.

## Failure case selection

The evaluation foundation now implements the preregistered mechanical
failure-case selection rule:

- accepts only a complete, valid final-test label reveal result
- selects false positives by anomaly score descending, then relative path
  ascending
- selects false negatives by anomaly score ascending, then relative path
  ascending
- returns at most five cases of each type, as fixed in `configs/v0.1.yaml`
- retains score-generation failures and their source failure codes
- records the fixed-threshold score, threshold, signed margin, true class,
  predicted class, and deterministic rank for each selected case
- rejects invalid revealed input or an invalid produced selection without
  returning a partial selection

This primitive does not open or display images, add subjective observations,
recalculate metrics, apply hard gates, or make an adoption decision. Any later
technical observation or public image must remain a separate, attributed
reporting step and cannot alter this selection.

## Hard-gate decision

The evaluation foundation now implements the fixed per-method decision rule:

- evaluates final-test normal FPR, anomaly recall, CPU p95 scoring latency,
  normal reference count, anomaly-training-label use, and reproducibility in
  the preregistered order
- records each observed value, required value, comparison operator, and
  pass/fail outcome without creating a weighted score
- rejects a method when any hard gate fails
- rejects an otherwise passing evaluation when test leakage is recorded
- allows `ADOPT WITH CONDITIONS` only when every hard gate passes and the
  explicit failure review requires a documented guardrail
- does not allow a failure-review condition to waive a failed hard gate
- returns `REJECT` when an otherwise passing failure review contradicts the
  intended use

The limits, order, no-weighted-score rule, and no-waiver rule are fixed in
`configs/v0.1.yaml`. Invalid metric, latency, process, or cross-method evidence
produces no partial decision. This primitive evaluates one method at a time;
the preregistered cross-method recommendation rule remains part of the later
final evaluation. Tests use synthetic records only, so no project adoption
decision exists yet.

## Evaluation artifact schema

The versioned `evaluation-artifacts/v0.1` contract defines the reproducible
output boundary before final-test scoring:

- label-free score and fixed-threshold classification CSV files
- a separate post-boundary revealed-label CSV
- image-level metrics JSON
- CPU latency summary JSON and every timed observation in CSV
- mechanically selected failure-case CSV
- ordered hard-gate decision JSON
- one bundle manifest with Git/config/partition provenance, record counts, and
  SHA-256 file identities

The contract fixes headers, required JSON keys and types, primary keys, sort
orders, null encoding, finite-number handling, UTF-8/LF serialization, and
non-overwrite behavior. It forbids timestamps, absolute paths, raw dataset
content, duplicate keys, and non-finite numbers. See the
[artifact schema guide](docs/evaluation-artifact-schema.md) and its
[machine-readable descriptor](schemas/v0.1/evaluation-artifacts.json).

The schema milestone created no empty result files. The only committed
evaluation bundle at the current stage is the explicitly synthetic fixture
described below; no VisA-derived bundle exists.

## Synthetic end-to-end evaluation

The synthetic integration path connects both method-specific score types
through:

```text
normal-only calibration scores
    -> fixed threshold
    -> label-free final scores
    -> fixed-threshold batch classification
    -> exact-path label reveal
    -> image-level metrics
    -> mechanical failure selection
    -> fixed hard-gate decision
    -> immutable JSON/CSV bundle
```

All inputs are generated records. No image is opened, no VisA path or label is
used, and the latency records describe a synthetic timer fixture rather than a
measurement. Each method intentionally produces one false positive and two
false negatives, exercising the exact 5% normal-FPR and 90% anomaly-recall gate
boundaries. A synthetic `ADOPT` proves only that the preregistered plumbing can
produce that label from passing inputs.

The generator validates every intermediate result by recomputing the chain,
writes through a temporary directory, refuses overwrite, records every file
digest, and has byte-for-byte reproduction tests. The
[committed synthetic bundle](artifacts/v0.1/evaluation/synthetic-e2e) was
generated from source commit
`7193a89e0cff8d543c0f7274e834d902026752d5`. Its design and claim boundary are
documented in the [synthetic evaluation record](docs/synthetic-evaluation.md).

To generate the same bundle under a separate output root:

```bash
uv run --locked --no-sync python scripts/run_synthetic_evaluation.py \
  --output-root /tmp/few-shot-anomaly-poc-synthetic-reproduction \
  --source-commit 7193a89e0cff8d543c0f7274e834d902026752d5
```

This command is not a VisA experiment and cannot support a method-performance
claim.

## Pre-evaluation freeze

The machine-readable
[pre-evaluation freeze record](artifacts/v0.1/freeze/pre-evaluation-freeze.json)
fixes the v0.1 evaluation source before final-test use:

- evaluation source commit
  `fd9857acb29903fadb570680ecb5d4d8ebf5a5aa`
- successful
  [GitHub Actions run #21](https://github.com/cab0a/few-shot-anomaly-poc/actions/runs/30434900673)
- config, dependency lock, CI, license boundary, data records, method
  specification, evaluation plan, artifact schema, and evaluation code hashes
- seed `42`, the exact 20 reference IDs, and 884 normal calibration IDs
- official split revision and checksum
- threshold, latency, failure-selection, and six ordered hard-gate rules
- a boundary state confirming that final-test scoring, label join, metrics, and
  decision have not started

The aggregate frozen-tree SHA-256 is
`cf9460eb919025417c988771926e00d06641ea63b242c397be466dd7823970f9`.
Tests recompute every frozen file identity. A frozen-file change invalidates
the checkpoint and cannot be justified by a later result. See the
[freeze rationale and change policy](docs/pre-evaluation-freeze.md).

## Non-goals

v0.1 does not attempt to provide:

- A production inspection system
- A claim about performance on real customer data
- Pixel-level anomaly localization or segmentation evaluation
- DINOv2, PatchCore, AnomalyDINO, or another deep model implementation
- Model training or fine-tuning
- GPU benchmarking
- Multiple VisA categories
- A web interface, API, database, Docker image, or cloud deployment
- Automated threshold tuning from anomaly or final-test labels
- A state-of-the-art benchmark claim

## Current Project Stage

The repository contains preregistered design documents, the data foundation,
deterministic shared preprocessing, bounded ECC registration, deterministic ECC
normal-template fitting, fixed ECC residual image scoring, and fixed Patch HOG
feature extraction with position-wise reference scaling and One-Class SVM
fitting and image scoring, plus normal-only threshold calibration and
fixed-threshold single-image and batch classification, followed by the
final-test label reveal boundary, image-level metrics, CPU latency measurement,
mechanical failure-case selection, and the per-method hard-gate decision:

- [Problem and requirements](docs/problem-and-requirements.md)
- [Research and method selection](docs/research-and-method-selection.md)
- [v0.1 method specification](docs/method-specification.md)
- [Runtime dependencies and license boundaries](docs/dependencies-and-licenses.md)
- [Evaluation plan](docs/evaluation-plan.md)
- [Evaluation artifact schema](docs/evaluation-artifact-schema.md)
- [Synthetic end-to-end evaluation record](docs/synthetic-evaluation.md)
- [Pre-evaluation freeze record](docs/pre-evaluation-freeze.md)
- [VisA `pcb1` acquisition and partition record](docs/data-acquisition-record.md)
- [v0.1 public evaluation report](docs/v0.1-evaluation-report.md)
- [Final evaluation and decision record](docs/final-evaluation-and-decision.md)
- [v0.1 completion review](docs/v0.1-completion-review.md)
- [Data preparation and final-test boundary](data/README.md)

The runtime baseline is locked, while split pinning, safe acquisition, archive
provenance, safe extraction, deterministic manifests, integrity tests, linting,
CI, shared image preprocessing, ECC registration, ECC template fitting, and ECC
residual scoring are implemented, together with Patch HOG feature extraction.
Position-wise Patch HOG StandardScaler fitting is also implemented. The
position-wise One-Class SVM fitting and validation are implemented and tested
with synthetic reference features. Fixed Patch HOG image scoring is implemented
and tested with generated inputs. The normal-only threshold calibration rule is
implemented and tested with generated score records. Fixed-threshold
single-image and batch classification is also implemented and tested without
observed labels. The final-test label reveal boundary is implemented and tested
with generated inputs. Image-level AUROC, AUPRC, fixed-threshold confusion
counts, normal FPR, anomaly recall, and score-failure counting are implemented
and tested with generated records. The fixed CPU warm-up, three-pass timing,
environment capture, median, and nearest-rank p95 protocol is implemented and
tested with generated inputs and synthetic clocks. The fixed highest-score
false-positive and lowest-score false-negative selection rule is implemented
and tested with generated labeled records without image access. The repository
also applies every preregistered hard gate in fixed order and tests all three
allowed decision labels with synthetic evidence. A versioned, deterministic
JSON/CSV contract now covers scores, classifications, revealed labels, metrics,
latency, selected failures, decisions, and artifact provenance. A
synthetic-record generator connects the full primitive chain for both methods
and verifies deterministic, non-overwritable bundle generation in temporary
test directories. The byte-reproducible `synthetic-e2e` bundle is committed and
explicitly marked as plumbing evidence. The evaluation source, reference IDs,
normal calibration partition, dependencies, evaluation rules, and artifact
contract were frozen with file-level hashes and successful CI evidence before
final-test scoring. The normal-only calibration, first fixed label-free
final-test scoring, one-time class reveal, metrics, mechanical failure
selection, and ordered hard-gate decision are complete. Both methods are
`REJECT`, and the negative result is documented in the public evaluation
report. The repository contains no raw VisA image, mask, result figure, or
image-based causal failure analysis.
