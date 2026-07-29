# Few-Shot Anomaly PoC

Evaluate whether two CPU-only, normal-only visual anomaly detection methods justify a follow-up prototype for one VisA category.

> **Status: Milestone 9 Patch HOG image scoring primitive**
>
> The v0.1 problem, method shortlist, evaluation protocol, and decision gates
> are fixed. Reproducible data handling and deterministic shared image
> preprocessing are implemented, together with the bounded ECC registration
> primitive, deterministic normal-template fitting, and fixed residual image
> scoring. Fixed Patch HOG feature extraction and position-wise StandardScaler
> fitting, One-Class SVM fitting, and image scoring are also implemented. No
> calibrated threshold, dataset result, benchmark, method comparison, or
> decision is reported.

This is a source-available, noncommercially licensed public portfolio project.

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

## Planned Evaluation

The preregistered primary evidence is:

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
fitting and image scoring:

- [Problem and requirements](docs/problem-and-requirements.md)
- [Research and method selection](docs/research-and-method-selection.md)
- [v0.1 method specification](docs/method-specification.md)
- [Runtime dependencies and license boundaries](docs/dependencies-and-licenses.md)
- [Evaluation plan](docs/evaluation-plan.md)
- [VisA `pcb1` acquisition and partition record](docs/data-acquisition-record.md)
- [Data preparation and final-test boundary](data/README.md)

The runtime baseline is locked, while split pinning, safe acquisition, archive
provenance, safe extraction, deterministic manifests, integrity tests, linting,
CI, shared image preprocessing, ECC registration, ECC template fitting, and ECC
residual scoring are implemented, together with Patch HOG feature extraction.
Position-wise Patch HOG StandardScaler fitting is also implemented. The
position-wise One-Class SVM fitting and validation are implemented and tested
with synthetic reference features. Fixed Patch HOG image scoring is implemented
and tested with generated inputs. The repository still contains no VisA image,
calibrated threshold, evaluation pipeline, experiment, result figure, failure
analysis, or final decision.
