# Few-Shot Anomaly PoC

Evaluate whether two CPU-only, normal-only visual anomaly detection methods justify a follow-up prototype for one VisA category.

> **Status: Design and preregistration**
>
> The v0.1 problem, method shortlist, evaluation protocol, and decision gates are being fixed before implementation. No dataset has been downloaded, no algorithm has been implemented, and no result or decision is reported yet.

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

This method will align an input to a template built from the fixed normal references and derive an image-level anomaly score from the residual. It is the low-complexity, interpretable baseline.

### Patch HOG + One-Class SVM

This method will describe local appearance with Histogram of Oriented Gradients features and fit a one-class decision function using only the fixed normal references. It is the classical learned comparator.

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
- The official archive version, source URL, and checksum will be recorded after the authorized data-acquisition milestone.
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

The repository currently contains design documents only:

- [Problem and requirements](docs/problem-and-requirements.md)
- [Research and method selection](docs/research-and-method-selection.md)
- [Evaluation plan](docs/evaluation-plan.md)

Implementation, tests, data acquisition, experiments, result figures, failure analysis, and the final decision have not started.
