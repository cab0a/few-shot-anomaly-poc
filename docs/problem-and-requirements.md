# Problem and Requirements

## Status

This document preregisters the v0.1 case assumptions and requirements before implementation or dataset access. It describes a hypothetical public portfolio case study, not a real engagement.

No company, customer, production line, private dataset, internal metric, or confidential requirement is represented here.

## Hypothetical Stakeholder

The stakeholder is a small computer-vision engineering team deciding whether to fund a second anomaly-detection validation phase.

The team can provide a limited set of normal examples for one visual category but cannot rely on representative anomalous training data. It has access to an ordinary CPU environment and wants evidence that is understandable without a large machine-learning platform.

The stakeholder needs a bounded feasibility decision, not a production delivery.

## Business Problem

Manual inspection effort may justify further automation research, but committing to a larger implementation before testing technical feasibility would create unnecessary cost.

The v0.1 question is:

> Can either of two CPU-only methods, trained or configured from no more than 20 normal references, detect anomalies in the VisA `pcb1` case strongly enough to justify a follow-up prototype?

The answer may be negative. A reproducible rejection with documented failure conditions is an acceptable project outcome.

## Assumptions

- VisA `pcb1` is a public proxy for a visual-inspection problem; it is not customer data.
- The camera and object layout are sufficiently stable for image registration to be a plausible baseline.
- Only normal examples are available for fitting the v0.1 methods.
- Anomaly labels and image-level test labels are available to the evaluator only after method configuration and threshold calibration are fixed.
- Twenty normal reference images are the maximum fitting budget.
- Remaining eligible normal training images may be used only to calibrate an operating threshold.
- The official one-class test partition is held out for final evaluation.
- Scoring occurs on a CPU in an offline or low-throughput setting.
- A one-second p95 latency gate is a case-study assumption, not a general real-time requirement.
- Image-level detection is sufficient for the v0.1 feasibility decision.
- One fixed seed is sufficient for v0.1, while sample-sensitivity analysis remains later work.

## Mandatory Requirements

### Scope

- Use only the VisA `pcb1` category.
- Compare only:
  - ECC-aligned normal-template residual
  - Patch HOG + One-Class SVM
- Use no more than 20 normal reference images.
- Use CPU execution only for the reported latency result.
- Keep raw VisA data outside Git.

### Data and leakage control

- Separate normal reference, normal threshold-calibration, and final-test partitions.
- Select reference IDs with the preregistered seed and deterministic procedure.
- Commit the selected reference-ID manifest before inspecting final-test results.
- Use anomaly labels only for final evaluation.
- Do not use final-test normal or anomaly labels to select features, parameters, thresholds, or methods.
- Do not change the acceptance gates after inspecting final-test results.

### Evaluation

- Report image-level AUROC.
- Report image-level AUPRC with anomaly as the positive class.
- Report final-test anomaly recall and normal false-positive rate at the calibration-only threshold.
- Report CPU median and p95 per-image scoring latency.
- Report false-positive and false-negative counts at the fixed threshold.
- Review the highest-scoring false positives.
- Review the lowest-scoring false negatives.

### Decision

- Apply hard gates before qualitative comparison.
- Do not create a weighted aggregate score.
- Record exactly one final decision:
  - `ADOPT`
  - `ADOPT WITH CONDITIONS`
  - `REJECT`
- Preserve unfavorable results and method-specific failure conditions.

### Reproducibility

- Record the official data source, archive identifier, archive checksum, split source, and selected relative paths.
- Record the fixed seed and all method configuration.
- Record Python and dependency versions, operating system, CPU model, and thread settings.
- Provide one documented reproduction workflow before v0.1 is declared complete.

## Optional Requirements

These may be added only if they do not expand the v0.1 method or data scope:

- A compact machine-readable summary of gates and pass/fail outcomes
- A small attributed failure-case figure after the evaluation milestone
- Descriptive fitting time in addition to required scoring latency
- Peak memory as non-gating diagnostic evidence
- A recommendation for the first v0.2 experiment

Optional requirements must not delay or redefine the mandatory decision.

## Non-goals

- Production readiness
- Safety-critical or unattended rejection decisions
- Real customer validation
- Generalization beyond VisA `pcb1`
- Pixel-level metrics or segmentation claims
- Defect-type classification
- DINOv2 implementation
- PatchCore or AnomalyDINO reproduction
- Deep-model training or fine-tuning
- GPU evaluation
- Hyperparameter optimization against anomaly or final-test labels
- Multi-category benchmarking
- Confidence intervals or multi-seed robustness claims
- Web UI, API, service, database, Docker, or cloud infrastructure
- A reusable anomaly-detection framework
- State-of-the-art accuracy claims

## Acceptance Gates

The following gates are fixed before results:

| Gate | Required outcome |
| --- | --- |
| Normal false-positive rate | `<= 5%` on the final-test normal images at the calibration-only threshold |
| Anomaly recall | `>= 90%` on the final-test anomaly images at the same threshold |
| CPU p95 scoring latency | `<= 1.0 second per image` |
| Normal reference count | `<= 20` |
| Anomaly training labels | None used |
| Reproducibility | Same assets and configuration reproduce the reported decision evidence |

AUROC, AUPRC, median latency, and failure examples remain required evidence but are not additional hard gates.

## Risks

### External-validity risk

One public category cannot establish performance for another product, camera, lighting condition, or defect distribution.

### Split and sampling risk

One fixed reference set may be unusually favorable or unfavorable. v0.1 records this limitation instead of making a multi-seed robustness claim.

### Registration risk

ECC alignment may fail or converge to a plausible but incorrect transformation. Registration failure must remain visible rather than being silently discarded.

### Feature-model risk

HOG may miss low-contrast or non-edge anomalies, while a One-Class SVM may be unstable with a small and high-dimensional reference set.

### Threshold risk

A threshold satisfying the empirical calibration false-positive target may not preserve that rate on held-out normal images. The final-test false-positive gate is therefore evaluated separately.

### Measurement risk

CPU latency depends on hardware, thread settings, image decoding policy, image size, and warm-up. The measurement boundary must be recorded with the result.

### License and provenance risk

CC BY 4.0 permits reuse but requires attribution and change disclosure. Dataset-derived figures can become noncompliant if source, authors, license, or modifications are omitted.

### Decision risk

A passing result can justify only the next validation step. It cannot establish production readiness.
