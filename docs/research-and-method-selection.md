# Research and Method Selection

## Purpose

This document records the v0.1 method longlist, shortlist, and deferral decisions before implementation and evaluation.

The goal is not to reproduce every anomaly-detection family. The goal is to select the smallest comparison that can test whether a low-complexity baseline or a classical one-class model merits a follow-up CPU prototype.

## Selection Criteria

Methods were reviewed against the following v0.1 constraints:

- Learn or configure from no more than 20 normal reference images
- Require no anomalous training labels
- Run on a general CPU
- Produce an image-level anomaly score
- Fit a short implementation milestone
- Support a clear failure analysis
- Use dependencies and assets with reviewable license terms

Accuracy reported by prior work is not treated as evidence that a method will pass this case study.

## Longlist

### Registered pixel difference

**Basic idea:** Align an input image to a normal reference or normal template, calculate a pixel-wise residual, and aggregate the residual into one image-level anomaly score.

**Expected benefit:** Low compute cost, direct visual interpretation, no learned model, and a clear connection between alignment quality and residual error.

**Expected failure:** Illumination change, imperfect registration, repetitive structures, local deformation, and normal appearance variation can create large residuals. An anomaly can also be suppressed by an incorrect alignment.

**Data requirement:** One or more normal references. More than one reference can form a robust normal template.

**Compute requirement:** Low to moderate CPU cost, dominated by registration and image operations.

**License or dependency concern:** The selected `opencv-python-headless==4.13.0.92` wheel has a composite boundary: packaging scripts use MIT terms, OpenCV uses Apache License 2.0, FFmpeg is included under LGPL-2.1, and other bundled binaries carry separate notices. The full resolved distribution and notices must be retained in the environment record.

**v0.1 decision:** **Adopt.** It is the minimum interpretable baseline. The implemented form will be an ECC-aligned normal-template residual, not an unregistered raw difference.

### SSIM residual

**Basic idea:** Compare local luminance, contrast, and structure between an aligned normal reference and the input, then convert dissimilarity into an anomaly score.

**Expected benefit:** More tolerant than raw pixel subtraction to some local photometric variation.

**Expected failure:** Window size and data range affect the result; registration error and legitimate local contrast changes can still dominate; aggregation may hide small defects.

**Data requirement:** At least one normal reference or template.

**Compute requirement:** Low to moderate CPU cost.

**License or dependency concern:** The selected runtime includes `scikit-image==0.26.0`. Its main terms are BSD-3-Clause, and its license file identifies BSD-2-Clause and MIT components. SSIM remains outside the selected method scope even though its likely package is already required for HOG.

**v0.1 decision:** **Do not select.** It belongs to the same reference-residual family as the registered pixel baseline. Including both would spend the limited milestone on two closely related methods rather than testing a different modeling assumption.

### HOG + One-Class SVM

**Basic idea:** Describe local gradient orientation patterns with HOG features and fit a one-class boundary from normal examples only.

**Expected benefit:** Tests whether a learned normal boundary over edge and layout information is more tolerant than direct image residuals while remaining CPU-friendly and classical.

**Expected failure:** HOG can miss color-only and low-gradient anomalies. Pose or alignment variation changes the descriptor. A One-Class SVM can be sensitive to feature scaling, kernel settings, and small-sample geometry.

**Data requirement:** Normal reference images only; v0.1 fixes the budget at 20.

**Compute requirement:** Moderate CPU cost for patch feature extraction and one-class scoring.

**License or dependency concern:** The selected direct dependencies are `scikit-image==0.26.0` for HOG and `scikit-learn==1.9.0` for One-Class SVM. scikit-image mainly uses BSD-3-Clause terms with separately identified BSD-2-Clause and MIT components; scikit-learn uses BSD-3-Clause. Their transitive dependencies and exact distribution notices must be recorded in the resolved lock before implementation.

**v0.1 decision:** **Adopt.** It supplies a distinct classical learned comparator without anomalous training labels or model weights.

### PCA reconstruction

**Basic idea:** Fit a low-dimensional linear normal subspace and use reconstruction error as the anomaly score.

**Expected benefit:** Simple, deterministic, CPU-friendly, and able to expose whether normal variation can be represented by a compact linear model.

**Expected failure:** Nonlinear normal variation is poorly represented; a flexible subspace may reconstruct anomalies; scaling and dimensionality choices can dominate with only 20 references.

**Data requirement:** Normal samples only, but stable component estimation generally benefits from more diverse normal data than the v0.1 reference budget.

**Compute requirement:** Low to moderate CPU cost, depending on feature dimensionality.

**License or dependency concern:** The selected baseline includes `numpy==2.5.1` and `scikit-learn==1.9.0`, but a PCA prototype is not authorized to use them in v0.1. NumPy's published metadata identifies multiple bundled-component license terms in addition to its top-level BSD-3-Clause license.

**v0.1 decision:** **Do not select.** The 20-image budget makes the fitted subspace a prominent confound, and adding it would create a third prototype without resolving the main baseline-versus-one-class question.

### Autoencoder

**Basic idea:** Train a neural network to reconstruct normal images and treat reconstruction error as evidence of anomaly.

**Expected benefit:** Can model nonlinear normal appearance and produce spatial residuals.

**Expected failure:** It may reconstruct anomalies, overfit a small normal set, depend heavily on architecture and training choices, or require augmentation that changes the research question.

**Data requirement:** Normal training data, typically more than the fixed 20-reference v0.1 budget for a defensible training study.

**Compute requirement:** Higher implementation and training cost; CPU training would be slow and GPU use is outside scope.

**License or dependency concern:** Framework, pretrained asset, and training-code licenses would need separate review. Model provenance would also need documentation.

**v0.1 decision:** **Reject for v0.1.** It conflicts with the short CPU-only milestone and would make training design a larger variable than the feasibility question.

### PatchCore

**Basic idea:** Store representative pretrained patch embeddings from normal images and score test patches by nearest-neighbor distance to the normal memory bank.

**Expected benefit:** Strong anomaly-detection precedent, no task-specific gradient training, and both image-level and spatial anomaly evidence.

**Expected failure:** CPU feature extraction and nearest-neighbor search may be expensive; memory-bank and coreset decisions add complexity; pretrained-domain mismatch can produce false alarms.

**Data requirement:** Normal reference images plus pretrained feature-extractor weights.

**Compute requirement:** Moderate to high CPU, memory, model-download, and implementation cost relative to v0.1.

**License or dependency concern:** The exact implementation, backbone, model weights, and their notices must be reviewed together before reuse. A paper description does not by itself license code or weights.

**v0.1 decision:** **Hold for v0.2 research.** It is relevant, but it would introduce pretrained assets and memory-bank choices before the two lower-cost baselines are evaluated.

### DINOv2 patch nearest neighbor

**Basic idea:** Extract DINOv2 patch embeddings, store normal reference features, and use nearest-neighbor distance as an anomaly score.

**Expected benefit:** Training-free reuse of a general visual representation and a direct test of whether pretrained features outperform handcrafted HOG under a small reference budget.

**Expected failure:** Patch resolution can miss small defects; pretrained semantics can ignore visually important local changes; CPU latency and model size may violate the v0.1 gates.

**Data requirement:** Normal reference images and pinned DINOv2 model weights.

**Compute requirement:** Higher CPU latency and memory than the v0.1 shortlist.

**License or dependency concern:** The official DINOv2 repository states that its code and standard model weights are Apache-2.0. Exact checkpoint identity, checksum, download terms, PyTorch stack, and notices would still need to be pinned.

**v0.1 decision:** **Hold for v0.2.** DINOv2 is explicitly excluded from implementation until a separate CPU and asset-management milestone is justified.

### AnomalyDINO

**Basic idea:** Use DINOv2 patch features in a one-shot or few-shot nearest-neighbor anomaly-detection design, with method choices intended for image-level detection and spatial localization.

**Expected benefit:** Directly relevant published evidence for few-shot industrial anomaly detection without task-specific fine-tuning.

**Expected failure:** A faithful reproduction adds preprocessing, masking, augmentation, patch-matching, and evaluation choices. Simplifying those choices without clear labeling could misrepresent the published method.

**Data requirement:** Few normal references and DINOv2 weights; benchmark evaluation requires labeled anomalies.

**Compute requirement:** Similar pretrained-feature cost to DINOv2 nearest neighbor, plus method-specific processing.

**License or dependency concern:** The paper is available for research review, but any reused implementation and model assets require their own license and notice review.

**v0.1 decision:** **Research reference only; hold implementation.** v0.1 may cite the method as motivation but must not claim to implement or reproduce AnomalyDINO.

## v0.1 Shortlist

| Method | Role in the comparison | Why it remains |
| --- | --- | --- |
| ECC-aligned normal-template residual | Interpretable low-complexity baseline | Tests whether alignment plus direct residual evidence is already sufficient |
| Patch HOG + One-Class SVM | Classical learned normal-only comparator | Tests a different modeling assumption without deep weights or anomalous training labels |

The shortlist is intentionally small. If neither method passes the preregistered gates, `REJECT` remains a valid result and motivates, but does not automatically authorize, a v0.2 pretrained-feature experiment.

The exact preprocessing, fitting, scoring, and failure rules are fixed in the [v0.1 method specification](method-specification.md).

The selected Python and direct package versions, composite license boundaries, and remaining transitive-lock gate are recorded in [the runtime dependency inventory](dependencies-and-licenses.md).

## Deferred v0.2 Question

The first deferred question is:

> Does a pinned DINOv2 patch nearest-neighbor prototype improve fixed-threshold recall enough to justify its additional CPU, model-asset, and dependency cost?

That question is not answered in v0.1.

## Licensing Boundary

This repository is a source-available, noncommercially licensed public portfolio project. The PolyForm Noncommercial License 1.0.0 applies only to original project code and documentation. Commercial use of those original materials requires a separate written license from the copyright holder.

VisA remains separately licensed under CC BY 4.0, and this repository does not impose additional restrictions on VisA data. Third-party dependencies and model assets remain governed by their respective licenses. Dependency-license descriptions in the longlist do not replace, modify, or extend those licenses.

## Sources and License References

- VisA official project and dataset notice:
  <https://github.com/amazon-science/spot-diff>
- VisA AWS Open Data entry:
  <https://registry.opendata.aws/visa/>
- CC BY 4.0:
  <https://creativecommons.org/licenses/by/4.0/>
- Runtime versions and license boundaries:
  [dependencies-and-licenses.md](dependencies-and-licenses.md)
- OpenCV license:
  <https://github.com/opencv/opencv/blob/4.x/LICENSE>
- DINOv2 official repository:
  <https://github.com/facebookresearch/dinov2>
- AnomalyDINO paper:
  <https://openaccess.thecvf.com/content/WACV2025/html/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.html>
- scikit-image 0.26.0 license:
  <https://github.com/scikit-image/scikit-image/blob/v0.26.0/LICENSE.txt>
- scikit-learn 1.9.0 license:
  <https://github.com/scikit-learn/scikit-learn/blob/1.9.0/COPYING>

This document records a technical review, not legal advice. Exact distributed dependencies and notices must be verified again when implementation versions are pinned.
