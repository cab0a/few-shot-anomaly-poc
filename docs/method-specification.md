# v0.1 Method Specification

## Status and Change Control

This document preregistered the v0.1 method definitions before data download, implementation, or result inspection. Later implementation-status notes do not change the fixed method choices.

The values below are fixed for the first valid final-test run. They must not be tuned with anomaly labels, final-test labels, final-test scores, or failure examples. A change required by an implementation defect or an unavailable API must be documented and committed before another final-test run.

## Shared Input Preprocessing

Both methods use the same deterministic input conversion:

1. Decode the image as 8-bit grayscale while ignoring orientation metadata and
   preserving the encoded pixel-array order.
2. Reject an input that cannot be decoded into a non-empty two-dimensional array.
3. Resize directly to `512 x 512` pixels with area interpolation.
4. Convert to `float32` and divide by `255.0`.
5. Reject the result if its shape is not exactly `(512, 512)` or if any value is non-finite.

No aspect-ratio preservation, crop, histogram equalization, contrast normalization, denoising, augmentation, or color feature is used in v0.1.

The orientation rule is fixed before implementation and before any VisA image
content is decoded. It maps to OpenCV `IMREAD_GRAYSCALE |
IMREAD_IGNORE_ORIENTATION` and prevents library-default metadata handling from
changing the input array.

Direct square resizing is a deliberate CPU and reproducibility tradeoff. It can distort geometry and suppress small anomalies; that limitation must remain in the final report.

## Method A: ECC-Aligned Normal-Template Residual

### Role

This is the low-complexity, interpretable baseline. It tests whether rigid in-plane alignment followed by direct appearance comparison is sufficient.

### Fixed registration parameters

| Parameter | Fixed value |
| --- | --- |
| Motion model | Euclidean: rotation and translation only |
| Initial warp | `2 x 3` identity matrix |
| Termination type | iteration count plus epsilon |
| Maximum iterations | `100` |
| Epsilon | `1e-6` |
| ECC Gaussian filter size | `5` |
| Warp interpolation | linear |
| Mask interpolation | nearest neighbor |
| Warp border | constant zero |
| Maximum absolute rotation | `10 degrees` |
| Maximum absolute horizontal translation | `64 pixels` |
| Maximum absolute vertical translation | `64 pixels` |
| Minimum warped valid area | `80%` of the resized image |

The transform is estimated with the template as the reference image and the candidate image as the moving input. The moving image is warped into template coordinates with the inverse-map convention documented by OpenCV.

### Registration validity

A registration succeeds only when all of the following hold:

- ECC returns without an exception.
- The correlation coefficient is finite.
- Every warp-matrix value is finite.
- Rotation and translation remain within the fixed plausibility limits.
- The warped binary validity mask covers at least `80%` of the resized image.

A low but finite correlation coefficient is recorded but is not, by itself, grounds for rejection. This prevents a result-dependent correlation cutoff.

The registration primitive returns `status=failed` with one of these stable
codes rather than retrying with different parameters:

- `ECC_OPTIMIZATION_FAILED`
- `ECC_RESULT_INVALID`
- `ECC_RESULT_NONFINITE`
- `ECC_ROTATION_LIMIT_EXCEEDED`
- `ECC_TRANSLATION_LIMIT_EXCEEDED`
- `ECC_WARP_FAILED`
- `ECC_VALID_AREA_TOO_SMALL`

Successful registration records the correlation, `2 x 3` warp matrix, rotation,
horizontal and vertical translations, and valid-area fraction. It also returns
the aligned `float32` image and binary validity mask for the later template and
residual stages. This primitive does not fit a template or calculate an anomaly
score.

### Template fitting

1. Sort the 20 reference relative paths in ascending Unicode code-point order.
2. Use the first path as the anchor with an identity transform.
3. Register each of the other 19 reference images to the anchor.
4. Retain every successful registration and record every rejected reference with a reason.
5. Require at least 16 successful references, including the anchor. Otherwise, record `FIT_FAILED`.
6. Form a support mask from the intersection of all successful warped validity masks.
7. Erode the support mask once with a `5 x 5` square kernel.
8. Require the eroded support to cover at least `75%` of the resized image. Otherwise, record `FIT_FAILED`.
9. Calculate a full-size pixel-wise median from the valid successfully aligned reference values at each pixel. Retain the eroded intersection only as the residual-scoring support mask.

The erosion uses one iteration and a constant-zero border. Template fitting
returns `FIT_FAILED` with one stable method-level code:

- `FIT_REFERENCE_COUNT_INVALID`
- `FIT_REFERENCE_SET_INVALID`
- `FIT_ANCHOR_PREPROCESSING_FAILED`
- `FIT_INSUFFICIENT_REFERENCES`
- `FIT_SUPPORT_EROSION_FAILED`
- `FIT_SUPPORT_TOO_SMALL`
- `FIT_TEMPLATE_INVALID`

A successful fit records the anchor path, sorted per-reference diagnostics,
successful and failed reference counts, eroded support fraction, full-size
template, and residual-scoring support mask. No incomplete fitted state is
returned after a method-level failure.

No calibration image may influence the anchor, accepted reference set, support mask, or template.

### Image scoring

1. Register the preprocessed image to the fitted template with the fixed ECC settings.
2. Erode the warped image-validity mask once with a `5 x 5` square kernel.
3. Intersect it with the template support mask.
4. Require the effective mask to retain at least `95%` of the template-support pixels.
5. Calculate the absolute grayscale residual inside the effective mask.
6. Smooth the residual map with a `5 x 5` Gaussian filter using automatic sigma.
7. Let `N` be the number of effective pixels and set `k = max(1, ceil(0.01 * N))`.
8. The image anomaly score is the arithmetic mean of the `k` largest residual values.

The warped validity-mask erosion uses one iteration and a constant-zero border.
The Gaussian filter uses sigma `0.0` and a constant-zero border. It is applied
to the full absolute residual, while only values inside the effective mask may
enter the top-`k` aggregation. The two fixed erosions ensure that every selected
pixel has the required `5 x 5` valid neighborhood.

Higher scores mean more anomalous. A valid score lies in `[0, 1]`.

Scoring requires a successful fitted method state. A method-level `FIT_FAILED`
state is rejected before image registration and is not converted into the
finite image-level failure score.

An image-level scoring failure preserves any shared-preprocessing or ECC
registration failure code. Failures introduced by the residual stage use:

- `SCORE_MASK_EROSION_FAILED`
- `SCORE_EFFECTIVE_SUPPORT_TOO_SMALL`
- `SCORE_RESIDUAL_FILTER_FAILED`
- `SCORE_RESULT_INVALID`

### Required diagnostic record

For each scored image, record:

- ECC status and correlation coefficient
- Warp matrix
- Rotation and translations
- Effective valid-area fraction
- Effective pixel count and top-`k` pixel count
- Anomaly score
- Failure code when applicable

## Method B: Patch HOG + One-Class SVM

### Role

This is the classical learned comparator. It tests whether local gradient structure modeled from normal references is more tolerant than direct residual comparison.

It does not use ECC alignment. Pose sensitivity is therefore an explicit failure condition rather than a hidden shared preprocessing step.

### Patch grid

| Parameter | Fixed value |
| --- | --- |
| Patch size | `64 x 64` pixels |
| Horizontal stride | `32 pixels` |
| Vertical stride | `32 pixels` |
| Top-left coordinates per axis | `0, 32, ..., 448` |
| Patch positions per image | `15 x 15 = 225` |

Each patch position has its own scaler and One-Class SVM. The 20 descriptors at one fixed position are its only fitting samples. Position-specific models preserve coarse layout without adding spatial coordinates to the descriptor.

### HOG parameters

| Parameter | Fixed value |
| --- | --- |
| Orientations | `9` |
| Pixels per cell | `(16, 16)` |
| Cells per block | `(2, 2)` |
| Block normalization | `L2-Hys` |
| Power-law compression | enabled |
| Visualization output | disabled |
| Flattened feature vector | enabled |
| Channel axis | none |
| Descriptor length | `324` |

No color histogram, raw pixel value, texture descriptor, PCA, feature selection, or augmentation is appended.

### Feature extraction output

The shared preprocessed image is traversed in row-major patch order. Extraction
returns one complete `float32` matrix with shape `(225, 324)`, where each row
corresponds to the fixed patch position with the same index. No partial feature
matrix is returned after a failed patch.

Shared input failures preserve their preprocessing failure code. Failures
introduced by the Patch HOG stage use:

- `HOG_GRID_INVALID`
- `HOG_EXTRACTION_FAILED`
- `HOG_DESCRIPTOR_INVALID`

The diagnostic state records the complete planned patch positions and the
failed patch index when extraction reached a specific patch. Feature extraction
does not fit or apply a scaler, fit a One-Class SVM, or produce an image anomaly
score.

### Feature scaling

Fit one standard scaler per patch position from its 20 reference descriptors:

| Parameter | Fixed value |
| --- | --- |
| Center features | enabled |
| Scale to unit variance | enabled |
| Copy input | enabled |

Scaler fitting requires exactly 20 complete reference feature matrices. Paths
are sorted in ascending Unicode code-point order, and patch position `p` uses
only the 20 descriptors at row `p`, producing 225 independent fitted scalers.
Each fit input therefore has shape `(20, 324)`.

Every fitted scaler must record 324 finite means, variances, and scales, with
non-negative variance, strictly positive scale, `n_features_in_ = 324`, and
`n_samples_seen_ = 20`. A constant feature dimension is valid: scikit-learn
records variance zero and scale one for that dimension.

No partial scaler collection is returned after a failed reference or patch
position. Fitting failures use:

- `HOG_FIT_REFERENCE_COUNT_INVALID`
- `HOG_FIT_REFERENCE_SET_INVALID`
- `HOG_FIT_REFERENCE_FEATURES_INVALID`
- `HOG_FIT_SCALER_FAILED`
- `HOG_FIT_SCALER_STATE_INVALID`

The scaler may use only reference descriptors. Calibration descriptors must be transformed with the fitted scaler and must not update it.

### One-Class SVM parameters

Fit one model per patch position with:

| Parameter | Fixed value |
| --- | --- |
| Kernel | radial basis function |
| `gamma` | `scale` |
| `nu` | `0.05` |
| Solver tolerance | `1e-3` |
| Shrinking heuristic | enabled |
| Kernel cache | `200 MB` |
| Maximum iterations | unlimited |
| Verbose output | disabled |

Every model must finish with a successful fit status and finite fitted state. Otherwise, record `FIT_FAILED` for the method.

### Image scoring

1. Extract the 225 HOG descriptors in row-major patch order.
2. Transform each descriptor with the scaler for that position.
3. Calculate the One-Class SVM decision function for that position.
4. Negate the decision function so that higher values mean more anomalous.
5. Reject any non-finite patch score or any absolute patch score greater than or equal to `1e12`.
6. Set `k = ceil(0.05 * 225) = 12`.
7. The image anomaly score is the arithmetic mean of the 12 largest patch anomaly scores.

The 12 contributing patch positions may be recorded for diagnosis, but they are not a pixel-level prediction and must not be evaluated as localization.

## Failure-Score Policy

An image-level preprocessing, registration, feature, transform, or scoring failure must not be dropped or retried with different parameters.

Shared preprocessing uses these stable failure codes:

- `IMAGE_NOT_FOUND`
- `IMAGE_READ_FAILED`
- `IMAGE_DECODE_FAILED`
- `INVALID_DECODED_IMAGE`
- `IMAGE_RESIZE_FAILED`
- `INVALID_PREPROCESSED_IMAGE`

| Method | Fixed finite failure score |
| --- | --- |
| ECC residual | `1.0` |
| Patch HOG + One-Class SVM | `1e12` |

Every score record also contains `score_status`, which is either `ok` or `failed`, plus a failure code.

A failed image is always classified as anomalous regardless of the numeric threshold. Its fixed failure score remains in ranking-metric inputs and threshold-calibration inputs. This fail-safe convention can improve anomaly recall while increasing normal false positives, so the separate failure count is mandatory evidence.

A method-level `FIT_FAILED` status means that the method is unavailable, receives no passing result, and cannot be selected for `ADOPT` or `ADOPT WITH CONDITIONS`.

## Calibration Boundary

Calibration may choose only one threshold per successfully fitted method.

For `n` calibration scores sorted in ascending order:

```text
rank = ceil(0.95 * n)
threshold = sorted_scores[rank - 1]
```

An image is classified as anomalous when `score_status` is `failed` or when its score is strictly greater than the threshold.

Calibration must not change preprocessing, registration limits, template fitting, residual aggregation, patch geometry, HOG parameters, scaler behavior, One-Class SVM parameters, or failure handling.

## Explicitly Prohibited v0.1 Adjustments

- Parameter sweeps
- Method selection with calibration performance
- Feature or score normalization fitted from calibration images
- Threshold selection from anomaly labels
- Threshold selection from final-test labels
- Per-defect thresholds
- Manual exclusion of registration or scoring failures
- A second method configuration after inspecting final-test failures
- DINOv2, PatchCore, AnomalyDINO, or pixel-level evaluation

## Sources

- OpenCV image decoding and orientation flags:
  <https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html>
- OpenCV `findTransformECC`:
  <https://docs.opencv.org/4.x/dc/d6b/group__video__track.html>
- scikit-image `hog`:
  <https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.hog>
- scikit-learn `StandardScaler`:
  <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html>
- scikit-learn `OneClassSVM`:
  <https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html>

The Python and direct package versions are preregistered in [the runtime dependency inventory](dependencies-and-licenses.md). This specification fixes method behavior; the dependency inventory fixes the selected runtime baseline and records the remaining transitive-lock gate.
