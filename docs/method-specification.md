# v0.1 Method Specification

## Status and Change Control

This document preregisters the v0.1 method definitions before data download, implementation, or result inspection.

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

Higher scores mean more anomalous. A valid score lies in `[0, 1]`.

### Required diagnostic record

For each scored image, record:

- ECC status and correlation coefficient
- Warp matrix
- Rotation and translations
- Effective valid-area fraction
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

### Feature scaling

Fit one standard scaler per patch position from its 20 reference descriptors:

| Parameter | Fixed value |
| --- | --- |
| Center features | enabled |
| Scale to unit variance | enabled |
| Copy input | enabled |

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
