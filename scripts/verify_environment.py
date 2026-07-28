"""Verify the locked runtime and the shortlisted methods' required APIs."""

from __future__ import annotations

import json
import platform
from importlib.metadata import version

import cv2
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


EXPECTED_PYTHON = "3.13.14"
EXPECTED_DISTRIBUTIONS = {
    "numpy": "2.5.1",
    "opencv-python-headless": "4.13.0.92",
    "scikit-image": "0.26.0",
    "scikit-learn": "1.9.0",
}


def verify_versions() -> dict[str, str]:
    """Return verified runtime versions or raise on a mismatch."""
    actual_python = platform.python_version()
    if actual_python != EXPECTED_PYTHON:
        raise RuntimeError(
            f"Expected Python {EXPECTED_PYTHON}, found {actual_python}."
        )

    actual_distributions = {
        name: version(name) for name in EXPECTED_DISTRIBUTIONS
    }
    if actual_distributions != EXPECTED_DISTRIBUTIONS:
        raise RuntimeError(
            "Locked distribution mismatch: "
            f"expected {EXPECTED_DISTRIBUTIONS}, found {actual_distributions}."
        )

    return {"python": actual_python, **actual_distributions}


def verify_ecc_api() -> float:
    """Exercise OpenCV ECC registration on a deterministic synthetic image."""
    template = np.zeros((64, 64), dtype=np.uint8)
    cv2.rectangle(template, (16, 20), (46, 44), color=255, thickness=-1)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        20,
        1e-6,
    )
    correlation, fitted_warp = cv2.findTransformECC(
        template,
        template.copy(),
        warp,
        cv2.MOTION_EUCLIDEAN,
        criteria,
    )
    if not np.isfinite(correlation) or not np.isfinite(fitted_warp).all():
        raise RuntimeError("OpenCV ECC returned a non-finite result.")
    return float(correlation)


def verify_hog_and_svm_apis() -> tuple[int, float]:
    """Exercise the fixed HOG shape and a minimal one-class fit."""
    patch = np.zeros((64, 64), dtype=np.float32)
    descriptor = hog(
        patch,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
        channel_axis=None,
    )
    if descriptor.shape != (324,):
        raise RuntimeError(f"Expected HOG shape (324,), found {descriptor.shape}.")

    samples = np.random.default_rng(42).normal(size=(20, descriptor.size))
    scaled = StandardScaler().fit_transform(samples)
    model = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=0.05,
        tol=1e-3,
        shrinking=True,
        cache_size=200,
        max_iter=-1,
        verbose=False,
    ).fit(scaled)
    score = model.decision_function(scaled[:1])
    if score.shape != (1,) or not np.isfinite(score).all():
        raise RuntimeError("One-Class SVM returned an invalid decision score.")

    return descriptor.size, float(score[0])


def main() -> None:
    """Run dependency checks and print a machine-readable summary."""
    versions = verify_versions()
    ecc_correlation = verify_ecc_api()
    hog_length, svm_score = verify_hog_and_svm_apis()
    print(
        json.dumps(
            {
                "status": "ok",
                "versions": versions,
                "checks": {
                    "opencv_find_transform_ecc": "ok",
                    "ecc_correlation": ecc_correlation,
                    "hog_descriptor_length": hog_length,
                    "standard_scaler": "ok",
                    "one_class_svm": "ok",
                    "one_class_svm_finite_score": svm_score,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
