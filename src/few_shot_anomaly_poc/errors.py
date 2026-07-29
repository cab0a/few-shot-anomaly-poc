"""Project-specific exceptions."""

from enum import StrEnum


class DataPreparationError(Exception):
    """Base exception for expected data-preparation failures."""


class ConfigurationError(DataPreparationError):
    """Raised when a configuration file is invalid."""


class ChecksumMismatchError(DataPreparationError):
    """Raised when content does not match its pinned checksum."""


class UnsafeArchiveError(DataPreparationError):
    """Raised when an archive member cannot be extracted safely."""


class ManifestIntegrityError(DataPreparationError):
    """Raised when split metadata or generated manifests are inconsistent."""


class PreprocessingFailureCode(StrEnum):
    """Stable codes for an image that cannot enter a v0.1 method."""

    IMAGE_NOT_FOUND = "IMAGE_NOT_FOUND"
    IMAGE_READ_FAILED = "IMAGE_READ_FAILED"
    IMAGE_DECODE_FAILED = "IMAGE_DECODE_FAILED"
    INVALID_DECODED_IMAGE = "INVALID_DECODED_IMAGE"
    IMAGE_RESIZE_FAILED = "IMAGE_RESIZE_FAILED"
    INVALID_PREPROCESSED_IMAGE = "INVALID_PREPROCESSED_IMAGE"


class ImagePreprocessingError(Exception):
    """Carry a stable failure code without converting a failure into a score."""

    def __init__(self, code: PreprocessingFailureCode, message: str) -> None:
        super().__init__(message)
        self.code = code


class ECCRegistrationFailureCode(StrEnum):
    """Stable failure codes for the v0.1 ECC registration primitive."""

    ECC_OPTIMIZATION_FAILED = "ECC_OPTIMIZATION_FAILED"
    ECC_RESULT_INVALID = "ECC_RESULT_INVALID"
    ECC_RESULT_NONFINITE = "ECC_RESULT_NONFINITE"
    ECC_ROTATION_LIMIT_EXCEEDED = "ECC_ROTATION_LIMIT_EXCEEDED"
    ECC_TRANSLATION_LIMIT_EXCEEDED = "ECC_TRANSLATION_LIMIT_EXCEEDED"
    ECC_WARP_FAILED = "ECC_WARP_FAILED"
    ECC_VALID_AREA_TOO_SMALL = "ECC_VALID_AREA_TOO_SMALL"


class ECCTemplateFailureCode(StrEnum):
    """Stable method-level failure codes for ECC normal-template fitting."""

    FIT_REFERENCE_COUNT_INVALID = "FIT_REFERENCE_COUNT_INVALID"
    FIT_REFERENCE_SET_INVALID = "FIT_REFERENCE_SET_INVALID"
    FIT_ANCHOR_PREPROCESSING_FAILED = "FIT_ANCHOR_PREPROCESSING_FAILED"
    FIT_INSUFFICIENT_REFERENCES = "FIT_INSUFFICIENT_REFERENCES"
    FIT_SUPPORT_EROSION_FAILED = "FIT_SUPPORT_EROSION_FAILED"
    FIT_SUPPORT_TOO_SMALL = "FIT_SUPPORT_TOO_SMALL"
    FIT_TEMPLATE_INVALID = "FIT_TEMPLATE_INVALID"


class ECCResidualFailureCode(StrEnum):
    """Stable image-level failure codes for ECC residual scoring."""

    SCORE_MASK_EROSION_FAILED = "SCORE_MASK_EROSION_FAILED"
    SCORE_EFFECTIVE_SUPPORT_TOO_SMALL = "SCORE_EFFECTIVE_SUPPORT_TOO_SMALL"
    SCORE_RESIDUAL_FILTER_FAILED = "SCORE_RESIDUAL_FILTER_FAILED"
    SCORE_RESULT_INVALID = "SCORE_RESULT_INVALID"


class ECCResidualStateError(Exception):
    """Reject scoring when no valid fitted ECC method state is available."""


class HOGFeatureFailureCode(StrEnum):
    """Stable failure codes for patch-grid and HOG extraction."""

    HOG_GRID_INVALID = "HOG_GRID_INVALID"
    HOG_EXTRACTION_FAILED = "HOG_EXTRACTION_FAILED"
    HOG_DESCRIPTOR_INVALID = "HOG_DESCRIPTOR_INVALID"


class HOGScalerFailureCode(StrEnum):
    """Stable method-level failure codes for position-wise scaler fitting."""

    HOG_FIT_REFERENCE_COUNT_INVALID = "HOG_FIT_REFERENCE_COUNT_INVALID"
    HOG_FIT_REFERENCE_SET_INVALID = "HOG_FIT_REFERENCE_SET_INVALID"
    HOG_FIT_REFERENCE_FEATURES_INVALID = "HOG_FIT_REFERENCE_FEATURES_INVALID"
    HOG_FIT_SCALER_FAILED = "HOG_FIT_SCALER_FAILED"
    HOG_FIT_SCALER_STATE_INVALID = "HOG_FIT_SCALER_STATE_INVALID"


class HOGModelFailureCode(StrEnum):
    """Stable method-level failure codes for position-wise SVM fitting."""

    HOG_MODEL_REFERENCE_COUNT_INVALID = "HOG_MODEL_REFERENCE_COUNT_INVALID"
    HOG_MODEL_REFERENCE_SET_INVALID = "HOG_MODEL_REFERENCE_SET_INVALID"
    HOG_MODEL_REFERENCE_FEATURES_INVALID = "HOG_MODEL_REFERENCE_FEATURES_INVALID"
    HOG_MODEL_SCALER_STATE_INVALID = "HOG_MODEL_SCALER_STATE_INVALID"
    HOG_MODEL_TRANSFORM_FAILED = "HOG_MODEL_TRANSFORM_FAILED"
    HOG_MODEL_TRANSFORM_INVALID = "HOG_MODEL_TRANSFORM_INVALID"
    HOG_MODEL_FIT_FAILED = "HOG_MODEL_FIT_FAILED"
    HOG_MODEL_STATE_INVALID = "HOG_MODEL_STATE_INVALID"


class HOGScoringFailureCode(StrEnum):
    """Stable image-level failure codes for Patch HOG scoring."""

    HOG_SCORE_FEATURE_RESULT_INVALID = "HOG_SCORE_FEATURE_RESULT_INVALID"
    HOG_SCORE_TRANSFORM_FAILED = "HOG_SCORE_TRANSFORM_FAILED"
    HOG_SCORE_TRANSFORM_INVALID = "HOG_SCORE_TRANSFORM_INVALID"
    HOG_SCORE_DECISION_FAILED = "HOG_SCORE_DECISION_FAILED"
    HOG_SCORE_PATCH_INVALID = "HOG_SCORE_PATCH_INVALID"
    HOG_SCORE_AGGREGATION_INVALID = "HOG_SCORE_AGGREGATION_INVALID"


class HOGScoringStateError(Exception):
    """Reject scoring when no valid fitted Patch HOG state is available."""


class ThresholdCalibrationFailureCode(StrEnum):
    """Stable failure codes for normal-only threshold calibration."""

    CALIBRATION_METHOD_INVALID = "CALIBRATION_METHOD_INVALID"
    CALIBRATION_EMPTY = "CALIBRATION_EMPTY"
    CALIBRATION_PATH_INVALID = "CALIBRATION_PATH_INVALID"
    CALIBRATION_SCORE_TYPE_MISMATCH = "CALIBRATION_SCORE_TYPE_MISMATCH"
    CALIBRATION_SCORE_RECORD_INVALID = "CALIBRATION_SCORE_RECORD_INVALID"
    CALIBRATION_RESULT_INVALID = "CALIBRATION_RESULT_INVALID"


class ThresholdClassificationFailureCode(StrEnum):
    """Stable failure codes for fixed-threshold image classification."""

    CLASSIFICATION_PATH_INVALID = "CLASSIFICATION_PATH_INVALID"
    CLASSIFICATION_CALIBRATION_INVALID = "CLASSIFICATION_CALIBRATION_INVALID"
    CLASSIFICATION_SCORE_TYPE_MISMATCH = "CLASSIFICATION_SCORE_TYPE_MISMATCH"
    CLASSIFICATION_SCORE_RECORD_INVALID = "CLASSIFICATION_SCORE_RECORD_INVALID"
    CLASSIFICATION_RESULT_INVALID = "CLASSIFICATION_RESULT_INVALID"
