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
