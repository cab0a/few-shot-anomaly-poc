"""Project-specific exceptions."""


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
