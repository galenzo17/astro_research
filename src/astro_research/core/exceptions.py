"""Custom exceptions for the asteroid detection pipeline."""


class PipelineError(Exception):
    """Base exception for all pipeline errors."""
    
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


class DataError(PipelineError):
    """Exception raised for data-related errors."""
    pass


class DetectionError(PipelineError):
    """Exception raised during object detection."""
    pass


class ValidationError(PipelineError):
    """Exception raised during validation against catalogs."""
    pass


class ConfigurationError(PipelineError):
    """Exception raised for configuration errors."""
    pass


class DownloadError(DataError):
    """Exception raised during data download."""
    pass


class PreprocessingError(PipelineError):
    """Exception raised during image preprocessing."""
    pass