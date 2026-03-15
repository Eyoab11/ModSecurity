"""
Custom Exception Hierarchy for ML Training Pipeline

Provides a hierarchy of custom exceptions for different error scenarios
in the training pipeline, enabling precise error handling and reporting.
"""


class PipelineError(Exception):
    """
    Base exception class for all ML training pipeline errors.
    
    All custom exceptions in the pipeline inherit from this base class,
    allowing for catch-all error handling when needed.
    """
    pass


class DataLoadError(PipelineError):
    """
    Exception raised when data loading fails.
    
    This includes scenarios such as:
    - Missing or unreadable dataset files
    - Invalid dataset schema (missing required columns)
    - Empty datasets after cleaning
    - CSV parsing errors
    """
    pass


class FeatureExtractionError(PipelineError):
    """
    Exception raised when feature extraction fails.
    
    This includes scenarios such as:
    - Vectorizer fitting failures
    - Transform operations on unfitted vectorizers
    - Empty payload lists during fitting
    - Feature extraction serialization/deserialization errors
    """
    pass


class ModelTrainingError(PipelineError):
    """
    Exception raised when model training fails.
    
    This is a non-fatal exception for individual algorithm failures.
    The pipeline can continue if at least one algorithm succeeds.
    """
    pass


class AllAlgorithmsFailedError(PipelineError):
    """
    Exception raised when all algorithms fail to train.
    
    This is a fatal error that terminates the pipeline, as no models
    can be produced if all training attempts fail.
    """
    pass
