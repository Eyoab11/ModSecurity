"""
Unit tests for error handling infrastructure.

Tests the custom exception hierarchy and error logging with stack traces.
"""

import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path

from src.ml_core.data_loader import DataLoader
from src.ml_core.extractor import FeatureExtractor
from src.ml_core.trainer import ModelTrainer
from src.ml_core.exceptions import (
    PipelineError,
    DataLoadError,
    FeatureExtractionError,
    ModelTrainingError,
    AllAlgorithmsFailedError
)


class TestExceptionHierarchy:
    """Test the custom exception hierarchy."""
    
    def test_pipeline_error_is_base_exception(self):
        """Test that PipelineError is the base exception."""
        assert issubclass(PipelineError, Exception)
    
    def test_data_load_error_inherits_from_pipeline_error(self):
        """Test that DataLoadError inherits from PipelineError."""
        assert issubclass(DataLoadError, PipelineError)
    
    def test_feature_extraction_error_inherits_from_pipeline_error(self):
        """Test that FeatureExtractionError inherits from PipelineError."""
        assert issubclass(FeatureExtractionError, PipelineError)
    
    def test_model_training_error_inherits_from_pipeline_error(self):
        """Test that ModelTrainingError inherits from PipelineError."""
        assert issubclass(ModelTrainingError, PipelineError)
    
    def test_all_algorithms_failed_error_inherits_from_pipeline_error(self):
        """Test that AllAlgorithmsFailedError inherits from PipelineError."""
        assert issubclass(AllAlgorithmsFailedError, PipelineError)
    
    def test_can_catch_all_errors_with_pipeline_error(self):
        """Test that all custom errors can be caught with PipelineError."""
        # DataLoadError
        with pytest.raises(PipelineError):
            raise DataLoadError("test error")
        
        # FeatureExtractionError
        with pytest.raises(PipelineError):
            raise FeatureExtractionError("test error")
        
        # ModelTrainingError
        with pytest.raises(PipelineError):
            raise ModelTrainingError("test error")
        
        # AllAlgorithmsFailedError
        with pytest.raises(PipelineError):
            raise AllAlgorithmsFailedError("test error")


class TestDataLoaderErrorHandling:
    """Test error handling in DataLoader component."""
    
    def test_missing_file_raises_data_load_error(self):
        """Test that missing file raises DataLoadError."""
        loader = DataLoader("nonexistent.csv")
        with pytest.raises(DataLoadError, match="Dataset file not found"):
            loader.load_and_validate()
    
    def test_empty_dataset_raises_data_load_error(self):
        """Test that empty dataset raises DataLoadError."""
        loader = DataLoader()
        
        # Create DataFrame with all null values
        test_df = pd.DataFrame({
            'payload': [None, None],
            'label': [None, None],
            'source_file': ['file1', 'file2']
        })
        
        with pytest.raises(DataLoadError, match="empty after removing"):
            loader.clean_data(test_df)


class TestFeatureExtractorErrorHandling:
    """Test error handling in FeatureExtractor component."""
    
    def test_fit_on_empty_payloads_raises_feature_extraction_error(self):
        """Test that fitting on empty payloads raises FeatureExtractionError."""
        payloads = pd.Series([])
        extractor = FeatureExtractor()
        
        with pytest.raises(FeatureExtractionError, match="Cannot fit on empty payload list"):
            extractor.fit(payloads)
    
    def test_transform_before_fit_raises_feature_extraction_error(self):
        """Test that transform before fit raises FeatureExtractionError."""
        payloads = pd.Series(["test payload"])
        extractor = FeatureExtractor()
        
        with pytest.raises(FeatureExtractionError, match="not fitted yet"):
            extractor.transform(payloads)
    
    def test_save_unfitted_raises_feature_extraction_error(self):
        """Test that saving unfitted extractor raises FeatureExtractionError."""
        extractor = FeatureExtractor()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "extractor.pkl")
            with pytest.raises(FeatureExtractionError, match="Cannot save unfitted"):
                extractor.save(path)
    
    def test_load_nonexistent_file_raises_feature_extraction_error(self):
        """Test that loading nonexistent file raises FeatureExtractionError."""
        with pytest.raises(FeatureExtractionError, match="file not found"):
            FeatureExtractor.load("nonexistent_file.pkl")


class TestModelTrainerErrorHandling:
    """Test error handling in ModelTrainer component."""
    
    def test_train_invalid_algorithm_raises_model_training_error(self):
        """Test that training invalid algorithm raises ModelTrainingError."""
        trainer = ModelTrainer(algorithms=["InvalidAlgorithm"])
        
        # Create minimal training data
        X_train = [[1, 2], [3, 4]]
        y_train = [0, 1]
        
        with pytest.raises(ModelTrainingError):
            trainer.train_algorithm("InvalidAlgorithm", X_train, y_train)
    
    def test_all_algorithms_failed_raises_all_algorithms_failed_error(self):
        """Test that all algorithms failing raises AllAlgorithmsFailedError."""
        # Use invalid algorithm names to force all to fail
        trainer = ModelTrainer(algorithms=["Invalid1", "Invalid2"])
        
        # Create minimal training data
        X_train = [[1, 2], [3, 4]]
        y_train = [0, 1]
        X_test = [[5, 6]]
        y_test = [1]
        
        with pytest.raises(AllAlgorithmsFailedError, match="All algorithms failed"):
            trainer.train_all(X_train, y_train, X_test, y_test)
