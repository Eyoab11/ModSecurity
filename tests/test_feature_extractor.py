"""
Unit tests for FeatureExtractor component.

Tests the feature extraction functionality including TF-IDF vectorization,
serialization, and validation.
"""

import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path

from src.ml_core.extractor import FeatureExtractor
from src.ml_core.exceptions import FeatureExtractionError


class TestFeatureExtractorBasic:
    """Basic functionality tests for FeatureExtractor."""
    
    def test_initialization(self):
        """Test FeatureExtractor initialization with default parameters."""
        extractor = FeatureExtractor()
        assert extractor.max_features == 5000
        assert extractor.ngram_range == (2, 5)
        assert not extractor._is_fitted
    
    def test_initialization_custom_params(self):
        """Test FeatureExtractor initialization with custom parameters."""
        extractor = FeatureExtractor(max_features=1000, ngram_range=(3, 4))
        assert extractor.max_features == 1000
        assert extractor.ngram_range == (3, 4)
    
    def test_fit_on_valid_payloads(self):
        """Test fitting vectorizer on valid payload data."""
        payloads = pd.Series([
            "SELECT * FROM users",
            "<script>alert('xss')</script>",
            "../../etc/passwd",
            "normal request"
        ])
        
        extractor = FeatureExtractor()
        result = extractor.fit(payloads)
        
        # Should return self for chaining
        assert result is extractor
        assert extractor._is_fitted
    
    def test_fit_on_empty_payloads_raises_error(self):
        """Test that fitting on empty payloads raises FeatureExtractionError."""
        payloads = pd.Series([])
        extractor = FeatureExtractor()
        
        with pytest.raises(FeatureExtractionError, match="Cannot fit on empty payload list"):
            extractor.fit(payloads)
    
    def test_transform_before_fit_raises_error(self):
        """Test that transform before fit raises FeatureExtractionError."""
        payloads = pd.Series(["test payload"])
        extractor = FeatureExtractor()
        
        with pytest.raises(FeatureExtractionError):
            extractor.transform(payloads)
    
    def test_fit_transform(self):
        """Test fit_transform method."""
        payloads = pd.Series([
            "SELECT * FROM users",
            "<script>alert('xss')</script>",
            "../../etc/passwd"
        ])
        
        extractor = FeatureExtractor(max_features=100)
        feature_vectors = extractor.fit_transform(payloads)
        
        # Check shape
        assert feature_vectors.shape[0] == 3  # 3 samples
        assert feature_vectors.shape[1] <= 100  # At most max_features
        assert extractor._is_fitted
    
    def test_transform_after_fit(self):
        """Test transform on new data after fitting."""
        train_payloads = pd.Series([
            "SELECT * FROM users",
            "<script>alert('xss')</script>",
            "../../etc/passwd"
        ])
        
        test_payloads = pd.Series([
            "SELECT id FROM accounts",
            "<img src=x onerror=alert(1)>"
        ])
        
        extractor = FeatureExtractor(max_features=100)
        extractor.fit(train_payloads)
        feature_vectors = extractor.transform(test_payloads)
        
        # Check shape
        assert feature_vectors.shape[0] == 2  # 2 test samples
        assert feature_vectors.shape[1] <= 100
    
    def test_feature_vector_dimensionality(self):
        """Test that feature vectors have correct dimensionality."""
        payloads = pd.Series([
            "a" * 100,  # Simple payload
            "b" * 100,
            "c" * 100
        ])
        
        max_features = 50
        extractor = FeatureExtractor(max_features=max_features)
        feature_vectors = extractor.fit_transform(payloads)
        
        # Dimensionality should be <= max_features
        assert feature_vectors.shape[1] <= max_features
    
    def test_get_feature_names_before_fit_raises_error(self):
        """Test that get_feature_names before fit raises FeatureExtractionError."""
        extractor = FeatureExtractor()
        
        with pytest.raises(FeatureExtractionError):
            extractor.get_feature_names()
    
    def test_get_feature_names_after_fit(self):
        """Test get_feature_names returns vocabulary."""
        payloads = pd.Series([
            "SELECT * FROM users",
            "<script>alert('xss')</script>"
        ])
        
        extractor = FeatureExtractor(max_features=100)
        extractor.fit(payloads)
        feature_names = extractor.get_feature_names()
        
        # Should return a list of strings
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        assert all(isinstance(name, str) for name in feature_names)


class TestFeatureExtractorSerialization:
    """Tests for FeatureExtractor serialization and deserialization."""
    
    def test_save_unfitted_extractor_raises_error(self):
        """Test that saving unfitted extractor raises FeatureExtractionError."""
        extractor = FeatureExtractor()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "extractor.pkl")
            with pytest.raises(FeatureExtractionError):
                extractor.save(path)
    
    def test_save_and_load_round_trip(self):
        """Test that save and load produces identical transformations."""
        payloads = pd.Series([
            "SELECT * FROM users",
            "<script>alert('xss')</script>",
            "../../etc/passwd",
            "normal request"
        ])
        
        test_payload = pd.Series(["SELECT id FROM accounts"])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "extractor.pkl")
            
            # Fit and transform with original extractor
            extractor1 = FeatureExtractor(max_features=100)
            extractor1.fit(payloads)
            features1 = extractor1.transform(test_payload)
            
            # Save extractor
            extractor1.save(path)
            
            # Load extractor
            extractor2 = FeatureExtractor.load(path)
            features2 = extractor2.transform(test_payload)
            
            # Features should be identical
            assert (features1.toarray() == features2.toarray()).all()
    
    def test_load_nonexistent_file_raises_error(self):
        """Test that loading nonexistent file raises FeatureExtractionError."""
        with pytest.raises(FeatureExtractionError):
            FeatureExtractor.load("nonexistent_file.pkl")
    
    def test_save_creates_directory(self):
        """Test that save creates parent directory if it doesn't exist."""
        payloads = pd.Series(["test payload"])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "subdir", "extractor.pkl")
            
            extractor = FeatureExtractor()
            extractor.fit(payloads)
            extractor.save(path)
            
            # Directory should be created
            assert os.path.exists(path)
            assert os.path.isfile(path)


class TestFeatureExtractorConsistency:
    """Tests for feature extraction consistency."""
    
    def test_transform_same_payload_multiple_times(self):
        """Test that transforming same payload multiple times produces identical results."""
        train_payloads = pd.Series([
            "SELECT * FROM users",
            "<script>alert('xss')</script>",
            "../../etc/passwd"
        ])
        
        test_payload = pd.Series(["SELECT id FROM accounts"])
        
        extractor = FeatureExtractor(max_features=100)
        extractor.fit(train_payloads)
        
        # Transform same payload multiple times
        features1 = extractor.transform(test_payload)
        features2 = extractor.transform(test_payload)
        features3 = extractor.transform(test_payload)
        
        # All should be identical
        assert (features1.toarray() == features2.toarray()).all()
        assert (features2.toarray() == features3.toarray()).all()
    
    def test_special_characters_handling(self):
        """Test that special characters are handled correctly."""
        payloads = pd.Series([
            "payload with spaces",
            "payload\twith\ttabs",
            "payload\nwith\nnewlines",
            "payload with unicode: café, naïve, 日本語"
        ])
        
        extractor = FeatureExtractor(max_features=100)
        feature_vectors = extractor.fit_transform(payloads)
        
        # Should not raise errors and produce valid output
        assert feature_vectors.shape[0] == 4
        assert feature_vectors.shape[1] > 0


class TestFeatureExtractorEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_single_payload(self):
        """Test fitting and transforming with single payload."""
        payloads = pd.Series(["single payload"])
        
        extractor = FeatureExtractor(max_features=100)
        feature_vectors = extractor.fit_transform(payloads)
        
        assert feature_vectors.shape[0] == 1
        assert feature_vectors.shape[1] > 0
    
    def test_very_long_payload(self):
        """Test handling of very long payloads."""
        payloads = pd.Series([
            "a" * 10000,  # Very long payload
            "b" * 10000
        ])
        
        extractor = FeatureExtractor(max_features=100)
        feature_vectors = extractor.fit_transform(payloads)
        
        assert feature_vectors.shape[0] == 2
        assert feature_vectors.shape[1] <= 100
    
    def test_empty_string_payload(self):
        """Test handling of empty string payloads."""
        payloads = pd.Series([
            "normal payload",
            "",  # Empty string
            "another payload"
        ])
        
        extractor = FeatureExtractor(max_features=100)
        feature_vectors = extractor.fit_transform(payloads)
        
        # Should handle empty strings gracefully
        assert feature_vectors.shape[0] == 3
    
    def test_max_features_larger_than_vocabulary(self):
        """Test when max_features is larger than actual vocabulary."""
        payloads = pd.Series([
            "aa",  # Very limited vocabulary
            "bb"
        ])
        
        extractor = FeatureExtractor(max_features=10000)
        feature_vectors = extractor.fit_transform(payloads)
        
        # Actual features should be less than max_features
        assert feature_vectors.shape[1] < 10000
