"""
Unit tests for DataLoader component
"""

import pytest
import pandas as pd
from pathlib import Path
from src.ml_core.data_loader import DataLoader
from src.ml_core.exceptions import DataLoadError


class TestDataLoader:
    """Test suite for DataLoader component"""
    
    def test_load_valid_dataset(self):
        """Test loading a valid CSV file with correct schema."""
        loader = DataLoader("data/processed/master_fused_payloads.csv")
        df = loader.load_and_validate()
        
        assert len(df) > 0
        assert all(col in df.columns for col in ['payload', 'label', 'source_file'])
    
    def test_missing_file_raises_error(self):
        """Test that missing file raises DataLoadError."""
        loader = DataLoader("nonexistent.csv")
        with pytest.raises(DataLoadError):
            loader.load_and_validate()
    
    def test_label_normalization(self):
        """Test that various label formats are normalized correctly."""
        loader = DataLoader()
        
        # Test cases: (input, expected_output)
        test_cases = [
            ("0", 0), ("1", 1),
            ("benign", 0), ("malicious", 1),
            ("normal", 0), ("attack", 1),
            (0, 0), (1, 1),
            (0.0, 0), (1.0, 1)
        ]
        
        for input_label, expected in test_cases:
            result = loader._normalize_label(input_label)
            assert result == expected, f"Failed for input {input_label}: expected {expected}, got {result}"
    
    def test_clean_data_removes_nulls(self):
        """Test that clean_data removes records with missing payload or label."""
        loader = DataLoader()
        
        # Create test DataFrame with some null values
        test_df = pd.DataFrame({
            'payload': ['test1', None, 'test3', 'test4'],
            'label': [0, 1, None, 1],
            'source_file': ['file1', 'file2', 'file3', 'file4']
        })
        
        cleaned_df = loader.clean_data(test_df)
        
        # Should only have 2 records (test1 and test4)
        assert len(cleaned_df) == 2
        assert cleaned_df['payload'].notna().all()
        assert cleaned_df['label'].notna().all()
    
    def test_split_data_ratio(self):
        """Test that split_data produces correct train/test ratio."""
        loader = DataLoader(test_size=0.2)
        
        # Create test DataFrame
        test_df = pd.DataFrame({
            'payload': [f'payload_{i}' for i in range(100)],
            'label': [i % 2 for i in range(100)],  # Alternating 0 and 1
            'source_file': [f'file_{i}' for i in range(100)]
        })
        
        train_df, test_df = loader.split_data(test_df)
        
        # Check sizes (80/20 split)
        assert len(train_df) == 80
        assert len(test_df) == 20
        assert len(train_df) + len(test_df) == 100
    
    def test_split_reproducibility(self):
        """Test that splits with same seed produce identical results."""
        # Create test DataFrame
        test_df = pd.DataFrame({
            'payload': [f'payload_{i}' for i in range(100)],
            'label': [i % 2 for i in range(100)],
            'source_file': [f'file_{i}' for i in range(100)]
        })
        
        # First split
        loader1 = DataLoader(random_seed=42)
        train1, test1 = loader1.split_data(test_df)
        
        # Second split with same seed
        loader2 = DataLoader(random_seed=42)
        train2, test2 = loader2.split_data(test_df)
        
        # Assert identical splits
        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(test1, test2)
    
    def test_get_statistics(self):
        """Test that get_statistics calculates correct metrics."""
        loader = DataLoader()
        
        # Create test DataFrame with known distribution
        test_df = pd.DataFrame({
            'payload': [f'payload_{i}' for i in range(100)],
            'label': [0] * 60 + [1] * 40,  # 60% benign, 40% malicious
            'source_file': [f'file_{i}' for i in range(100)]
        })
        
        stats = loader.get_statistics(test_df, "Test Dataset")
        
        assert stats['total_records'] == 100
        assert stats['label_counts'][0] == 60
        assert stats['label_counts'][1] == 40
        assert abs(stats['label_distribution'][0] - 60.0) < 0.1
        assert abs(stats['label_distribution'][1] - 40.0) < 0.1
    
    def test_empty_dataset_after_cleaning_raises_error(self):
        """Test that empty dataset after cleaning raises DataLoadError."""
        loader = DataLoader()
        
        # Create DataFrame with all null values
        test_df = pd.DataFrame({
            'payload': [None, None],
            'label': [None, None],
            'source_file': ['file1', 'file2']
        })
        
        with pytest.raises(DataLoadError, match="empty after removing"):
            loader.clean_data(test_df)
