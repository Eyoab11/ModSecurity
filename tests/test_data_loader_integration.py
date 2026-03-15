"""
Integration tests for DataLoader component with real dataset
"""

import pytest
from src.ml_core.data_loader import DataLoader


class TestDataLoaderIntegration:
    """Integration test suite for DataLoader with actual dataset"""
    
    def test_complete_workflow(self):
        """Test complete DataLoader workflow with real dataset."""
        # Initialize DataLoader
        loader = DataLoader(
            data_path="data/processed/master_fused_payloads.csv",
            test_size=0.2,
            random_seed=42
        )
        
        # Load and validate
        df = loader.load_and_validate()
        assert len(df) > 0
        assert 'payload' in df.columns
        assert 'label' in df.columns
        assert 'source_file' in df.columns
        
        # Get initial statistics
        initial_stats = loader.get_statistics(df, "Initial Dataset")
        assert initial_stats['total_records'] > 0
        
        # Clean data
        df_cleaned = loader.clean_data(df)
        assert len(df_cleaned) > 0
        assert df_cleaned['payload'].notna().all()
        assert df_cleaned['label'].notna().all()
        
        # Verify labels are normalized to 0 or 1
        unique_labels = df_cleaned['label'].unique()
        assert all(label in [0, 1] for label in unique_labels)
        
        # Split data
        train_df, test_df = loader.split_data(df_cleaned)
        
        # Verify split sizes
        total_size = len(df_cleaned)
        expected_test_size = int(total_size * 0.2)
        expected_train_size = total_size - expected_test_size
        
        # Allow for ±1 record due to rounding
        assert abs(len(train_df) - expected_train_size) <= 1
        assert abs(len(test_df) - expected_test_size) <= 1
        assert len(train_df) + len(test_df) == total_size
        
        # Get statistics for train and test sets
        train_stats = loader.get_statistics(train_df, "Training Set")
        test_stats = loader.get_statistics(test_df, "Test Set")
        
        assert train_stats['total_records'] == len(train_df)
        assert test_stats['total_records'] == len(test_df)
        
        # Verify stratification preserved label distribution (within ±5%)
        if 0 in initial_stats['label_distribution'] and 1 in initial_stats['label_distribution']:
            original_dist_0 = initial_stats['label_distribution'][0]
            train_dist_0 = train_stats['label_distribution'].get(0, 0)
            test_dist_0 = test_stats['label_distribution'].get(0, 0)
            
            # Check that distributions are similar (within 5 percentage points)
            assert abs(train_dist_0 - original_dist_0) < 5.0
            assert abs(test_dist_0 - original_dist_0) < 5.0
