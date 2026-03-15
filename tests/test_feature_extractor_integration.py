"""
Integration tests for FeatureExtractor with DataLoader.

Tests the integration between DataLoader and FeatureExtractor components.
"""

import pytest
import pandas as pd
import tempfile
import os

from src.ml_core.data_loader import DataLoader
from src.ml_core.extractor import FeatureExtractor


class TestFeatureExtractorDataLoaderIntegration:
    """Integration tests between FeatureExtractor and DataLoader."""
    
    def test_feature_extraction_on_loaded_data(self):
        """Test feature extraction on data loaded by DataLoader."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("payload,label,source_file\n")
            f.write("SELECT * FROM users,1,file1.log\n")
            f.write("<script>alert('xss')</script>,1,file2.log\n")
            f.write("../../etc/passwd,1,file3.log\n")
            f.write("normal request,0,file4.log\n")
            f.write("GET /index.html,0,file5.log\n")
            f.write("POST /api/data,0,file6.log\n")
            temp_path = f.name
        
        try:
            # Load data using DataLoader
            loader = DataLoader(data_path=temp_path, test_size=0.3, random_seed=42)
            df = loader.load_and_validate()
            df = loader.clean_data(df)
            train_df, test_df = loader.split_data(df)
            
            # Extract features using FeatureExtractor
            extractor = FeatureExtractor(max_features=100)
            
            # Fit on training data
            X_train = extractor.fit_transform(train_df['payload'])
            
            # Transform test data
            X_test = extractor.transform(test_df['payload'])
            
            # Verify shapes
            assert X_train.shape[0] == len(train_df)
            assert X_test.shape[0] == len(test_df)
            assert X_train.shape[1] == X_test.shape[1]  # Same number of features
            assert X_train.shape[1] <= 100  # At most max_features
            
        finally:
            # Clean up
            os.unlink(temp_path)
    
    def test_feature_extraction_preserves_label_alignment(self):
        """Test that feature extraction preserves alignment with labels."""
        # Create a temporary CSV file with enough samples for stratified split
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("payload,label,source_file\n")
            f.write("malicious payload 1,1,file1.log\n")
            f.write("benign payload 1,0,file2.log\n")
            f.write("malicious payload 2,1,file3.log\n")
            f.write("benign payload 2,0,file4.log\n")
            f.write("malicious payload 3,1,file5.log\n")
            f.write("benign payload 3,0,file6.log\n")
            f.write("malicious payload 4,1,file7.log\n")
            f.write("benign payload 4,0,file8.log\n")
            temp_path = f.name
        
        try:
            # Load data
            loader = DataLoader(data_path=temp_path, test_size=0.25, random_seed=42)
            df = loader.load_and_validate()
            df = loader.clean_data(df)
            train_df, test_df = loader.split_data(df)
            
            # Extract features
            extractor = FeatureExtractor(max_features=50)
            X_train = extractor.fit_transform(train_df['payload'])
            X_test = extractor.transform(test_df['payload'])
            
            # Get labels
            y_train = train_df['label'].values
            y_test = test_df['label'].values
            
            # Verify alignment (number of samples matches)
            assert X_train.shape[0] == len(y_train)
            assert X_test.shape[0] == len(y_test)
            
        finally:
            os.unlink(temp_path)
    
    def test_save_and_load_with_real_data(self):
        """Test saving and loading extractor with real data flow."""
        # Create a temporary CSV file with enough samples
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("payload,label,source_file\n")
            f.write("SELECT * FROM users WHERE id=1,1,file1.log\n")
            f.write("<img src=x onerror=alert(1)>,1,file2.log\n")
            f.write("normal GET request,0,file3.log\n")
            f.write("POST /api/data,0,file4.log\n")
            f.write("SELECT id FROM accounts,1,file5.log\n")
            f.write("GET /index.html,0,file6.log\n")
            temp_path = f.name
        
        try:
            # Load data
            loader = DataLoader(data_path=temp_path, test_size=0.33, random_seed=42)
            df = loader.load_and_validate()
            df = loader.clean_data(df)
            train_df, test_df = loader.split_data(df)
            
            # Create and fit extractor
            extractor1 = FeatureExtractor(max_features=50)
            X_train1 = extractor1.fit_transform(train_df['payload'])
            X_test1 = extractor1.transform(test_df['payload'])
            
            # Save extractor
            with tempfile.TemporaryDirectory() as tmpdir:
                extractor_path = os.path.join(tmpdir, "feature_extractor.pkl")
                extractor1.save(extractor_path)
                
                # Load extractor
                extractor2 = FeatureExtractor.load(extractor_path)
                
                # Transform with loaded extractor
                X_train2 = extractor2.transform(train_df['payload'])
                X_test2 = extractor2.transform(test_df['payload'])
                
                # Verify identical transformations
                assert (X_train1.toarray() == X_train2.toarray()).all()
                assert (X_test1.toarray() == X_test2.toarray()).all()
        
        finally:
            os.unlink(temp_path)
