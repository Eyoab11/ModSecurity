"""
Demo script for FeatureExtractor component.

Demonstrates the usage of FeatureExtractor for converting payload text
to numerical feature vectors using TF-IDF vectorization.
"""

import pandas as pd
from pathlib import Path

from src.ml_core.extractor import FeatureExtractor
from src.ml_core.logging_config import setup_logging


def main():
    """Demonstrate FeatureExtractor functionality."""
    # Set up logging
    logger = setup_logging()
    logger.info("=" * 70)
    logger.info("FeatureExtractor Demo")
    logger.info("=" * 70)
    
    # Sample payloads (mix of malicious and benign)
    train_payloads = pd.Series([
        "SELECT * FROM users WHERE id=1",
        "SELECT password FROM accounts",
        "<script>alert('XSS')</script>",
        "<img src=x onerror=alert(1)>",
        "../../etc/passwd",
        "../../../windows/system32",
        "normal GET /index.html HTTP/1.1",
        "POST /api/data HTTP/1.1",
        "GET /images/logo.png HTTP/1.1",
        "PUT /api/update HTTP/1.1"
    ])
    
    test_payloads = pd.Series([
        "SELECT id FROM users",  # SQL injection variant
        "<iframe src=javascript:alert(1)>",  # XSS variant
        "GET /about.html HTTP/1.1"  # Benign request
    ])
    
    logger.info("\n" + "=" * 70)
    logger.info("Step 1: Initialize FeatureExtractor")
    logger.info("=" * 70)
    extractor = FeatureExtractor(max_features=100, ngram_range=(2, 5))
    logger.info(f"Max features: {extractor.max_features}")
    logger.info(f"N-gram range: {extractor.ngram_range}")
    
    logger.info("\n" + "=" * 70)
    logger.info("Step 2: Fit on training payloads")
    logger.info("=" * 70)
    logger.info(f"Training on {len(train_payloads)} payloads:")
    for i, payload in enumerate(train_payloads, 1):
        logger.info(f"  {i}. {payload[:50]}...")
    
    X_train = extractor.fit_transform(train_payloads)
    logger.info(f"\nTraining feature matrix shape: {X_train.shape}")
    logger.info(f"  - {X_train.shape[0]} samples")
    logger.info(f"  - {X_train.shape[1]} features")
    logger.info(f"  - Sparsity: {(1 - X_train.nnz / (X_train.shape[0] * X_train.shape[1])) * 100:.2f}%")
    
    logger.info("\n" + "=" * 70)
    logger.info("Step 3: Transform test payloads")
    logger.info("=" * 70)
    logger.info(f"Transforming {len(test_payloads)} test payloads:")
    for i, payload in enumerate(test_payloads, 1):
        logger.info(f"  {i}. {payload[:50]}...")
    
    X_test = extractor.transform(test_payloads)
    logger.info(f"\nTest feature matrix shape: {X_test.shape}")
    logger.info(f"  - {X_test.shape[0]} samples")
    logger.info(f"  - {X_test.shape[1]} features")
    
    logger.info("\n" + "=" * 70)
    logger.info("Step 4: Examine feature names (sample)")
    logger.info("=" * 70)
    feature_names = extractor.get_feature_names()
    logger.info(f"Total features: {len(feature_names)}")
    logger.info(f"Sample feature names (first 20):")
    for i, name in enumerate(feature_names[:20], 1):
        logger.info(f"  {i}. '{name}'")
    
    logger.info("\n" + "=" * 70)
    logger.info("Step 5: Save and load extractor")
    logger.info("=" * 70)
    
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    save_path = "models/demo_feature_extractor.pkl"
    extractor.save(save_path)
    logger.info(f"Extractor saved to: {save_path}")
    
    # Load extractor
    loaded_extractor = FeatureExtractor.load(save_path)
    logger.info(f"Extractor loaded from: {save_path}")
    
    # Verify loaded extractor produces same results
    X_test_loaded = loaded_extractor.transform(test_payloads)
    identical = (X_test.toarray() == X_test_loaded.toarray()).all()
    logger.info(f"Loaded extractor produces identical results: {identical}")
    
    logger.info("\n" + "=" * 70)
    logger.info("Step 6: Feature vector analysis")
    logger.info("=" * 70)
    
    # Analyze a specific test payload
    test_idx = 0
    test_payload = test_payloads.iloc[test_idx]
    test_vector = X_test[test_idx]
    
    logger.info(f"Analyzing payload: '{test_payload}'")
    logger.info(f"Feature vector shape: {test_vector.shape}")
    logger.info(f"Non-zero features: {test_vector.nnz}")
    
    # Get top features for this payload
    if test_vector.nnz > 0:
        vector_array = test_vector.toarray().flatten()
        top_indices = vector_array.argsort()[-10:][::-1]
        logger.info(f"\nTop 10 features for this payload:")
        for i, idx in enumerate(top_indices, 1):
            if vector_array[idx] > 0:
                logger.info(f"  {i}. '{feature_names[idx]}': {vector_array[idx]:.4f}")
    
    logger.info("\n" + "=" * 70)
    logger.info("Demo completed successfully!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
