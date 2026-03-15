"""
Demo script to test the ModelTrainer component.

This script demonstrates the complete workflow:
1. Load data using DataLoader
2. Extract features using FeatureExtractor
3. Train models using ModelTrainer
4. Save models and metadata
"""

import sys
from pathlib import Path

from src.ml_core.data_loader import DataLoader
from src.ml_core.extractor import FeatureExtractor
from src.ml_core.trainer import ModelTrainer
from src.ml_core.logging_config import setup_logging


def main():
    """Run the model training demo."""
    # Set up logging
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("MODEL TRAINER DEMO")
    logger.info("=" * 80)
    
    # Check if dataset exists
    data_path = "data/processed/master_fused_payloads.csv"
    if not Path(data_path).exists():
        logger.error(f"Dataset not found: {data_path}")
        logger.error("Please run the data exploration notebook first to generate the fused dataset.")
        return 1
    
    try:
        # Step 1: Load and prepare data
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Loading and preparing data")
        logger.info("=" * 80)
        
        data_loader = DataLoader(data_path=data_path, test_size=0.2, random_seed=42)
        
        # Load and validate
        df = data_loader.load_and_validate()
        
        # Clean data
        df_cleaned = data_loader.clean_data(df)
        
        # Split data
        train_df, test_df = data_loader.split_data(df_cleaned)
        
        # Get statistics
        data_loader.get_statistics(train_df, "Training Set")
        data_loader.get_statistics(test_df, "Test Set")
        
        # Step 2: Extract features
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Extracting features")
        logger.info("=" * 80)
        
        feature_extractor = FeatureExtractor(max_features=5000, ngram_range=(2, 5))
        
        # Fit and transform training data
        X_train = feature_extractor.fit_transform(train_df['payload'])
        
        # Transform test data
        X_test = feature_extractor.transform(test_df['payload'])
        
        # Get labels
        y_train = train_df['label'].values
        y_test = test_df['label'].values
        
        logger.info(f"Training features shape: {X_train.shape}")
        logger.info(f"Test features shape: {X_test.shape}")
        
        # Save feature extractor
        feature_extractor.save("models/feature_extractor.pkl")
        
        # Step 3: Train models
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: Training models")
        logger.info("=" * 80)
        
        trainer = ModelTrainer(
            algorithms=["RandomForest", "SVM", "LogisticRegression"],
            random_seed=42
        )
        
        # Train all algorithms
        results = trainer.train_all(X_train, y_train, X_test, y_test)
        
        # Step 4: Select best model
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Selecting best model")
        logger.info("=" * 80)
        
        best_name, best_model, best_metrics = trainer.select_best_model(results)
        
        # Step 5: Save models and metadata
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: Saving models and metadata")
        logger.info("=" * 80)
        
        # Save all trained models
        for algorithm, result in results.items():
            model_filename = f"models/{algorithm.lower()}_model.pkl"
            trainer.save_model(result['model'], model_filename)
        
        # Save best model
        trainer.save_model(best_model, "models/best_model.pkl")
        
        # Create and save metadata
        metadata = trainer.create_model_metadata(
            algorithm_name=best_name,
            metrics=best_metrics,
            feature_count=X_train.shape[1],
            training_samples=X_train.shape[0],
            test_samples=X_test.shape[0],
            version=1
        )
        trainer.save_metadata(metadata, "models/model_metadata.json")
        
        # Save evaluation metrics
        trainer.save_evaluation_metrics(results, best_name, "models/evaluation_metrics.json")
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Best Model: {best_name}")
        logger.info(f"F1-Score: {best_metrics['f1_score']:.4f}")
        logger.info(f"Accuracy: {best_metrics['accuracy']:.4f}")
        logger.info(f"False Positive Rate: {best_metrics['false_positive_rate']:.4f}")
        logger.info("\nArtifacts saved to models/ directory:")
        logger.info("  - feature_extractor.pkl")
        logger.info("  - best_model.pkl")
        logger.info("  - model_metadata.json")
        logger.info("  - evaluation_metrics.json")
        for algorithm in results.keys():
            logger.info(f"  - {algorithm.lower()}_model.pkl")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
