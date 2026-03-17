"""
Script to train models individually to manage memory usage.
This avoids memory conflicts by training one model at a time.
"""

import sys
import gc
from pathlib import Path

from src.ml_core.data_loader import DataLoader
from src.ml_core.extractor import FeatureExtractor
from src.ml_core.trainer import ModelTrainer
from src.ml_core.logging_config import setup_logging


def train_single_model(algorithm_name, X_train, y_train, X_test, y_test):
    """Train a single model and save it."""
    logger = setup_logging()
    
    logger.info(f"\n{'='*80}")
    logger.info(f"TRAINING {algorithm_name.upper()}")
    logger.info(f"{'='*80}")
    
    # Create trainer with single algorithm
    trainer = ModelTrainer(algorithms=[algorithm_name], random_seed=42)
    
    try:
        # Train the algorithm
        model = trainer.train_algorithm(algorithm_name, X_train, y_train)
        
        if model is None:
            logger.error(f"Failed to train {algorithm_name}")
            return None
            
        # Evaluate the model
        metrics = trainer.evaluate_model(model, X_test, y_test)
        
        # Save the model
        model_filename = f"models/{algorithm_name.lower()}_model.pkl"
        trainer.save_model(model, model_filename)
        
        logger.info(f"{algorithm_name} training completed successfully!")
        logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Model saved to: {model_filename}")
        
        return {"model": model, "metrics": metrics}
        
    except Exception as e:
        logger.error(f"Error training {algorithm_name}: {str(e)}")
        return None
    finally:
        # Force garbage collection to free memory
        gc.collect()


def main():
    """Train each model individually."""
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("INDIVIDUAL MODEL TRAINING")
    logger.info("=" * 80)
    
    # Check if dataset exists
    data_path = "data/processed/master_fused_payloads.csv"
    if not Path(data_path).exists():
        logger.error(f"Dataset not found: {data_path}")
        logger.error("Please run the data exploration notebook first to generate the fused dataset.")
        return 1
    
    try:
        # Step 1: Load and prepare data (same as before)
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Loading and preparing data")
        logger.info("=" * 80)
        
        data_loader = DataLoader(data_path=data_path, test_size=0.2, random_seed=42)
        df = data_loader.load_and_validate()
        df_cleaned = data_loader.clean_data(df)
        train_df, test_df = data_loader.split_data(df_cleaned)
        
        # Step 2: Extract features (same as before)
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Extracting features")
        logger.info("=" * 80)
        
        feature_extractor = FeatureExtractor(max_features=5000, ngram_range=(2, 5))
        X_train = feature_extractor.fit_transform(train_df['payload'])
        X_test = feature_extractor.transform(test_df['payload'])
        y_train = train_df['label'].values
        y_test = test_df['label'].values
        
        logger.info(f"Training features shape: {X_train.shape}")
        logger.info(f"Test features shape: {X_test.shape}")
        
        # Save feature extractor
        feature_extractor.save("models/feature_extractor.pkl")
        
        # Step 3: Train models individually (skip already-trained ones)
        algorithms = ["RandomForest", "SVM", "LogisticRegression"]
        results = {}
        
        # Check which models are already trained
        for alg in algorithms:
            pkl_path = f"models/{alg.lower()}_model.pkl"
            if Path(pkl_path).exists():
                logger.info(f"Skipping {alg} — already trained ({pkl_path})")
                results[alg] = {"model": None, "metrics": None, "skipped": True}
        
        for algorithm in algorithms:
            logger.info(f"\n{'='*80}")
            logger.info(f"TRAINING {algorithm}")
            logger.info(f"{'='*80}")
            
            result = train_single_model(algorithm, X_train, y_train, X_test, y_test)
            if result:
                results[algorithm] = result
                logger.info(f"✓ {algorithm} completed successfully")
            else:
                logger.warning(f"✗ {algorithm} failed to train")
            
            # Force garbage collection between models
            gc.collect()
        
        # Step 4: Select best model and save metadata
        if results:
            logger.info("\n" + "=" * 80)
            logger.info("SELECTING BEST MODEL")
            logger.info("=" * 80)
            
            # Find best model by F1 score
            best_name = max(results.keys(), key=lambda k: results[k]['metrics']['f1_score'])
            best_model = results[best_name]['model']
            best_metrics = results[best_name]['metrics']
            
            # Save best model
            trainer = ModelTrainer()
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
            eval_metrics = {alg: result['metrics'] for alg, result in results.items()}
            eval_metrics['best_model'] = best_name
            trainer.save_evaluation_metrics(eval_metrics, best_name, "models/evaluation_metrics.json")
            
            # Summary
            logger.info("\n" + "=" * 80)
            logger.info("TRAINING COMPLETE")
            logger.info("=" * 80)
            logger.info(f"Successfully trained {len(results)} models:")
            for alg in results.keys():
                f1_score = results[alg]['metrics']['f1_score']
                logger.info(f"  - {alg}: F1-Score = {f1_score:.4f}")
            
            logger.info(f"\nBest Model: {best_name}")
            logger.info(f"Best F1-Score: {best_metrics['f1_score']:.4f}")
            
        else:
            logger.error("No models were successfully trained!")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())