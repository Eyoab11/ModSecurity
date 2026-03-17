"""
Train only SVM (LinearSVC) — RandomForest and LogisticRegression already done.
Updates evaluation_metrics.json and model_metadata.json when complete.
"""

import sys
import gc
import json
import shutil
from pathlib import Path

from src.ml_core.data_loader import DataLoader
from src.ml_core.extractor import FeatureExtractor
from src.ml_core.trainer import ModelTrainer
from src.ml_core.logging_config import setup_logging


def main():
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("TRAINING SVM (LinearSVC)")
    logger.info("=" * 80)

    data_path = "data/processed/master_fused_payloads.csv"
    if not Path(data_path).exists():
        logger.error(f"Dataset not found: {data_path}")
        return 1

    try:
        # Load data
        data_loader = DataLoader(data_path=data_path, test_size=0.2, random_seed=42)
        df = data_loader.load_and_validate()
        df_cleaned = data_loader.clean_data(df)
        train_df, test_df = data_loader.split_data(df_cleaned)

        # Load the already-saved feature extractor
        logger.info("Loading saved feature extractor...")
        feature_extractor = FeatureExtractor.load("models/feature_extractor.pkl")

        X_train = feature_extractor.transform(train_df['payload'])
        X_test = feature_extractor.transform(test_df['payload'])
        y_train = train_df['label'].values
        y_test = test_df['label'].values

        logger.info(f"Training features shape: {X_train.shape}")

        # Train SVM
        trainer = ModelTrainer(algorithms=["SVM"], random_seed=42)
        model = trainer.train_algorithm("SVM", X_train, y_train)

        if model is None:
            logger.error("SVM training failed")
            return 1

        metrics = trainer.evaluate_model(model, X_test, y_test)
        trainer.save_model(model, "models/svm_model.pkl")

        logger.info(f"SVM training complete!")
        logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")

        gc.collect()

        # Update evaluation_metrics.json with all 3 models
        with open("models/evaluation_metrics.json", "r") as f:
            eval_metrics = json.load(f)

        eval_metrics["SVM"] = metrics

        # Re-determine best model
        model_names = ["RandomForest", "SVM", "LogisticRegression"]
        best_name = max(
            [m for m in model_names if m in eval_metrics and isinstance(eval_metrics[m], dict)],
            key=lambda k: eval_metrics[k].get("f1_score", 0)
        )
        eval_metrics["best_model"] = best_name

        with open("models/evaluation_metrics.json", "w") as f:
            json.dump(eval_metrics, f, indent=2)
        logger.info(f"Updated evaluation_metrics.json")

        # Copy best model
        best_pkl = f"models/{best_name.lower()}_model.pkl"
        shutil.copy(best_pkl, "models/best_model.pkl")
        logger.info(f"Best model: {best_name} (F1={eval_metrics[best_name]['f1_score']:.4f})")

        # Update model_metadata.json
        with open("models/model_metadata.json", "r") as f:
            metadata = json.load(f)

        metadata["algorithm"] = best_name
        metadata["metrics"] = eval_metrics[best_name]
        metadata["all_trained_models"] = model_names

        with open("models/model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Updated model_metadata.json")

        return 0

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
