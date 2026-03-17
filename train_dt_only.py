"""
Train only DecisionTree — RandomForest, SVM, and LogisticRegression already done.
Updates evaluation_metrics.json, model_metadata.json, and training_report.txt when complete.
"""

import sys
import gc
import json
import shutil
from pathlib import Path
from datetime import datetime

from src.ml_core.data_loader import DataLoader
from src.ml_core.extractor import FeatureExtractor
from src.ml_core.trainer import ModelTrainer
from src.ml_core.logging_config import setup_logging


def main():
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("TRAINING DecisionTree")
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

        # Train DecisionTree
        trainer = ModelTrainer(algorithms=["DecisionTree"], random_seed=42)
        model = trainer.train_algorithm("DecisionTree", X_train, y_train)

        if model is None:
            logger.error("DecisionTree training failed")
            return 1

        metrics = trainer.evaluate_model(model, X_test, y_test)
        trainer.save_model(model, "models/decisiontree_model.pkl")

        logger.info("DecisionTree training complete!")
        logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")

        gc.collect()

        # Update evaluation_metrics.json
        with open("models/evaluation_metrics.json", "r") as f:
            eval_metrics = json.load(f)

        eval_metrics["DecisionTree"] = metrics

        # Re-determine best model across all 4
        model_names = ["RandomForest", "SVM", "LogisticRegression", "DecisionTree"]
        best_name = max(
            [m for m in model_names if m in eval_metrics and isinstance(eval_metrics[m], dict)],
            key=lambda k: eval_metrics[k].get("f1_score", 0)
        )
        eval_metrics["best_model"] = best_name

        with open("models/evaluation_metrics.json", "w") as f:
            json.dump(eval_metrics, f, indent=2)
        logger.info("Updated evaluation_metrics.json")

        # Copy best model
        best_pkl = f"models/{best_name.lower()}_model.pkl"
        shutil.copy(best_pkl, "models/best_model.pkl")
        logger.info(f"Best model: {best_name} (F1={eval_metrics[best_name]['f1_score']:.4f})")

        # Update model_metadata.json
        with open("models/model_metadata.json", "r") as f:
            metadata = json.load(f)

        metadata["algorithm_name"] = best_name
        metadata["algorithm"] = best_name
        metadata["metrics"] = eval_metrics[best_name]
        metadata["all_trained_models"] = model_names

        with open("models/model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Updated model_metadata.json")

        # Regenerate training report
        _regenerate_report(eval_metrics, metadata, model_names)
        logger.info("Updated training_report.txt")

        return 0

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return 1


def _regenerate_report(eval_metrics, metadata, model_names):
    best_model = eval_metrics.get("best_model", "RandomForest")
    lib_versions = metadata.get("library_versions", {})
    training_date = metadata.get("training_date", datetime.now().isoformat())
    training_samples = metadata.get("training_samples", "N/A")
    test_samples = metadata.get("test_samples", "N/A")
    feature_count = metadata.get("feature_count", 5000)
    random_seed = metadata.get("random_seed", 42)

    lines = []
    lines.append("=" * 80)
    lines.append("ML MODEL TRAINING REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Training Date: {training_date}")
    lines.append("")
    lines.append("-" * 80)
    lines.append("DATASET STATISTICS")
    lines.append("-" * 80)
    lines.append(f"Training Set: {training_samples:,}" if isinstance(training_samples, int) else f"Training Set: {training_samples}")
    lines.append(f"Test Set:     {test_samples:,}" if isinstance(test_samples, int) else f"Test Set:     {test_samples}")
    lines.append("")
    lines.append("-" * 80)
    lines.append("FEATURE EXTRACTION")
    lines.append("-" * 80)
    lines.append("Method: TF-IDF with Character N-grams")
    lines.append("N-gram Range: (2, 5)")
    lines.append(f"Max Features: {feature_count}")
    lines.append("")
    lines.append("-" * 80)
    lines.append("MODEL TRAINING RESULTS")
    lines.append("-" * 80)

    for algo in model_names:
        if algo not in eval_metrics or not isinstance(eval_metrics[algo], dict):
            continue
        m = eval_metrics[algo]
        lines.append("")
        lines.append(f"Algorithm: {algo}")
        if "training_time_seconds" in m:
            lines.append(f"  Training Time:       {m['training_time_seconds']:.1f} seconds")
        if "accuracy" in m:
            lines.append(f"  Accuracy:            {m['accuracy']*100:.2f}%")
        if "precision" in m:
            lines.append(f"  Precision:           {m['precision']*100:.2f}%")
        if "recall" in m:
            lines.append(f"  Recall:              {m['recall']*100:.2f}%")
        if "f1_score" in m:
            lines.append(f"  F1-Score:            {m['f1_score']*100:.2f}%")
        if "false_positive_rate" in m:
            lines.append(f"  False Positive Rate: {m['false_positive_rate']*100:.2f}%")
        if "confusion_matrix" in m:
            lines.append(f"  Confusion Matrix:    {m['confusion_matrix']}")

    lines.append("")
    lines.append("-" * 80)
    lines.append("BEST MODEL SELECTION")
    lines.append("-" * 80)
    best_f1 = eval_metrics.get(best_model, {}).get("f1_score", 0)
    lines.append(f"Selected Algorithm: {best_model}")
    lines.append(f"Selection Criteria: Highest F1-Score ({best_f1*100:.2f}%)")
    lines.append(f"Model Saved: models/best_model.pkl")
    lines.append("")
    lines.append("-" * 80)
    lines.append("ARTIFACTS GENERATED")
    lines.append("-" * 80)
    lines.append("  models/feature_extractor.pkl")
    for algo in model_names:
        lines.append(f"  models/{algo.lower()}_model.pkl")
    lines.append("  models/best_model.pkl")
    lines.append("  models/model_metadata.json")
    lines.append("  models/evaluation_metrics.json")
    lines.append("  models/training_report.txt")
    lines.append("")
    lines.append("-" * 80)
    lines.append("REPRODUCIBILITY INFORMATION")
    lines.append("-" * 80)
    lines.append(f"Random Seed: {random_seed}")
    lines.append("Library Versions:")
    for lib, ver in lib_versions.items():
        lines.append(f"  - {lib}: {ver}")
    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    Path("models/training_report.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
