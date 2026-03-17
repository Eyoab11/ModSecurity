"""
Generate training_report.txt from existing evaluation_metrics.json and model_metadata.json.
"""

import json
from datetime import datetime
from pathlib import Path


def main():
    with open("models/evaluation_metrics.json") as f:
        eval_metrics = json.load(f)

    with open("models/model_metadata.json") as f:
        metadata = json.load(f)

    best_model = eval_metrics.get("best_model", "RandomForest")
    lib_versions = metadata.get("library_versions", {})
    training_date = metadata.get("training_date", datetime.now().isoformat())
    training_samples = metadata.get("training_samples", "N/A")
    test_samples = metadata.get("test_samples", "N/A")
    feature_count = metadata.get("feature_count", 5000)
    random_seed = metadata.get("random_seed", 42)

    algorithms = ["RandomForest", "SVM", "LogisticRegression"]

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
    lines.append(f"Max Features: {feature_count:,}" if isinstance(feature_count, int) else f"Max Features: {feature_count}")
    lines.append(f"Feature Vector Dimensionality: {feature_count}")
    lines.append("")
    lines.append("-" * 80)
    lines.append("MODEL TRAINING RESULTS")
    lines.append("-" * 80)

    for algo in algorithms:
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
            cm = m["confusion_matrix"]
            lines.append(f"  Confusion Matrix:")
            lines.append(f"    {cm}")

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
    for algo in algorithms:
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

    report = "\n".join(lines)
    Path("models/training_report.txt").write_text(report)
    print("Report saved to models/training_report.txt")
    print(report)


if __name__ == "__main__":
    main()
