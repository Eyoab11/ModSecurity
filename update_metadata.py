"""
Update evaluation_metrics.json and model_metadata.json to reflect
all trained models (RandomForest + LogisticRegression).
Also copies the best model (RandomForest) to best_model.pkl.
"""

import pickle
import json
import shutil
from pathlib import Path

# Known metrics from training logs
results = {
    "RandomForest": {
        "accuracy": 0.9927,
        "f1_score": 0.9741,
        "training_time_seconds": 1121.31
    },
    "LogisticRegression": {
        "accuracy": 0.9647289479529848,
        "precision": 0.8060965283657917,
        "recall": 0.9850313858039594,
        "f1_score": 0.8866261020737614,
        "false_positive_rate": 0.038576434122456316,
        "confusion_matrix": [[85609, 3435], [217, 14280]],
        "true_positives": 14280,
        "false_positives": 3435,
        "true_negatives": 85609,
        "false_negatives": 217,
        "training_time_seconds": 37.0467963218689
    }
}

best_model_name = "RandomForest"

# Update evaluation_metrics.json
eval_metrics = dict(results)
eval_metrics["best_model"] = best_model_name

with open("models/evaluation_metrics.json", "w") as f:
    json.dump(eval_metrics, f, indent=2)
print("Updated evaluation_metrics.json")

# Copy best model
shutil.copy("models/randomforest_model.pkl", "models/best_model.pkl")
print("Copied randomforest_model.pkl -> best_model.pkl")

# Update model_metadata.json
with open("models/model_metadata.json", "r") as f:
    metadata = json.load(f)

metadata["algorithm"] = best_model_name
metadata["metrics"] = results[best_model_name]
metadata["all_trained_models"] = list(results.keys())

with open("models/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("Updated model_metadata.json")

print("\nDone. Models available:")
for name in results:
    print(f"  - {name}: F1={results[name]['f1_score']:.4f}")
print(f"\nBest model: {best_model_name}")
