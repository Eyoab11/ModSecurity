"""
Model Training Module

Provides the ModelTrainer component for training multiple ML algorithms,
evaluating their performance, selecting the best model, and serializing
trained models and metadata.
"""

import time
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from .logging_config import get_logger
from .exceptions import ModelTrainingError, AllAlgorithmsFailedError


class ModelTrainer:
    """
    Model trainer that trains multiple ML algorithms and selects the best performer.
    
    Implements an algorithm tournament approach where multiple algorithms are trained
    and evaluated, with the best model selected based on F1-score and false positive rate.
    
    Attributes:
        algorithms: List of algorithm names to train
        random_seed: Random seed for reproducibility
        logger: Logger instance for this component
    """
    
    def __init__(self, algorithms: List[str] = None, random_seed: int = 42):
        """
        Initialize trainer with list of algorithms to train.
        
        Args:
            algorithms: List of algorithm names (default: ["RandomForest", "SVM", "LogisticRegression"])
            random_seed: Random seed for reproducibility (default: 42)
        """
        if algorithms is None:
            algorithms = ["RandomForest", "SVM", "LogisticRegression"]
        
        self.algorithms = algorithms
        self.random_seed = random_seed
        self.logger = get_logger("ModelTrainer")
        
        self.logger.info(f"ModelTrainer initialized with algorithms: {', '.join(algorithms)}")
        self.logger.info(f"Random seed: {random_seed}")
    
    def _create_algorithm(self, algorithm: str) -> Any:
        """
        Factory method to instantiate an algorithm with proper configuration.
        
        Args:
            algorithm: Name of the algorithm
            
        Returns:
            Configured algorithm instance
            
        Raises:
            ValueError: If algorithm name is not recognized
        """
        algorithm_map = {
            "RandomForest": lambda: RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=self.random_seed,
                n_jobs=-1  # Use all available cores
            ),
            "SVM": lambda: SVC(
                kernel='rbf',
                class_weight='balanced',
                random_state=self.random_seed,
                probability=True  # Enable probability estimates
            ),
            "LogisticRegression": lambda: LogisticRegression(
                penalty='l2',
                class_weight='balanced',
                random_state=self.random_seed,
                max_iter=1000,
                solver='lbfgs'
            )
        }
        
        if algorithm not in algorithm_map:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        return algorithm_map[algorithm]()
    
    def train_algorithm(self, algorithm: str, X_train, y_train) -> Optional[Any]:
        """
        Train a single algorithm.
        
        Args:
            algorithm: Name of the algorithm to train
            X_train: Training feature matrix
            y_train: Training labels
            
        Returns:
            Trained model or None on failure
            
        Raises:
            ModelTrainingError: If algorithm training fails
        """
        self.logger.info(f"Training {algorithm}...")
        
        try:
            # Create algorithm instance
            model = self._create_algorithm(algorithm)
            
            # Train and measure time
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            self.logger.info(f"{algorithm} training completed in {training_time:.2f} seconds")
            
            return model
            
        except Exception as e:
            error_msg = f"Failed to train {algorithm}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise ModelTrainingError(error_msg)
    
    def evaluate_model(self, model: Any, X_test, y_test) -> Dict[str, float]:
        """
        Evaluate model and return metrics dict.
        
        Args:
            model: Trained model to evaluate
            X_test: Test feature matrix
            y_test: Test labels
            
        Returns:
            Dictionary containing accuracy, precision, recall, f1_score, 
            false_positive_rate, and confusion_matrix
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Extract confusion matrix components
        # cm[0,0] = TN, cm[0,1] = FP, cm[1,0] = FN, cm[1,1] = TP
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate false positive rate
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'false_positive_rate': float(false_positive_rate),
            'confusion_matrix': cm.tolist(),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        }
        
        return metrics
    
    def train_all(self, X_train, y_train, X_test, y_test) -> Dict[str, Any]:
        """
        Train all algorithms and return results dict.
        
        Args:
            X_train: Training feature matrix
            y_train: Training labels
            X_test: Test feature matrix
            y_test: Test labels
            
        Returns:
            Dictionary mapping algorithm names to their results (model, metrics, training_time)
            
        Raises:
            AllAlgorithmsFailedError: If all algorithms fail to train
        """
        results = {}
        
        for algorithm in self.algorithms:
            try:
                # Train algorithm
                start_time = time.time()
                model = self.train_algorithm(algorithm, X_train, y_train)
                training_time = time.time() - start_time
                
                # Evaluate model
                metrics = self.evaluate_model(model, X_test, y_test)
                metrics['training_time_seconds'] = float(training_time)
                
                # Store results
                results[algorithm] = {
                    'model': model,
                    'metrics': metrics
                }
                
                # Log metrics
                self.logger.info(f"{algorithm} Evaluation Metrics:")
                self.logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
                self.logger.info(f"  Precision: {metrics['precision']:.4f}")
                self.logger.info(f"  Recall: {metrics['recall']:.4f}")
                self.logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
                self.logger.info(f"  False Positive Rate: {metrics['false_positive_rate']:.4f}")
                
            except ModelTrainingError as e:
                self.logger.warning(f"Skipping {algorithm} due to training failure: {str(e)}")
                continue
            except Exception as e:
                self.logger.error(f"Unexpected error processing {algorithm}: {str(e)}", exc_info=True)
                continue
        
        # Check if at least one algorithm succeeded
        if not results:
            error_msg = "All algorithms failed to train"
            self.logger.error(error_msg, exc_info=True)
            raise AllAlgorithmsFailedError(error_msg)
        
        self.logger.info(f"Successfully trained {len(results)} out of {len(self.algorithms)} algorithms")
        
        return results
    
    def select_best_model(self, results: Dict[str, Any]) -> Tuple[str, Any, Dict]:
        """
        Select best model by F1-score. Returns (name, model, metrics).
        
        Selection criteria:
        1. Highest F1-score
        2. If tied, lowest false positive rate
        3. If still tied, alphabetical order
        
        Args:
            results: Dictionary of training results from train_all()
            
        Returns:
            Tuple of (algorithm_name, model, metrics)
        """
        self.logger.info("Selecting best model...")
        
        # Sort algorithms by F1-score (descending), then FPR (ascending), then name (ascending)
        sorted_algorithms = sorted(
            results.items(),
            key=lambda x: (
                -x[1]['metrics']['f1_score'],  # Negative for descending
                x[1]['metrics']['false_positive_rate'],  # Ascending
                x[0]  # Alphabetical
            )
        )
        
        # Select the best
        best_name, best_result = sorted_algorithms[0]
        best_model = best_result['model']
        best_metrics = best_result['metrics']
        
        self.logger.info(f"Best model selected: {best_name}")
        self.logger.info(f"  F1-Score: {best_metrics['f1_score']:.4f}")
        self.logger.info(f"  False Positive Rate: {best_metrics['false_positive_rate']:.4f}")
        
        return best_name, best_model, best_metrics
    
    def save_model(self, model: Any, path: str) -> None:
        """
        Serialize model to disk using joblib.
        
        Args:
            model: Trained model to save
            path: File path to save model (e.g., models/random_forest_model.pkl)
            
        Raises:
            ModelTrainingError: If serialization fails
        """
        try:
            # Create directory if it doesn't exist
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save model
            joblib.dump(model, path)
            self.logger.info(f"Model saved to {path}")
            
        except Exception as e:
            error_msg = f"Failed to save model to {path}: {e}"
            self.logger.error(error_msg, exc_info=True)
            raise ModelTrainingError(error_msg)
    
    def save_metadata(self, metadata: Dict, path: str) -> None:
        """
        Save model metadata to JSON file.
        
        Args:
            metadata: Dictionary containing model metadata
            path: File path to save metadata (e.g., models/model_metadata.json)
            
        Raises:
            ModelTrainingError: If saving fails
        """
        try:
            # Create directory if it doesn't exist
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save metadata as JSON
            with open(path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Metadata saved to {path}")
            
        except Exception as e:
            error_msg = f"Failed to save metadata to {path}: {e}"
            self.logger.error(error_msg, exc_info=True)
            raise ModelTrainingError(error_msg)
    
    def save_evaluation_metrics(self, results: Dict[str, Any], best_model_name: str, path: str) -> None:
        """
        Save evaluation metrics for all algorithms to JSON file.
        
        Args:
            results: Dictionary of training results from train_all()
            best_model_name: Name of the best model
            path: File path to save metrics (e.g., models/evaluation_metrics.json)
            
        Raises:
            ModelTrainingError: If saving fails
        """
        try:
            # Create directory if it doesn't exist
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            # Build metrics dictionary (exclude model objects)
            metrics_dict = {}
            for algorithm, result in results.items():
                metrics_dict[algorithm] = result['metrics']
            
            # Add best model indicator
            metrics_dict['best_model'] = best_model_name
            
            # Save to JSON
            with open(path, 'w') as f:
                json.dump(metrics_dict, f, indent=2)
            
            self.logger.info(f"Evaluation metrics saved to {path}")
            
        except Exception as e:
            error_msg = f"Failed to save evaluation metrics to {path}: {e}"
            self.logger.error(error_msg, exc_info=True)
            raise ModelTrainingError(error_msg)
    
    def create_model_metadata(
        self,
        algorithm_name: str,
        metrics: Dict[str, float],
        feature_count: int,
        training_samples: int,
        test_samples: int,
        version: int = 1
    ) -> Dict[str, Any]:
        """
        Create model metadata dictionary.
        
        Args:
            algorithm_name: Name of the algorithm
            metrics: Evaluation metrics dictionary
            feature_count: Number of features in the model
            training_samples: Number of training samples
            test_samples: Number of test samples
            version: Model version number (default: 1)
            
        Returns:
            Dictionary containing complete model metadata
        """
        # Get library versions
        import sklearn
        import numpy
        
        metadata = {
            'algorithm_name': algorithm_name,
            'training_date': datetime.now().isoformat(),
            'feature_count': feature_count,
            'training_samples': training_samples,
            'test_samples': test_samples,
            'random_seed': self.random_seed,
            'library_versions': {
                'scikit-learn': sklearn.__version__,
                'numpy': numpy.__version__,
                'pandas': pd.__version__
            },
            'metrics': {
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'false_positive_rate': metrics['false_positive_rate']
            },
            'confusion_matrix': {
                'true_positives': metrics['true_positives'],
                'false_positives': metrics['false_positives'],
                'true_negatives': metrics['true_negatives'],
                'false_negatives': metrics['false_negatives']
            },
            'version': version
        }
        
        return metadata
