# Requirements Document

## Introduction

This document specifies the requirements for the ML Model Training Pipeline feature of ModIntel, an AI-enhanced ModSecurity system. The training pipeline processes labeled security event data to produce machine learning models that classify web application firewall alerts, reducing false positives in production ModSecurity deployments.

The training pipeline consumes the fused dataset created during data exploration (data/processed/master_fused_payloads.csv) and produces trained models with evaluation metrics for deployment to the MLClassifier component.

## Glossary

- **Training_Pipeline**: The complete workflow that loads data, extracts features, trains models, evaluates performance, and saves artifacts
- **FeatureExtractor**: Component that converts raw ModSecurity audit log data and payload text into numerical feature vectors
- **ModelTrainer**: Component that trains multiple ML algorithms and selects the best performing model
- **Fused_Dataset**: The processed CSV file at data/processed/master_fused_payloads.csv containing payload text and labels
- **Feature_Vector**: Numerical representation of a security event derived from payload text and metadata
- **Model_Artifact**: Serialized trained model file saved to disk for deployment
- **Evaluation_Metrics**: Performance measurements including accuracy, precision, recall, and false positive rate
- **Algorithm_Tournament**: Process of training multiple algorithms and comparing their performance
- **MLClassifier**: Production component that uses trained models to classify ModSecurity alerts in real-time
- **ModSecurity_Audit_Log**: Structured log format containing HTTP request details, rule matches, and anomaly scores
- **False_Positive_Rate**: Percentage of benign requests incorrectly classified as malicious
- **Training_Set**: Subset of data used to train the model (typically 80%)
- **Test_Set**: Subset of data used to evaluate model performance (typically 20%)
- **Hyperparameters**: Configuration parameters for ML algorithms that control training behavior

## Requirements

### Requirement 1: Load Training Data

**User Story:** As a security analyst, I want the training pipeline to load the fused dataset, so that I can train models on processed security event data.

#### Acceptance Criteria

1. WHEN the Training_Pipeline is executed, THE Training_Pipeline SHALL load data from data/processed/master_fused_payloads.csv
2. IF the Fused_Dataset file does not exist, THEN THE Training_Pipeline SHALL terminate with a descriptive error message
3. THE Training_Pipeline SHALL validate that the Fused_Dataset contains required columns: payload, label, and source_file
4. IF required columns are missing, THEN THE Training_Pipeline SHALL terminate with a descriptive error message
5. THE Training_Pipeline SHALL log the number of records loaded and the label distribution
6. THE Training_Pipeline SHALL remove records with missing payload or label values
7. THE Training_Pipeline SHALL convert label values to a consistent binary format (0 for benign, 1 for malicious)

### Requirement 2: Split Data for Training and Testing

**User Story:** As a data scientist, I want the pipeline to split data into training and test sets, so that I can evaluate model performance on unseen data.

#### Acceptance Criteria

1. WHEN data is loaded, THE Training_Pipeline SHALL split the data into Training_Set and Test_Set
2. THE Training_Pipeline SHALL allocate 80% of records to the Training_Set and 20% to the Test_Set
3. THE Training_Pipeline SHALL use stratified sampling to preserve label distribution in both sets
4. THE Training_Pipeline SHALL use a fixed random seed for reproducible splits
5. THE Training_Pipeline SHALL log the size of Training_Set and Test_Set
6. THE Training_Pipeline SHALL log the label distribution in both Training_Set and Test_Set

### Requirement 3: Extract Features from Payloads

**User Story:** As a machine learning engineer, I want to extract numerical features from payload text, so that ML algorithms can process the security event data.

#### Acceptance Criteria

1. THE FeatureExtractor SHALL convert payload text into Feature_Vector using TF-IDF vectorization
2. THE FeatureExtractor SHALL limit the vocabulary to the 5000 most frequent terms
3. THE FeatureExtractor SHALL use character n-grams with range (2, 5) to capture attack patterns
4. THE FeatureExtractor SHALL fit the vectorizer on the Training_Set only
5. THE FeatureExtractor SHALL transform both Training_Set and Test_Set using the fitted vectorizer
6. THE FeatureExtractor SHALL save the fitted vectorizer to models/feature_extractor.pkl
7. THE FeatureExtractor SHALL log the dimensionality of the resulting Feature_Vector

### Requirement 4: Train Multiple ML Algorithms

**User Story:** As a machine learning engineer, I want to train multiple algorithms and compare their performance, so that I can select the best model for production.

#### Acceptance Criteria

1. THE ModelTrainer SHALL train a RandomForest classifier with 100 estimators
2. THE ModelTrainer SHALL train a Support Vector Machine classifier with RBF kernel
3. THE ModelTrainer SHALL train a Logistic Regression classifier with L2 regularization
4. WHEN training each algorithm, THE ModelTrainer SHALL fit the model on the Training_Set
5. THE ModelTrainer SHALL log the training time for each algorithm
6. THE ModelTrainer SHALL use class_weight='balanced' for all algorithms to handle class imbalance
7. IF training fails for any algorithm, THEN THE ModelTrainer SHALL log the error and continue with remaining algorithms

### Requirement 5: Evaluate Model Performance

**User Story:** As a security analyst, I want to see performance metrics for each trained model, so that I can understand their effectiveness at reducing false positives.

#### Acceptance Criteria

1. WHEN a model is trained, THE ModelTrainer SHALL evaluate it on the Test_Set
2. THE ModelTrainer SHALL calculate accuracy as the percentage of correct predictions
3. THE ModelTrainer SHALL calculate precision as the ratio of true positives to predicted positives
4. THE ModelTrainer SHALL calculate recall as the ratio of true positives to actual positives
5. THE ModelTrainer SHALL calculate False_Positive_Rate as the ratio of false positives to actual negatives
6. THE ModelTrainer SHALL generate a confusion matrix showing true positives, false positives, true negatives, and false negatives
7. THE ModelTrainer SHALL log all Evaluation_Metrics for each algorithm
8. THE ModelTrainer SHALL save Evaluation_Metrics to a JSON file at models/evaluation_metrics.json

### Requirement 6: Select Best Performing Model

**User Story:** As a machine learning engineer, I want the pipeline to automatically select the best model, so that the most effective model is deployed to production.

#### Acceptance Criteria

1. WHEN all algorithms are trained and evaluated, THE ModelTrainer SHALL compare their performance
2. THE ModelTrainer SHALL rank models by F1-score (harmonic mean of precision and recall)
3. THE ModelTrainer SHALL select the model with the highest F1-score as the best model
4. IF multiple models have equal F1-scores, THEN THE ModelTrainer SHALL select the model with the lowest False_Positive_Rate
5. THE ModelTrainer SHALL log the name of the selected best model and its Evaluation_Metrics
6. THE ModelTrainer SHALL save the best model to models/best_model.pkl
7. THE ModelTrainer SHALL save metadata about the best model to models/model_metadata.json

### Requirement 7: Save Model Artifacts for Deployment

**User Story:** As a DevOps engineer, I want trained models saved in a standard format, so that I can deploy them to the MLClassifier component.

#### Acceptance Criteria

1. WHEN a model is trained, THE ModelTrainer SHALL serialize it using joblib format
2. THE ModelTrainer SHALL save each trained model to models/{algorithm_name}_model.pkl
3. THE ModelTrainer SHALL save the FeatureExtractor to models/feature_extractor.pkl
4. THE ModelTrainer SHALL create a model_metadata.json file containing: algorithm name, training date, feature count, and Evaluation_Metrics
5. THE ModelTrainer SHALL ensure all Model_Artifact files have read permissions for the MLClassifier
6. IF saving fails, THEN THE ModelTrainer SHALL terminate with a descriptive error message
7. THE ModelTrainer SHALL log the file paths of all saved Model_Artifact files

### Requirement 8: Generate Training Report

**User Story:** As a security analyst, I want a comprehensive training report, so that I can review the training process and model performance.

#### Acceptance Criteria

1. WHEN training completes, THE Training_Pipeline SHALL generate a training report
2. THE Training_Pipeline SHALL include in the report: dataset statistics, feature extraction details, and per-algorithm Evaluation_Metrics
3. THE Training_Pipeline SHALL include the selected best model and its performance
4. THE Training_Pipeline SHALL save the report to models/training_report.txt
5. THE Training_Pipeline SHALL include timestamps for training start and completion
6. THE Training_Pipeline SHALL include the total training duration
7. THE Training_Pipeline SHALL log the path to the generated report

### Requirement 9: Handle Training Errors Gracefully

**User Story:** As a machine learning engineer, I want the pipeline to handle errors gracefully, so that I can diagnose and fix issues quickly.

#### Acceptance Criteria

1. IF the Fused_Dataset cannot be loaded, THEN THE Training_Pipeline SHALL log the error and terminate
2. IF feature extraction fails, THEN THE Training_Pipeline SHALL log the error and terminate
3. IF all algorithms fail to train, THEN THE Training_Pipeline SHALL log the error and terminate
4. IF at least one algorithm trains successfully, THEN THE Training_Pipeline SHALL continue and save that model
5. THE Training_Pipeline SHALL log all errors with timestamps and stack traces
6. THE Training_Pipeline SHALL save error logs to logs/training_errors.log
7. WHEN an error occurs, THE Training_Pipeline SHALL return a non-zero exit code

### Requirement 10: Support Reproducible Training

**User Story:** As a data scientist, I want training to be reproducible, so that I can verify results and debug issues.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL use a configurable random seed for all random operations
2. THE Training_Pipeline SHALL log the random seed value used for training
3. THE Training_Pipeline SHALL save the random seed to models/model_metadata.json
4. WHEN the same random seed is used, THE Training_Pipeline SHALL produce identical Feature_Vector and model weights
5. THE Training_Pipeline SHALL log the versions of all ML libraries used (scikit-learn, pandas, numpy)
6. THE Training_Pipeline SHALL save library versions to models/model_metadata.json
7. THE Training_Pipeline SHALL support loading configuration from a YAML file at config/training_config.yaml

### Requirement 11: Validate Feature Extraction Consistency

**User Story:** As a machine learning engineer, I want to ensure feature extraction is consistent between training and production, so that models perform correctly in deployment.

#### Acceptance Criteria

1. THE FeatureExtractor SHALL use the same feature extraction logic for training and production
2. THE FeatureExtractor SHALL save all vectorizer parameters to models/feature_extractor.pkl
3. WHEN the saved FeatureExtractor is loaded, THE FeatureExtractor SHALL produce identical Feature_Vector for the same input
4. THE FeatureExtractor SHALL validate that the vocabulary size matches the saved vectorizer
5. IF the vocabulary size differs, THEN THE FeatureExtractor SHALL log a warning
6. THE FeatureExtractor SHALL include a transform method that accepts raw payload text
7. FOR ALL valid payload strings, transforming with the saved FeatureExtractor SHALL produce a Feature_Vector with the correct dimensionality

### Requirement 12: Support Incremental Model Updates

**User Story:** As a security analyst, I want to retrain models with new data, so that the system adapts to evolving attack patterns.

#### Acceptance Criteria

1. WHERE incremental training is enabled, THE Training_Pipeline SHALL support loading an existing model
2. WHERE incremental training is enabled, THE Training_Pipeline SHALL append new data to the existing Training_Set
3. WHERE incremental training is enabled, THE Training_Pipeline SHALL retrain the model on the combined dataset
4. WHERE incremental training is enabled, THE Training_Pipeline SHALL compare new model performance to the previous model
5. WHERE incremental training is enabled, IF the new model performs worse, THEN THE Training_Pipeline SHALL log a warning
6. WHERE incremental training is enabled, THE Training_Pipeline SHALL save both old and new models with version numbers
7. WHERE incremental training is enabled, THE Training_Pipeline SHALL update model_metadata.json with the new version information
