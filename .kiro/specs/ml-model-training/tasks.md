# Implementation Plan: ML Model Training Pipeline

## Overview

This implementation plan breaks down the ML Model Training Pipeline into discrete coding tasks. The pipeline will process the fused dataset (data/processed/master_fused_payloads.csv) and produce trained models with comprehensive evaluation metrics. Implementation follows a bottom-up approach: core components first, then integration, then testing.

## Tasks

- [x] 1. Set up project structure and configuration
  - Create directory structure: config/, models/, logs/, src/ml_core/, tests/
  - Create config/training_config.yaml with all pipeline parameters
  - Create src/ml_core/__init__.py to define package
  - Set up logging configuration with console and file handlers
  - _Requirements: 10.7, 9.6_

- [x] 2. Implement DataLoader component
  - [x] 2.1 Create DataLoader class with CSV loading and validation
    - Implement __init__ method with data_path, test_size, random_seed parameters
    - Implement load_and_validate() method to load CSV and check required columns
    - Implement schema validation for payload, label, source_file columns
    - Raise descriptive errors for missing file or invalid schema
    - _Requirements: 1.1, 1.2, 1.3, 1.4_
  
  - [x] 2.2 Implement data cleaning and label normalization
    - Implement clean_data() method to remove records with missing payload/label
    - Implement label normalization to convert all label variants to binary (0/1)
    - Support string labels: "benign", "malicious", "0", "1"
    - Support numeric labels: 0, 1
    - _Requirements: 1.6, 1.7_
  
  - [x] 2.3 Implement stratified train/test split
    - Implement split_data() method using sklearn.model_selection.train_test_split
    - Use stratify parameter to preserve label distribution
    - Use random_state parameter for reproducibility
    - Return training and test DataFrames
    - _Requirements: 2.1, 2.2, 2.3, 2.4_
  
  - [x] 2.4 Implement dataset statistics logging
    - Implement get_statistics() method to calculate record counts and label distributions
    - Log total records, training/test sizes, and label distributions
    - _Requirements: 1.5, 2.5, 2.6_
  
  - [ ]* 2.5 Write property test for DataLoader
    - **Property 2: Data Cleaning Completeness**
    - **Property 3: Label Normalization Consistency**
    - **Property 4: Train-Test Split Ratio**
    - **Property 5: Stratified Sampling Preservation**
    - **Property 6: Split Reproducibility**
    - **Validates: Requirements 1.6, 1.7, 2.1, 2.2, 2.3, 2.4**

- [x] 3. Implement FeatureExtractor component
  - [x] 3.1 Create FeatureExtractor class with TF-IDF vectorization
    - Implement __init__ method with max_features and ngram_range parameters
    - Initialize TfidfVectorizer with analyzer='char', ngram_range=(2,5)
    - Implement fit() method to fit vectorizer on training payloads
    - Implement transform() method to convert payloads to feature vectors
    - Implement fit_transform() method for convenience
    - _Requirements: 3.1, 3.2, 3.3_
  
  - [x] 3.2 Implement vectorizer serialization
    - Implement save() method using joblib.dump to serialize fitted vectorizer
    - Implement load() static method using joblib.load to deserialize vectorizer
    - Save to models/feature_extractor.pkl
    - _Requirements: 3.6, 11.2, 11.3_
  
  - [x] 3.3 Implement feature extraction validation
    - Implement get_feature_names() method to return vocabulary
    - Add validation in transform() to check if vectorizer is fitted
    - Raise NotFittedError if transform called before fit
    - Log feature vector dimensionality after transformation
    - _Requirements: 3.4, 3.5, 3.7, 11.4, 11.7_
  
  - [ ]* 3.4 Write property tests for FeatureExtractor
    - **Property 7: Feature Vector Dimensionality**
    - **Property 8: Feature Extraction Consistency**
    - **Property 9: Feature Extractor Serialization Round-Trip**
    - **Validates: Requirements 3.2, 3.5, 11.3**

- [x] 4. Implement ModelTrainer component
  - [x] 4.1 Create ModelTrainer class with algorithm training
    - Implement __init__ method with algorithms list and random_seed
    - Create algorithm factory to instantiate RandomForest, SVM, LogisticRegression
    - Configure each algorithm with class_weight='balanced' and random_state
    - Implement train_algorithm() method to train a single algorithm
    - Log training time for each algorithm
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_
  
  - [x] 4.2 Implement model evaluation
    - Implement evaluate_model() method to calculate metrics on test set
    - Calculate accuracy, precision, recall, F1-score, false positive rate
    - Generate confusion matrix using sklearn.metrics.confusion_matrix
    - Return metrics dictionary with all calculated values
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_
  
  - [x] 4.3 Implement algorithm tournament and best model selection
    - Implement train_all() method to train all configured algorithms
    - Handle individual algorithm failures gracefully (log and continue)
    - Implement select_best_model() method to rank by F1-score
    - Use false positive rate as tiebreaker for equal F1-scores
    - Log selected best model name and metrics
    - _Requirements: 4.7, 5.7, 6.1, 6.2, 6.3, 6.4, 6.5_
  
  - [x] 4.4 Implement model serialization
    - Implement save_model() method using joblib.dump
    - Save each model to models/{algorithm_name}_model.pkl
    - Save best model to models/best_model.pkl
    - Implement save_metadata() method to create model_metadata.json
    - Include algorithm name, training date, feature count, metrics, library versions
    - Save evaluation_metrics.json with all algorithm results
    - _Requirements: 5.8, 6.6, 6.7, 7.1, 7.2, 7.3, 7.4, 7.5, 7.7_
  
  - [ ]* 4.5 Write property tests for ModelTrainer
    - **Property 10: Model Training Completeness**
    - **Property 11: Evaluation Metrics Calculation**
    - **Property 12: Best Model Selection by F1-Score**
    - **Property 13: Model Serialization Format**
    - **Validates: Requirements 4.4, 5.2, 5.3, 5.4, 5.5, 5.6, 6.2, 6.3, 6.4, 7.1**

- [x] 5. Implement error handling infrastructure
  - [x] 5.1 Create custom exception hierarchy
    - Create PipelineError base exception class
    - Create DataLoadError, FeatureExtractionError, ModelTrainingError subclasses
    - Create AllAlgorithmsFailedError for fatal training failures
    - _Requirements: 9.1, 9.2, 9.3_
  
  - [x] 5.2 Implement centralized logging
    - Configure logging with console handler (INFO level) and file handler (DEBUG level)
    - Set log format with timestamps, component names, severity levels
    - Create logs/training_errors.log file handler
    - Add error logging with stack traces to all components
    - _Requirements: 9.5, 9.6_
  
  - [ ]* 5.3 Write property tests for error handling
    - **Property 17: Error Logging Completeness**
    - **Property 18: Non-Zero Exit Code on Error**
    - **Property 19: Partial Success Handling**
    - **Validates: Requirements 9.4, 9.5, 9.7**

- [ ] 6. Checkpoint - Ensure core components work independently
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 7. Implement TrainingPipeline orchestrator
  - [ ] 7.1 Create TrainingPipeline class with workflow orchestration
    - Implement __init__ method to load configuration from YAML
    - Implement run() method to execute complete pipeline workflow
    - Coordinate DataLoader, FeatureExtractor, ModelTrainer components
    - Handle fatal errors and terminate with non-zero exit codes
    - _Requirements: 9.1, 9.2, 9.3, 9.7_
  
  - [ ] 7.2 Implement training report generation
    - Implement generate_report() method to create comprehensive text report
    - Include dataset statistics, feature extraction details, per-algorithm metrics
    - Include best model selection, artifacts generated, reproducibility info
    - Add training start/end timestamps and total duration
    - Implement save_report() method to write to models/training_report.txt
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7_
  
  - [ ] 7.3 Implement reproducibility tracking
    - Log random seed value used for all random operations
    - Capture library versions for scikit-learn, pandas, numpy
    - Save random seed and library versions to model_metadata.json
    - _Requirements: 10.1, 10.2, 10.3, 10.5, 10.6_
  
  - [ ]* 7.4 Write property tests for TrainingPipeline
    - **Property 1: Dataset Schema Validation**
    - **Property 14: Model Artifact File Naming**
    - **Property 15: Metadata Completeness**
    - **Property 16: Training Report Completeness**
    - **Property 20: Random Seed Consistency**
    - **Validates: Requirements 1.3, 1.4, 7.2, 7.4, 8.2, 8.3, 8.5, 8.6, 10.1, 10.3, 10.4, 10.6**

- [ ] 8. Implement incremental training support
  - [ ] 8.1 Add incremental training configuration
    - Add incremental training section to config/training_config.yaml
    - Add enabled flag, previous_model_path, version_number parameters
    - _Requirements: 12.1_
  
  - [ ] 8.2 Implement incremental training logic
    - Implement load_previous_model() method in TrainingPipeline
    - Implement combine_datasets() method to merge existing and new data
    - Implement compare_models() method to evaluate new vs old performance
    - Log warning if new model performs worse than previous model
    - Implement version tracking in model_metadata.json
    - _Requirements: 12.2, 12.3, 12.4, 12.5, 12.6, 12.7_
  
  - [ ]* 8.3 Write property tests for incremental training
    - **Property 21: Incremental Training Data Combination**
    - **Property 22: Incremental Training Version Increment**
    - **Validates: Requirements 12.2, 12.6, 12.7**

- [ ] 9. Create command-line interface
  - [ ] 9.1 Create main entry point script
    - Create src/ml_core/train.py with main() function
    - Add argument parser for config file path
    - Instantiate and run TrainingPipeline
    - Handle exceptions and return appropriate exit codes
    - Add if __name__ == "__main__" block
    - _Requirements: 9.7, 10.7_
  
  - [ ]* 9.2 Write unit tests for CLI
    - Test argument parsing
    - Test exit codes for various error conditions
    - Test configuration file loading

- [ ] 10. Write unit tests for components
  - [ ]* 10.1 Write unit tests for DataLoader
    - Test loading valid CSV file
    - Test missing file raises FileNotFoundError
    - Test missing columns raises ValueError
    - Test label normalization for all variants
    - Test empty dataset after cleaning raises error
    - Test stratified split preserves distributions
  
  - [ ]* 10.2 Write unit tests for FeatureExtractor
    - Test TF-IDF vectorization produces correct dimensionality
    - Test transform before fit raises NotFittedError
    - Test save and load round-trip produces identical transformations
    - Test special characters and Unicode in payloads
    - Test empty payload list raises error
  
  - [ ]* 10.3 Write unit tests for ModelTrainer
    - Test training each algorithm on synthetic dataset
    - Test evaluation metrics calculation correctness
    - Test best model selection by F1-score
    - Test tiebreaker by false positive rate
    - Test individual algorithm failure handling
    - Test all algorithms failing raises error
    - Test model serialization and deserialization
  
  - [ ]* 10.4 Write unit tests for TrainingPipeline
    - Test complete pipeline execution on small dataset
    - Test configuration loading from YAML
    - Test training report generation
    - Test error handling for missing dataset
    - Test error handling for feature extraction failure

- [ ] 11. Write integration tests
  - [ ]* 11.1 Write end-to-end pipeline test
    - Test complete pipeline from CSV to trained models
    - Verify all artifacts are created with correct names
    - Verify model_metadata.json contains all required fields
    - Verify evaluation_metrics.json contains all algorithms
    - Verify training_report.txt is generated
    - Use sample dataset from data/processed/master_fused_payloads.csv
  
  - [ ]* 11.2 Write artifact compatibility test
    - Test that saved feature_extractor.pkl can be loaded and used
    - Test that saved models can be loaded and make predictions
    - Test that model_metadata.json is valid JSON
    - Test that evaluation_metrics.json is valid JSON

- [ ] 12. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- Integration tests validate end-to-end workflows
- The pipeline uses Python with scikit-learn, pandas, numpy, joblib, and Hypothesis
- All random operations use configurable random seed for reproducibility
- Error handling follows three-tier strategy: fatal, non-fatal, warnings
