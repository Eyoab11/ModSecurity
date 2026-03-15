"""
DataLoader Component

Responsible for loading and validating the fused dataset from CSV,
performing data cleaning (removing missing values, normalizing labels),
splitting data into training and test sets using stratified sampling,
and logging dataset statistics.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from .logging_config import get_logger
from .exceptions import DataLoadError


class DataLoader:
    """
    DataLoader component for loading, validating, cleaning, and splitting training data.
    
    Attributes:
        data_path: Path to the fused dataset CSV file
        test_size: Fraction of data to use for test set (default: 0.2)
        random_seed: Random seed for reproducible splits (default: 42)
        logger: Logger instance for this component
    """
    
    def __init__(self, data_path: str = "data/processed/master_fused_payloads.csv", 
                 test_size: float = 0.2, 
                 random_seed: int = 42):
        """
        Initialize DataLoader with configuration.
        
        Args:
            data_path: Path to fused dataset CSV
            test_size: Fraction of data for test set (0.0 to 1.0)
            random_seed: Random seed for reproducibility
        """
        self.data_path = data_path
        self.test_size = test_size
        self.random_seed = random_seed
        self.logger = get_logger("DataLoader")
        
    def load_and_validate(self) -> pd.DataFrame:
        """
        Load CSV and validate schema.
        
        Returns:
            DataFrame with validated schema
            
        Raises:
            DataLoadError: If dataset file does not exist or validation fails
        """
        # Check if file exists
        if not Path(self.data_path).exists():
            error_msg = f"Dataset file not found: {self.data_path}"
            self.logger.error(error_msg, exc_info=True)
            raise DataLoadError(error_msg)
        
        # Load CSV
        self.logger.info(f"Loading dataset from {self.data_path}")
        try:
            df = pd.read_csv(self.data_path)
        except Exception as e:
            error_msg = f"Failed to load CSV file: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise DataLoadError(error_msg)
        
        # Validate required columns
        required_columns = ['payload', 'label', 'source_file']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            error_msg = f"Missing required columns: {', '.join(missing_columns)}"
            self.logger.error(error_msg, exc_info=True)
            raise DataLoadError(error_msg)
        
        self.logger.info(f"Dataset loaded successfully with {len(df)} records")
        self.logger.debug(f"Columns: {', '.join(df.columns)}")
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove records with missing payload or label values and normalize labels.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame with normalized labels
            
        Raises:
            DataLoadError: If dataset is empty after cleaning
        """
        original_size = len(df)
        self.logger.info(f"Cleaning data (original size: {original_size} records)")
        
        # Remove records with missing payload or label
        df_cleaned = df.dropna(subset=['payload', 'label']).copy()
        removed_count = original_size - len(df_cleaned)
        
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} records with missing payload or label")
        
        # Check if dataset is empty after cleaning
        if len(df_cleaned) == 0:
            error_msg = "Dataset is empty after removing records with missing values"
            self.logger.error(error_msg, exc_info=True)
            raise DataLoadError(error_msg)
        
        # Normalize labels to binary format (0/1)
        df_cleaned['label'] = df_cleaned['label'].apply(self._normalize_label)
        
        self.logger.info(f"Data cleaning complete ({len(df_cleaned)} records remaining)")
        
        return df_cleaned
    
    def _normalize_label(self, label: Any) -> int:
        """
        Normalize label value to binary format (0 for benign, 1 for malicious).
        
        Args:
            label: Input label (can be string or numeric)
            
        Returns:
            Normalized label (0 or 1)
        """
        # Convert to string and lowercase for comparison
        label_str = str(label).lower().strip()
        
        # Map benign variants to 0
        if label_str in ['0', 'benign', 'normal']:
            return 0
        
        # Map malicious variants to 1
        if label_str in ['1', 'malicious', 'attack']:
            return 1
        
        # Check if label contains attack-related keywords (e.g., "dos", "sql injection", etc.)
        attack_keywords = ['dos', 'ddos', 'injection', 'xss', 'csrf', 'exploit', 'intrusion', 'breach']
        if any(keyword in label_str for keyword in attack_keywords):
            return 1
        
        # If already numeric, convert to int
        try:
            label_int = int(float(label_str))
            if label_int in [0, 1]:
                return label_int
        except (ValueError, TypeError):
            pass
        
        # If we can't normalize, log warning and default to 1 (malicious) to be safe
        self.logger.warning(f"Unknown label value '{label}', defaulting to 1 (malicious)")
        return 1
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform stratified train/test split.
        
        Args:
            df: Input DataFrame with cleaned data
            
        Returns:
            Tuple of (train_df, test_df)
        """
        self.logger.info(f"Splitting data (test_size={self.test_size}, random_seed={self.random_seed})")
        
        # Perform stratified split to preserve label distribution
        train_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_seed,
            stratify=df['label']
        )
        
        self.logger.info(f"Split complete: {len(train_df)} training records, {len(test_df)} test records")
        
        return train_df, test_df
    
    def get_statistics(self, df: pd.DataFrame, name: str = "Dataset") -> Dict[str, Any]:
        """
        Calculate and return dataset statistics.
        
        Args:
            df: DataFrame to analyze
            name: Name of the dataset for logging
            
        Returns:
            Dictionary containing statistics
        """
        total_records = len(df)
        label_counts = df['label'].value_counts().to_dict()
        
        # Calculate label distribution percentages
        label_distribution = {
            label: (count / total_records * 100) 
            for label, count in label_counts.items()
        }
        
        stats = {
            'total_records': total_records,
            'label_counts': label_counts,
            'label_distribution': label_distribution
        }
        
        # Log statistics
        self.logger.info(f"{name} Statistics:")
        self.logger.info(f"  Total records: {total_records}")
        
        for label, count in sorted(label_counts.items()):
            label_name = "Benign" if label == 0 else "Malicious"
            percentage = label_distribution[label]
            self.logger.info(f"  {label_name}: {count} ({percentage:.1f}%)")
        
        return stats
