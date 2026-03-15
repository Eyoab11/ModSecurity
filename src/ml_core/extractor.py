"""
Feature Extraction Module

Provides the FeatureExtractor component for converting raw payload text into
numerical feature vectors using TF-IDF vectorization with character n-grams.
"""

import joblib
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.exceptions import NotFittedError
import scipy.sparse

from .logging_config import get_logger
from .exceptions import FeatureExtractionError


class FeatureExtractor:
    """
    Feature extractor that converts text payloads to numerical feature vectors.
    
    Uses TF-IDF vectorization with character n-grams to capture attack patterns
    in ModSecurity audit log payloads. The vectorizer is fitted on training data
    only to prevent data leakage.
    
    Attributes:
        max_features: Maximum vocabulary size (default: 5000)
        ngram_range: Character n-gram range (default: (2, 5))
        vectorizer: Fitted TfidfVectorizer instance
    """
    
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (2, 5)):
        """
        Initialize feature extractor with TF-IDF parameters.
        
        Args:
            max_features: Maximum vocabulary size (default: 5000)
            ngram_range: Character n-gram range (default: (2, 5))
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=ngram_range,
            max_features=max_features
        )
        self.logger = get_logger("FeatureExtractor")
        self._is_fitted = False
    
    def fit(self, payloads: pd.Series) -> 'FeatureExtractor':
        """
        Fit TF-IDF vectorizer on training payloads.
        
        Args:
            payloads: Series of payload text strings
            
        Returns:
            Self for method chaining
            
        Raises:
            FeatureExtractionError: If payloads is empty or fitting fails
        """
        if len(payloads) == 0:
            error_msg = "Cannot fit on empty payload list"
            self.logger.error(error_msg, exc_info=True)
            raise FeatureExtractionError(error_msg)
        
        try:
            self.logger.info(f"Fitting TF-IDF vectorizer on {len(payloads)} payloads")
            self.vectorizer.fit(payloads)
            self._is_fitted = True
            
            vocab_size = len(self.vectorizer.vocabulary_)
            self.logger.info(f"Vectorizer fitted with vocabulary size: {vocab_size}")
            
            return self
        except Exception as e:
            error_msg = f"Failed to fit vectorizer: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise FeatureExtractionError(error_msg)
    
    def transform(self, payloads: pd.Series) -> scipy.sparse.csr_matrix:
        """
        Transform payloads to feature vectors using fitted vectorizer.
        
        Args:
            payloads: Series of payload text strings
            
        Returns:
            Sparse matrix of feature vectors (n_samples, n_features)
            
        Raises:
            FeatureExtractionError: If transform called before fit or transformation fails
        """
        if not self._is_fitted:
            error_msg = (
                "This FeatureExtractor instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using 'transform'."
            )
            self.logger.error(error_msg, exc_info=True)
            raise FeatureExtractionError(error_msg)
        
        try:
            self.logger.debug(f"Transforming {len(payloads)} payloads to feature vectors")
            feature_vectors = self.vectorizer.transform(payloads)
            
            self.logger.info(
                f"Feature vector dimensionality: {feature_vectors.shape[1]} features, "
                f"{feature_vectors.shape[0]} samples"
            )
            
            return feature_vectors
        except Exception as e:
            error_msg = f"Failed to transform payloads: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise FeatureExtractionError(error_msg)
    
    def fit_transform(self, payloads: pd.Series) -> scipy.sparse.csr_matrix:
        """
        Fit and transform in one step (training data only).
        
        Args:
            payloads: Series of payload text strings
            
        Returns:
            Sparse matrix of feature vectors (n_samples, n_features)
            
        Raises:
            FeatureExtractionError: If payloads is empty or operation fails
        """
        self.fit(payloads)
        return self.transform(payloads)
    
    def save(self, path: str) -> None:
        """
        Serialize fitted vectorizer to disk using joblib.
        
        Args:
            path: File path to save vectorizer (e.g., models/feature_extractor.pkl)
            
        Raises:
            FeatureExtractionError: If vectorizer is not fitted or serialization fails
        """
        if not self._is_fitted:
            error_msg = "Cannot save unfitted FeatureExtractor. Call 'fit' before saving."
            self.logger.error(error_msg, exc_info=True)
            raise FeatureExtractionError(error_msg)
        
        try:
            # Create directory if it doesn't exist
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save the entire FeatureExtractor instance
            joblib.dump(self, path)
            self.logger.info(f"FeatureExtractor saved to {path}")
        except Exception as e:
            error_msg = f"Failed to save FeatureExtractor: {e}"
            self.logger.error(error_msg, exc_info=True)
            raise FeatureExtractionError(error_msg)
    
    @staticmethod
    def load(path: str) -> 'FeatureExtractor':
        """
        Load fitted vectorizer from disk.
        
        Args:
            path: File path to load vectorizer from
            
        Returns:
            Loaded FeatureExtractor instance
            
        Raises:
            FeatureExtractionError: If file does not exist or deserialization fails
        """
        logger = get_logger("FeatureExtractor")
        
        if not Path(path).exists():
            error_msg = f"FeatureExtractor file not found: {path}"
            logger.error(error_msg, exc_info=True)
            raise FeatureExtractionError(error_msg)
        
        try:
            extractor = joblib.load(path)
            logger.info(f"FeatureExtractor loaded from {path}")
            
            # Validate vocabulary size
            vocab_size = len(extractor.vectorizer.vocabulary_)
            if vocab_size != extractor.max_features and vocab_size < extractor.max_features:
                logger.warning(
                    f"Vocabulary size ({vocab_size}) is less than max_features "
                    f"({extractor.max_features}). This may indicate a small training corpus."
                )
            
            return extractor
        except Exception as e:
            error_msg = f"Failed to load FeatureExtractor: {e}"
            logger.error(error_msg, exc_info=True)
            raise FeatureExtractionError(error_msg)
    
    def get_feature_names(self) -> List[str]:
        """
        Return list of feature names (character n-grams).
        
        Returns:
            List of feature names from the vocabulary
            
        Raises:
            FeatureExtractionError: If vectorizer is not fitted
        """
        if not self._is_fitted:
            error_msg = (
                "This FeatureExtractor instance is not fitted yet. "
                "Call 'fit' before accessing feature names."
            )
            self.logger.error(error_msg, exc_info=True)
            raise FeatureExtractionError(error_msg)
        
        return self.vectorizer.get_feature_names_out().tolist()
