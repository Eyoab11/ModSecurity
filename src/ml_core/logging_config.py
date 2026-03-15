"""
Logging Configuration Module

Provides centralized logging configuration for the ML training pipeline.
Sets up both console and file handlers with appropriate log levels and formatting.
"""

import logging
import os
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


def setup_logging(logs_dir: str = "logs", log_file: str = "training_errors.log") -> logging.Logger:
    """
    Set up centralized logging with console and file handlers.
    
    Configuration:
    - Console handler: INFO level and above
    - File handler: DEBUG level and above
    - Format: [%(asctime)s] %(levelname)s [%(name)s] %(message)s
    - File rotation: Daily with 7-day retention
    
    Args:
        logs_dir: Directory for log files (default: "logs")
        log_file: Name of the log file (default: "training_errors.log")
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("ml_training_pipeline")
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatters
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler (INFO level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (DEBUG level) with daily rotation and 7-day retention
    log_path = os.path.join(logs_dir, log_file)
    file_handler = TimedRotatingFileHandler(
        log_path,
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info("Logging initialized")
    logger.debug(f"Log file: {log_path}")
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance for a specific component.
    
    Args:
        name: Name of the component (default: None, returns root pipeline logger)
        
    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f"ml_training_pipeline.{name}")
    return logging.getLogger("ml_training_pipeline")
