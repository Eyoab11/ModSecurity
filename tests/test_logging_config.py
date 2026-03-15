"""
Unit tests for logging configuration
"""

import os
import logging
from pathlib import Path
import pytest
from src.ml_core.logging_config import setup_logging, get_logger


def test_setup_logging_creates_log_directory(tmp_path):
    """Test that setup_logging creates the logs directory if it doesn't exist."""
    logs_dir = tmp_path / "test_logs"
    logger = setup_logging(logs_dir=str(logs_dir), log_file="test.log")
    
    assert logs_dir.exists()
    assert (logs_dir / "test.log").exists()


def test_setup_logging_returns_logger():
    """Test that setup_logging returns a logger instance."""
    logger = setup_logging()
    
    assert isinstance(logger, logging.Logger)
    assert logger.name == "ml_training_pipeline"


def test_logger_has_console_and_file_handlers():
    """Test that logger has both console and file handlers."""
    logger = setup_logging()
    
    # Should have 2 handlers: console and file
    assert len(logger.handlers) >= 2
    
    handler_types = [type(h).__name__ for h in logger.handlers]
    assert "StreamHandler" in handler_types
    assert "TimedRotatingFileHandler" in handler_types


def test_console_handler_level_is_info():
    """Test that console handler is set to INFO level."""
    logger = setup_logging()
    
    console_handlers = [h for h in logger.handlers if type(h).__name__ == "StreamHandler"]
    assert len(console_handlers) > 0
    assert console_handlers[0].level == logging.INFO


def test_file_handler_level_is_debug():
    """Test that file handler is set to DEBUG level."""
    logger = setup_logging()
    
    file_handlers = [h for h in logger.handlers if type(h).__name__ == "TimedRotatingFileHandler"]
    assert len(file_handlers) > 0
    assert file_handlers[0].level == logging.DEBUG


def test_get_logger_returns_component_logger():
    """Test that get_logger returns a logger for a specific component."""
    logger = get_logger("test_component")
    
    assert isinstance(logger, logging.Logger)
    assert logger.name == "ml_training_pipeline.test_component"


def test_get_logger_without_name_returns_root_logger():
    """Test that get_logger without name returns the root pipeline logger."""
    logger = get_logger()
    
    assert isinstance(logger, logging.Logger)
    assert logger.name == "ml_training_pipeline"


def test_log_format_includes_required_fields(tmp_path, caplog):
    """Test that log messages include timestamp, level, name, and message."""
    logs_dir = tmp_path / "test_logs"
    logger = setup_logging(logs_dir=str(logs_dir), log_file="test.log")
    
    with caplog.at_level(logging.INFO):
        logger.info("Test message")
    
    # Check that log file contains formatted message
    log_file = logs_dir / "test.log"
    log_content = log_file.read_text()
    
    # Format: [%(asctime)s] %(levelname)s [%(name)s] %(message)s
    assert "INFO" in log_content
    assert "[ml_training_pipeline]" in log_content
    assert "Test message" in log_content
