"""
Verification script for Task 1: Project structure and configuration setup

This script verifies that:
1. All required directories exist
2. Configuration file is valid and loadable
3. Logging system works correctly
"""

import os
import yaml
from pathlib import Path
from src.ml_core.logging_config import setup_logging, get_logger


def verify_directory_structure():
    """Verify all required directories exist."""
    required_dirs = ["config", "models", "logs", "src/ml_core", "tests"]
    
    print("Verifying directory structure...")
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"  ✓ {dir_name}/ exists")
        else:
            print(f"  ✗ {dir_name}/ missing")
            return False
    return True


def verify_configuration():
    """Verify configuration file exists and is valid."""
    config_path = "config/training_config.yaml"
    
    print("\nVerifying configuration file...")
    if not Path(config_path).exists():
        print(f"  ✗ {config_path} not found")
        return False
    
    print(f"  ✓ {config_path} exists")
    
    # Load and validate configuration
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ["data", "features", "training", "output", "incremental"]
        for section in required_sections:
            if section in config:
                print(f"  ✓ Section '{section}' present")
            else:
                print(f"  ✗ Section '{section}' missing")
                return False
        
        # Verify key parameters
        print("\nConfiguration parameters:")
        print(f"  - Input path: {config['data']['input_path']}")
        print(f"  - Test size: {config['data']['test_size']}")
        print(f"  - Random seed: {config['data']['random_seed']}")
        print(f"  - Max features: {config['features']['max_features']}")
        print(f"  - N-gram range: {config['features']['ngram_range']}")
        print(f"  - Algorithms: {', '.join(config['training']['algorithms'])}")
        print(f"  - Models directory: {config['output']['models_dir']}")
        print(f"  - Logs directory: {config['output']['logs_dir']}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error loading configuration: {e}")
        return False


def verify_logging():
    """Verify logging system works correctly."""
    print("\nVerifying logging system...")
    
    try:
        # Set up logging
        logger = setup_logging()
        print("  ✓ Logging initialized")
        
        # Test console and file logging
        logger.info("Test INFO message")
        logger.debug("Test DEBUG message")
        print("  ✓ Log messages written")
        
        # Verify log file exists
        log_file = Path("logs/training_errors.log")
        if log_file.exists():
            print(f"  ✓ Log file created: {log_file}")
            
            # Read and display log content
            with open(log_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 0:
                    print(f"  ✓ Log file contains {len(lines)} lines")
                    print("\n  Sample log entries:")
                    for line in lines[-3:]:  # Show last 3 lines
                        print(f"    {line.strip()}")
        else:
            print(f"  ✗ Log file not created")
            return False
        
        # Test component logger
        component_logger = get_logger("test_component")
        component_logger.info("Component test message")
        print("  ✓ Component logger works")
        
        return True
    except Exception as e:
        print(f"  ✗ Error setting up logging: {e}")
        return False


def verify_package():
    """Verify ml_core package is properly defined."""
    print("\nVerifying ml_core package...")
    
    try:
        import src.ml_core
        print(f"  ✓ Package imported successfully")
        print(f"  ✓ Package version: {src.ml_core.__version__}")
        
        # Check __init__.py exists
        init_file = Path("src/ml_core/__init__.py")
        if init_file.exists():
            print(f"  ✓ __init__.py exists")
        else:
            print(f"  ✗ __init__.py missing")
            return False
        
        return True
    except Exception as e:
        print(f"  ✗ Error importing package: {e}")
        return False


def main():
    """Run all verification checks."""
    print("=" * 70)
    print("ML Model Training Pipeline - Setup Verification")
    print("=" * 70)
    
    results = []
    results.append(("Directory Structure", verify_directory_structure()))
    results.append(("Configuration", verify_configuration()))
    results.append(("Package Definition", verify_package()))
    results.append(("Logging System", verify_logging()))
    
    print("\n" + "=" * 70)
    print("Verification Summary")
    print("=" * 70)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:.<50} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 70)
    
    if all_passed:
        print("\n✓ All verification checks passed!")
        print("\nTask 1 completed successfully:")
        print("  - Directory structure created (config/, models/, logs/, src/ml_core/, tests/)")
        print("  - Configuration file created (config/training_config.yaml)")
        print("  - Package defined (src/ml_core/__init__.py)")
        print("  - Logging system configured with console and file handlers")
        print("\nValidates Requirements: 10.7 (configuration), 9.6 (logging)")
        return 0
    else:
        print("\n✗ Some verification checks failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    exit(main())
