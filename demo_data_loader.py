"""
Demonstration script for DataLoader component

This script demonstrates the complete DataLoader workflow:
1. Load and validate the fused dataset
2. Clean data (remove missing values, normalize labels)
3. Split into training and test sets
4. Display statistics
"""

from src.ml_core.data_loader import DataLoader
from src.ml_core.logging_config import setup_logging


def main():
    """Run DataLoader demonstration."""
    # Setup logging
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("DataLoader Component Demonstration")
    logger.info("=" * 80)
    
    # Initialize DataLoader
    loader = DataLoader(
        data_path="data/processed/master_fused_payloads.csv",
        test_size=0.2,
        random_seed=42
    )
    
    # Step 1: Load and validate
    logger.info("\nStep 1: Loading and validating dataset...")
    df = loader.load_and_validate()
    
    # Step 2: Get initial statistics
    logger.info("\nStep 2: Initial dataset statistics...")
    initial_stats = loader.get_statistics(df, "Initial Dataset")
    
    # Step 3: Clean data
    logger.info("\nStep 3: Cleaning data...")
    df_cleaned = loader.clean_data(df)
    
    # Step 4: Split data
    logger.info("\nStep 4: Splitting data into training and test sets...")
    train_df, test_df = loader.split_data(df_cleaned)
    
    # Step 5: Get statistics for splits
    logger.info("\nStep 5: Training set statistics...")
    train_stats = loader.get_statistics(train_df, "Training Set")
    
    logger.info("\nStep 6: Test set statistics...")
    test_stats = loader.get_statistics(test_df, "Test Set")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("DataLoader Workflow Complete!")
    logger.info("=" * 80)
    logger.info(f"Total records processed: {len(df_cleaned)}")
    logger.info(f"Training records: {len(train_df)} ({len(train_df)/len(df_cleaned)*100:.1f}%)")
    logger.info(f"Test records: {len(test_df)} ({len(test_df)/len(df_cleaned)*100:.1f}%)")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
