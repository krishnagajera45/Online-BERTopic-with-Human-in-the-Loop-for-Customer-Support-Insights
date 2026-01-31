#!/usr/bin/env python3
"""
Simple Initial Model Training Script.

Replaces src/scheduler/run_window.py for initial setup.
Uses Prefect flows for consistency with scheduled runs.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from etl.flows.complete_pipeline import complete_pipeline_flow
from src.utils import load_config, setup_logger

logger = setup_logger(__name__, "logs/initial_setup.log")


def run_initial_training():
    """Run initial model training using Prefect flow."""
    logger.info("=" * 80)
    logger.info("Starting Initial Model Training")
    logger.info("=" * 80)
    
    try:
        config = load_config()
        
        # Run complete pipeline - it will auto-detect that no model exists
        # and train a seed model automatically
        logger.info("Running complete pipeline for initial setup...")
        logger.info("(Pipeline will auto-detect this is the first run)")
        
        result = complete_pipeline_flow(
            start_date="2017-10-01",  # TwCS dataset start
            end_date="2017-10-02"     # Small window for initial training
        )
        
        logger.info("=" * 80)
        logger.info("Initial Model Training Complete!")
        logger.info("=" * 80)
        logger.info(f"Result: {result}")
        
        print("\n✓ Initial model trained successfully!")
        print(f"  Model saved to: {config.storage.current_model_path}")
        print(f"  You can now run scheduled workflows or use the API/Dashboard")
        
    except Exception as e:
        logger.error(f"Error during initial training: {e}", exc_info=True)
        print(f"\n✗ Error during initial training: {e}")
        print(f"  Check logs at: logs/initial_setup.log")
        sys.exit(1)


if __name__ == "__main__":
    print("=" * 80)
    print("TwCS Topic Modeling - Initial Model Training")
    print("=" * 80)
    print()
    print("This will:")
    print("  1. Load and preprocess data")
    print("  2. Train initial BERTopic model")
    print("  3. Save model and metadata")
    print()
    
    run_initial_training()
