"""
Scheduler Script for TwCS Topic Modeling.

Runs ETL → Model Update → Drift Detection pipeline.
Can be triggered manually or by cron: 0 2 * * * python src/scheduler/run_window.py
"""
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import load_config, setup_logger
from src.etl import preprocess_batch
from src.modeling import BERTopicOnlineWrapper
from src.drift import DriftDetector
from src.storage import StorageManager

logger = setup_logger(__name__, "logs/scheduler.log")


def run_window_pipeline():
    """Run the complete window processing pipeline."""
    logger.info("=" * 80)
    logger.info(f"Starting window pipeline at {datetime.now()}")
    logger.info("=" * 80)
    
    try:
        # Load configuration
        config = load_config()
        storage = StorageManager(config)
        
        # Step 1: Determine window dates
        logger.info("Step 1: Determining window dates")
        state = storage.load_processing_state()
        last_processed = state.get('last_processed_timestamp')
        
        if last_processed:
            start_date = datetime.fromisoformat(last_processed)
            logger.info(f"Continuing from last processed: {start_date}")
        else:
            # First run - start from beginning of data
            start_date = datetime(2017, 10, 1)  # TwCS dataset start
            logger.info(f"First run, starting from: {start_date}")
        
        end_date = start_date + timedelta(days=config.scheduler.window_days)
        batch_id = f"batch_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}"
        
        logger.info(f"Processing window: {start_date} to {end_date}")
        logger.info(f"Batch ID: {batch_id}")
        
        # Step 2: ETL - Ingest and preprocess new data window
        logger.info("Step 2: Running ETL pipeline")
        parquet_path = f"data/processed/{batch_id}.parquet"
        
        new_data = preprocess_batch(
            csv_path=config.data.raw_csv_path,
            output_parquet=parquet_path,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            filter_inbound=True
        )
        
        if len(new_data) == 0:
            logger.warning("No new data in this window, skipping")
            return
        
        logger.info(f"Processed {len(new_data)} documents")
        
        # Step 3: Model Update
        logger.info("Step 3: Updating model")
        model_wrapper = BERTopicOnlineWrapper(config)
        
        # Check if this is initial training or update
        current_model_path = Path(config.storage.current_model_path)
        
        if not current_model_path.exists():
            # Initial seed model training
            logger.info("No existing model found, training seed model")
            topics, probs = model_wrapper.train_seed_model(
                documents=new_data['text_cleaned'].tolist(),
                batch_id=batch_id,
                window_start=start_date.strftime('%Y-%m-%d'),
                window_end=end_date.strftime('%Y-%m-%d')
            )
            logger.info(f"Seed model trained with {len(set(topics))} topics")
        else:
            # Update existing model
            logger.info("Existing model found, performing online update")
            
            # Archive current → previous
            previous_model_path = Path(config.storage.previous_model_path)
            previous_model_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(current_model_path, previous_model_path)
            logger.info("Archived current model as previous")
            
            # Update model
            topics, probs = model_wrapper.update_model_online(
                new_documents=new_data['text_cleaned'].tolist(),
                batch_id=batch_id,
                window_start=start_date.strftime('%Y-%m-%d'),
                window_end=end_date.strftime('%Y-%m-%d')
            )
            logger.info(f"Model updated with {len(set(topics))} topics")
        
        # Step 4: Save document assignments
        logger.info("Step 4: Saving document assignments")
        
        # Calculate confidence scores from probabilities
        # Handle different probability formats from BERTopic
        if probs is None:
            confidence = [0.0] * len(topics)
        elif len(probs.shape) == 2:
            # 2D array: each row is probability distribution
            confidence = [p.max() if len(p) > 0 else 0.0 for p in probs]
        else:
            # 1D array: single probability value per document
            confidence = probs.tolist()
        
        assignments = new_data[['doc_id']].copy()
        assignments['topic_id'] = topics
        assignments['timestamp'] = new_data['created_at']
        assignments['batch_id'] = batch_id
        assignments['confidence'] = confidence
        
        storage.append_doc_assignments(assignments)
        logger.info(f"Saved {len(assignments)} document assignments")
        
        # Step 5: Drift Detection (if not first run)
        if current_model_path.exists() and Path(config.storage.previous_model_path).exists():
            logger.info("Step 5: Running drift detection")
            
            detector = DriftDetector(config)
            current_model = model_wrapper.load_model(config.storage.current_model_path)
            previous_model = model_wrapper.load_model(config.storage.previous_model_path)
            
            # For drift detection, we need previous window documents
            # For simplicity, we'll skip document-based metrics in this iteration
            drift_metrics = detector.run_full_drift_detection(
                current_model=current_model,
                previous_model=previous_model,
                current_docs=new_data['text_cleaned'].tolist()[:1000],  # Sample for performance
                previous_docs=[],  # Would need previous window docs
                window_start=start_date.strftime('%Y-%m-%d')
            )
            
            alerts = detector.generate_drift_alerts(
                drift_metrics,
                start_date.strftime('%Y-%m-%d')
            )
            if alerts:
                storage.append_drift_alerts(alerts)
            
            logger.info(f"Drift detection complete: {len(alerts)} alerts generated")
            
            # Log drift summary
            prevalence_change = drift_metrics['prevalence_change'].get('prevalence_change', 0)
            logger.info(f"Prevalence change: {prevalence_change:.4f}")
            logger.info(f"New topics: {len(drift_metrics['topic_changes']['new_topics'])}")
            logger.info(f"Disappeared topics: {len(drift_metrics['topic_changes']['disappeared_topics'])}")
        else:
            logger.info("Step 5: Skipping drift detection (first run)")
        
        # Step 6: Update processing state
        logger.info("Step 6: Updating processing state")
        storage.save_processing_state({
            'last_processed_timestamp': end_date.isoformat(),
            'last_batch_id': batch_id,
            'last_run_timestamp': datetime.now().isoformat(),
            'documents_processed': len(new_data),
            'status': 'success'
        })
        
        logger.info("=" * 80)
        logger.info("Window pipeline completed successfully!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error in window pipeline: {e}", exc_info=True)
        
        # Update state with error
        try:
            storage = StorageManager()
            state = storage.load_processing_state()
            state['status'] = 'error'
            state['error_message'] = str(e)
            state['error_timestamp'] = datetime.now().isoformat()
            storage.save_processing_state(state)
        except:
            pass
        
        raise


def run_initial_setup():
    """Run initial setup with full dataset (raw CSV)."""
    logger.info("Running initial setup with raw data")
    
    try:
        config = load_config()
        storage = StorageManager(config)
        
        # Use raw data
        logger.info("Processing raw dataset")
        parquet_path = "data/processed/twcs_initial.parquet"
        
        sample_data = preprocess_batch(
            csv_path=config.data.raw_csv_path,
            output_parquet=parquet_path,
            filter_inbound=True
        )
        
        logger.info(f"Processed {len(sample_data)} documents from raw dataset")

        # Determine actual window from data if timestamps exist
        if 'created_at' in sample_data.columns and len(sample_data) > 0:
            window_start = sample_data['created_at'].min().isoformat()
            window_end = sample_data['created_at'].max().isoformat()
        else:
            window_start = "2017-10-01"
            window_end = "2017-10-31"
        
        # Train seed model
        logger.info("Training initial seed model")
        model_wrapper = BERTopicOnlineWrapper(config)
        
        topics, probs = model_wrapper.train_seed_model(
            documents=sample_data['text_cleaned'].tolist(),
            batch_id="batch_initial",
            window_start=window_start,
            window_end=window_end
        )
        
        logger.info(f"Initial model trained with {len(set(topics))} topics")
        
        # Calculate confidence scores from probabilities
        # Handle different probability formats from BERTopic
        if probs is None:
            confidence = [0.0] * len(topics)
        elif len(probs.shape) == 2:
            # 2D array: each row is probability distribution
            confidence = [p.max() if len(p) > 0 else 0.0 for p in probs]
        else:
            # 1D array: single probability value per document
            confidence = probs.tolist()
        
        # Save assignments
        assignments = sample_data[['doc_id']].copy()
        assignments['topic_id'] = topics
        assignments['timestamp'] = sample_data['created_at']
        assignments['batch_id'] = "batch_initial"
        assignments['confidence'] = confidence
        
        storage.append_doc_assignments(assignments)
        
        # Update state
        storage.save_processing_state({
            'last_processed_timestamp': window_end,
            'last_batch_id': "batch_initial",
            'last_run_timestamp': datetime.now().isoformat(),
            'documents_processed': len(sample_data),
            'status': 'success',
            'note': 'Initial setup with raw data'
        })
        
        logger.info("Initial setup completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in initial setup: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TwCS Topic Modeling Scheduler")
    parser.add_argument(
        '--init',
        action='store_true',
        help='Run initial setup with sample data'
    )
    parser.add_argument(
        '--window',
        action='store_true',
        help='Run window processing pipeline'
    )
    
    args = parser.parse_args()
    
    if args.init:
        run_initial_setup()
    elif args.window:
        run_window_pipeline()
    else:
        # Default: run window pipeline
        run_window_pipeline()

