"""Master Prefect flow orchestrating the complete pipeline."""
from prefect import flow
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path
from etl.flows.data_ingestion import data_ingestion_flow
from etl.flows.model_training import model_training_flow
from etl.flows.drift_detection import drift_detection_flow
from src.utils import setup_logger, load_config, generate_batch_id
from src.storage import StorageManager

logger = setup_logger(__name__, "logs/prefect_flows.log")


@flow(name="complete-pipeline-flow", log_prints=True)
def complete_pipeline_flow(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    is_initial: bool = False
):
    """
    Master flow orchestrating the complete TwCS topic modeling pipeline.
    
    This flow runs:
    1. Data Ingestion (ETL)
    2. Model Training (seed or online update)
    3. Drift Detection (if not initial)
    4. State Management
    
    Args:
        start_date: Start date for data window (YYYY-MM-DD)
        end_date: End date for data window (YYYY-MM-DD)
        is_initial: Whether this is initial setup
        
    Returns:
        Dictionary with pipeline results
    """
    logger.info("=" * 80)
    logger.info(f"Starting complete pipeline flow at {datetime.now()}")
    logger.info("=" * 80)
    
    config = load_config()
    storage = StorageManager(config)
    
    # Determine window dates
    if start_date is None or end_date is None:
        state = storage.load_processing_state()
        last_processed = state.get('last_processed_timestamp')
        
        if last_processed:
            start_dt = datetime.fromisoformat(last_processed)
        else:
            start_dt = datetime(2017, 10, 1)  # TwCS dataset start
        
        end_dt = start_dt + timedelta(days=config.scheduler.window_days)
        start_date = start_dt.strftime('%Y-%m-%d')
        end_date = end_dt.strftime('%Y-%m-%d')
    
    batch_id = generate_batch_id(start_date, end_date)
    
    logger.info(f"Processing window: {start_date} to {end_date}")
    logger.info(f"Batch ID: {batch_id}")
    
    try:
        # ========== STEP 1: DATA INGESTION ==========
        logger.info("Step 1: Running data ingestion flow")
        
        parquet_path = f"data/processed/{batch_id}.parquet"
        
        df = data_ingestion_flow(
            csv_path=config.data.raw_csv_path if not is_initial else config.data.sample_csv_path,
            output_parquet=parquet_path,
            start_date=start_date,
            end_date=end_date,
            filter_inbound=True,
            min_docs=5  # Reduced for small datasets
        )
        
        documents = df['text_cleaned'].tolist()
        logger.info(f"Data ingestion complete: {len(documents)} documents")
        
        # ========== STEP 2: MODEL TRAINING ==========
        logger.info("Step 2: Running model training flow")
        
        topics, probs = model_training_flow(
            documents=documents,
            batch_id=batch_id,
            window_start=start_date,
            window_end=end_date,
            is_initial=is_initial
        )
        
        logger.info(f"Model training complete: {len(set(topics))} topics")
        
        # ========== STEP 3: DRIFT DETECTION ==========
        if not is_initial and Path(config.storage.previous_model_path).exists():
            logger.info("Step 3: Running drift detection flow")
            
            # For drift, we'd need previous batch docs - simplified for now
            drift_metrics = drift_detection_flow(
                current_docs=documents[:1000],  # Sample
                previous_docs=[],  # Would load from previous batch
                window_start=start_date
            )
            
            logger.info("Drift detection complete")
        else:
            logger.info("Step 3: Skipping drift detection (initial run or no previous model)")
            drift_metrics = None
        
        # ========== STEP 4: UPDATE STATE ==========
        logger.info("Step 4: Updating processing state")
        
        storage.save_processing_state({
            'last_processed_timestamp': end_date,
            'last_batch_id': batch_id,
            'last_run_timestamp': datetime.now().isoformat(),
            'documents_processed': len(documents),
            'num_topics': len(set(topics)),
            'status': 'success'
        })
        
        logger.info("=" * 80)
        logger.info("Complete pipeline flow finished successfully!")
        logger.info("=" * 80)
        
        return {
            'status': 'success',
            'batch_id': batch_id,
            'documents_processed': len(documents),
            'num_topics': len(set(topics)),
            'drift_detected': drift_metrics is not None and len(drift_metrics.get('alerts', [])) > 0
        }
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        
        # Update state with error
        storage.save_processing_state({
            'status': 'error',
            'error_message': str(e),
            'error_timestamp': datetime.now().isoformat()
        })
        
        raise


if __name__ == "__main__":
    # Run the complete pipeline
    result = complete_pipeline_flow(is_initial=False)
    print(f"Pipeline result: {result}")

