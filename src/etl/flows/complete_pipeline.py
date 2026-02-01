"""Master Prefect flow orchestrating the complete pipeline."""
from prefect import flow, get_run_logger
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path
import time
import mlflow
import pandas as pd
from src.etl.flows.data_ingestion import data_ingestion_flow
from src.etl.flows.model_training import model_training_flow
from src.etl.flows.drift_detection import drift_detection_flow
from src.utils import load_config, generate_batch_id, MLflowLogger, get_prefect_context
from src.utils import StorageManager


@flow(name="complete-pipeline-flow", log_prints=True)
def complete_pipeline_flow(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Master flow orchestrating the complete TwCS topic modeling pipeline.
    
    This flow automatically:
    1. Data Ingestion (ETL)
    2. Model Training (auto-detects seed vs online update)
    3. Drift Detection (if previous model exists)
    4. State Management
    
    Args:
        start_date: Start date for data window (YYYY-MM-DD)
        end_date: End date for data window (YYYY-MM-DD)
        
    Returns:
        Dictionary with pipeline results
    """
    logger = get_run_logger()
    
    logger.info("=" * 80)
    logger.info(f"Starting complete pipeline flow at {datetime.now()}")
    logger.info("=" * 80)
    
    # Start timing
    pipeline_start_time = time.time()
    
    config = load_config()
    storage = StorageManager(config)
    
    # Get Prefect context for MLflow linking
    prefect_ctx = get_prefect_context()
    logger.info(f"Prefect Flow Run: {prefect_ctx.get('flow_run_name', 'N/A')}")
    logger.info(f"Prefect Flow Run ID: {prefect_ctx.get('flow_run_id', 'N/A')}")
    if prefect_ctx.get('flow_run_url'):
        logger.info(f"🔗 Prefect UI: {prefect_ctx['flow_run_url']}")
    
    # Determine window dates
    if start_date is None or end_date is None:
        state = storage.load_processing_state()
        last_processed = state.get('last_processed_timestamp')
        
        if last_processed:
            start_dt = datetime.fromisoformat(last_processed)
        else:
            start_dt = datetime(2017, 10, 1)  # TwCS dataset start
        
        window_minutes = getattr(config.scheduler, "window_minutes", None)
        if window_minutes:
            end_dt = start_dt + timedelta(minutes=window_minutes)
        else:
            end_dt = start_dt + timedelta(days=config.scheduler.window_days)
        
        start_date = start_dt.strftime('%Y-%m-%d %H:%M:%S')
        end_date = end_dt.strftime('%Y-%m-%d %H:%M:%S')
    
    batch_id = (
        f"batch_{datetime.fromisoformat(start_date).strftime('%Y%m%d_%H%M')}"
        f"_to_{datetime.fromisoformat(end_date).strftime('%Y%m%d_%H%M')}"
    )
    
    logger.info(f"Processing window: {start_date} to {end_date}")
    logger.info(f"Batch ID: {batch_id}")
    
    # Initialize MLflow logger
    mlflow_logger = MLflowLogger(
        tracking_uri=config.mlflow.tracking_uri,
        experiment_name=config.mlflow.experiment_name
    )
    
    try:
        # Start MLflow run with Prefect context
        mlflow_run = mlflow_logger.start_run_with_prefect_context(
            batch_id=batch_id,
            prefect_flow_run_id=prefect_ctx.get('flow_run_id'),
            prefect_flow_run_name=prefect_ctx.get('flow_run_name'),
            prefect_flow_run_url=prefect_ctx.get('flow_run_url')
        )
        
        # Log system info
        mlflow_logger.log_system_info()
        
        # ========== STEP 1: DATA INGESTION ==========
        logger.info("Step 1: Running data ingestion flow")
        step1_start = time.time()
        
        parquet_path = f"data/processed/{batch_id}.parquet"
        
        df = data_ingestion_flow(
            csv_path=config.data.raw_csv_path,
            output_parquet=parquet_path,
            start_date=start_date,
            end_date=end_date,
            filter_inbound=True,
            min_docs=5  # Reduced for small datasets
        )
        
        documents = df['text_cleaned'].tolist()
        logger.info(f"Data ingestion complete: {len(documents)} documents")
        
        # Log step 1 timing and batch statistics
        step1_duration = time.time() - step1_start
        mlflow_logger.log_processing_time("data_ingestion", step1_duration)
        mlflow_logger.log_batch_statistics(
            documents=documents,
            batch_id=batch_id,
            window_start=start_date,
            window_end=end_date,
            df=df
        )
        
        # ========== STEP 2: MODEL TRAINING ==========
        logger.info("Step 2: Running model training flow")
        step2_start = time.time()
        
        topics, probs = model_training_flow(
            documents=documents,
            batch_id=batch_id,
            window_start=start_date,
            window_end=end_date
        )
        
        logger.info(f"Model training complete: {len(set(topics))} topics")
        
        # Auto-detect model stage
        model_stage = 'initial' if not Path(config.storage.previous_model_path).exists() else 'online_update'

        # Log step 2 timing and model details
        step2_duration = time.time() - step2_start
        mlflow_logger.log_processing_time("model_training", step2_duration)
        
        # Load model to log details
        from src.utils import load_bertopic_model
        model = load_bertopic_model(config.storage.current_model_path)
        
        mlflow_logger.log_model_details(
            model=model,
            topics=topics,
            probs=probs,
            model_config={
                "embedding_model": config.model.embedding_model,
                "min_cluster_size": config.model.min_cluster_size,
                "min_samples": config.model.min_samples,
                "n_neighbors": config.model.umap_n_neighbors,
                "n_components": config.model.umap_n_components,
                "top_n_words": config.model.top_n_words
            },
            is_initial=(model_stage == 'initial')
        )
        
        # Log model artifact
        mlflow_logger.log_model_artifact(config.storage.current_model_path)
        
        # ========== STEP 3: DRIFT DETECTION ==========
        step3_start = time.time()
        if Path(config.storage.previous_model_path).exists():
            logger.info("Step 3: Running drift detection flow")
            
            # Load previous batch documents for drift comparison
            previous_docs = []
            try:
                # Try to find and load the previous batch's parquet file
                import glob
                parquet_files = sorted(glob.glob(f"{config.data.processed_parquet_dir}*.parquet"), reverse=True)
                if len(parquet_files) >= 2:
                    # Load second most recent parquet (previous batch)
                    previous_df = pd.read_parquet(parquet_files[1])
                    previous_docs = previous_df['text_cleaned'].tolist() if 'text_cleaned' in previous_df.columns else []
                    logger.info(f"Loaded {len(previous_docs)} previous documents for drift detection")
                else:
                    logger.warning(f"Could not find previous batch parquet file for drift detection")
            except Exception as e:
                logger.warning(f"Error loading previous documents for drift detection: {e}")
            
            drift_metrics = drift_detection_flow(
                current_docs=documents,  # Sample
                previous_docs=previous_docs if previous_docs else [],  # Sample
                window_start=start_date
            )
            
            logger.info("Drift detection complete")
            
            # Log drift metrics
            step3_duration = time.time() - step3_start
            mlflow_logger.log_processing_time("drift_detection", step3_duration)
            mlflow_logger.log_drift_metrics(drift_metrics, start_date)
            
            # Log alerts if any
            alerts = drift_metrics.get('alerts', [])
            mlflow_logger.log_alerts(alerts)
        else:
            logger.info("Step 3: Skipping drift detection (initial run or no previous model)")
            drift_metrics = None
            alerts = []
        
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
        
        # Calculate total pipeline duration
        pipeline_duration = time.time() - pipeline_start_time
        
        # Log pipeline summary
        mlflow_logger.log_pipeline_summary(
            status='success',
            documents_processed=len(documents),
            num_topics=len(set(topics)),
            drift_detected=drift_metrics is not None and len(drift_metrics.get('alerts', [])) > 0,
            num_alerts=len(alerts) if drift_metrics else 0,
            total_duration_seconds=pipeline_duration
        )
        
        logger.info("=" * 80)
        logger.info("Complete pipeline flow finished successfully!")
        logger.info(f"Total duration: {pipeline_duration:.2f}s ({pipeline_duration/60:.2f}m)")
        logger.info("=" * 80)
        
        # End MLflow run
        mlflow.end_run()
        
        return {
            'status': 'success',
            'batch_id': batch_id,
            'documents_processed': len(documents),
            'num_topics': len(set(topics)),
            'drift_detected': drift_metrics is not None and len(drift_metrics.get('alerts', [])) > 0,
            'mlflow_run_id': mlflow_run.info.run_id,
            'prefect_run_id': prefect_ctx.get('flow_run_id'),
            'duration_seconds': pipeline_duration
        }
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        
        # Log error to MLflow
        try:
            pipeline_duration = time.time() - pipeline_start_time
            mlflow_logger.log_pipeline_summary(
                status='error',
                documents_processed=0,
                num_topics=0,
                drift_detected=False,
                num_alerts=0,
                total_duration_seconds=pipeline_duration
            )
            mlflow.set_tag("error", str(e))
            mlflow.log_param("error_message", str(e)[:250])  # Truncate if too long
            mlflow.end_run(status="FAILED")
        except Exception as mlflow_error:
            logger.warning(f"Could not log error to MLflow: {mlflow_error}")
        
        # Update state with error
        storage.save_processing_state({
            'status': 'error',
            'error_message': str(e),
            'error_timestamp': datetime.now().isoformat()
        })
        
        raise


if __name__ == "__main__":
    # Run the complete pipeline
    result = complete_pipeline_flow()
    print(f"Pipeline result: {result}")

