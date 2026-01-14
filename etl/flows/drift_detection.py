"""Prefect flow for drift detection."""
from prefect import flow
from pathlib import Path
from etl.tasks.drift_tasks import calculate_drift_task, generate_alerts_task, save_alerts_task
from src.utils import setup_logger, load_config

logger = setup_logger(__name__, "logs/prefect_flows.log")


@flow(name="drift-detection-flow", log_prints=True)
def drift_detection_flow(
    current_docs: list,
    previous_docs: list,
    window_start: str
):
    """
    Prefect flow for drift detection.
    
    This flow:
    1. Loads current and previous models
    2. Calculates drift metrics (prevalence, centroid, JS divergence)
    3. Generates alerts if thresholds exceeded
    4. Saves alerts to storage
    
    Args:
        current_docs: Current batch documents
        previous_docs: Previous batch documents
        window_start: Window start timestamp
        
    Returns:
        Drift metrics dictionary
    """
    logger.info(f"Starting drift detection flow")
    logger.info(f"Current docs: {len(current_docs)}")
    logger.info(f"Previous docs: {len(previous_docs) if previous_docs else 0}")
    logger.info(f"Window: {window_start}")
    
    config = load_config()
    
    # Check if previous model exists
    previous_model_exists = Path(config.storage.previous_model_path).exists()
    
    if not previous_model_exists:
        logger.warning("No previous model found, skipping drift detection")
        return {"status": "skipped", "reason": "no_previous_model"}
    
    # Step 1: Calculate drift metrics
    drift_metrics = calculate_drift_task(
        current_docs=current_docs,
        previous_docs=previous_docs,
        window_start=window_start
    )
    
    # Step 2: Generate alerts
    alerts = generate_alerts_task(drift_metrics, window_start)
    
    # Step 3: Save alerts
    if alerts:
        save_alerts_task(alerts)
        logger.info(f"Generated and saved {len(alerts)} drift alerts")
    else:
        logger.info("No drift alerts generated")
    
    logger.info("Drift detection flow completed")
    
    return drift_metrics


if __name__ == "__main__":
    # Test the flow
    test_current = ["billing issue", "payment problem"]
    test_previous = ["technical support", "app crash"]
    
    metrics = drift_detection_flow(
        current_docs=test_current,
        previous_docs=test_previous,
        window_start="2024-01-01"
    )
    print(f"Drift metrics: {metrics}")

