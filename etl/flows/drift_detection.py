"""Prefect flow for drift detection (with granular task-level tracking)."""
from prefect import flow, get_run_logger
from pathlib import Path
from datetime import datetime
from etl.tasks.drift_tasks import (
    load_bertopic_models_task,
    calculate_prevalence_change_task,
    calculate_centroid_shift_task,
    calculate_keyword_divergence_task,
    detect_topic_changes_task,
    generate_drift_alerts_task,
    save_drift_alerts_task
)
from src.utils import load_config


@flow(name="drift-detection-flow", log_prints=True)
def drift_detection_flow(
    current_docs: list,
    previous_docs: list,
    window_start: str
):
    """
    Prefect flow for drift detection with granular task tracking.
    
    This flow orchestrates granular drift detection tasks:
    1. Load current and previous models
    2. Calculate prevalence change (task)
    3. Calculate centroid shifts (task)
    4. Calculate keyword divergence (task)
    5. Detect new/disappeared topics (task)
    6. Generate drift alerts (task)
    7. Save alerts to storage (task)
    
    Args:
        current_docs: Current batch documents
        previous_docs: Previous batch documents
        window_start: Window start timestamp
        
    Returns:
        Drift metrics dictionary with alerts
    """
    logger = get_run_logger()
    
    logger.info(f"Starting granular drift detection flow")
    logger.info(f"Current docs: {len(current_docs)}")
    logger.info(f"Previous docs: {len(previous_docs) if previous_docs else 0}")
    logger.info(f"Window: {window_start}")
    
    config = load_config()
    
    # Check if previous model exists
    previous_model_exists = Path(config.storage.previous_model_path).exists()
    current_model_exists = Path(config.storage.current_model_path).exists()
    
    if not previous_model_exists:
        logger.warning("No previous model found, skipping drift detection")
        return {"status": "skipped", "reason": "no_previous_model"}
    
    if not current_model_exists:
        logger.warning("No current model found, skipping drift detection")
        return {"status": "skipped", "reason": "no_current_model"}
    
    # Step 1: Load models (task)
    logger.info("Step 1: Loading BERTopic models")
    current_model, previous_model = load_bertopic_models_task(
        current_model_path=config.storage.current_model_path,
        previous_model_path=config.storage.previous_model_path
    )
    
    # Step 2: Calculate prevalence change (task)
    logger.info("Step 2: Calculating topic prevalence change")
    prevalence_metrics = calculate_prevalence_change_task(
        current_model=current_model,
        previous_model=previous_model
    )
    
    # Step 3: Calculate centroid shifts (task)
    logger.info("Step 3: Calculating topic centroid shifts")
    centroid_shifts = calculate_centroid_shift_task(
        current_model=current_model,
        previous_model=previous_model,
        current_docs=current_docs[:1000],  # Sample for performance
        previous_docs=previous_docs[:1000] if previous_docs else []
    )
    
    # Step 4: Calculate keyword divergence (task)
    logger.info("Step 4: Calculating keyword divergence")
    keyword_divergences = calculate_keyword_divergence_task(
        current_model=current_model,
        previous_model=previous_model
    )
    
    # Step 5: Detect topic changes (task)
    logger.info("Step 5: Detecting new and disappeared topics")
    topic_changes = detect_topic_changes_task(
        current_model=current_model,
        previous_model=previous_model
    )
    
    # Assemble drift metrics
    drift_metrics = {
        'prevalence_change': prevalence_metrics,
        'centroid_shifts': centroid_shifts,
        'keyword_divergences': keyword_divergences,
        'topic_changes': topic_changes,
        'window_start': window_start,
        'timestamp': datetime.now().isoformat()
    }
    
    # Step 6: Generate alerts (task)
    logger.info("Step 6: Generating drift alerts")
    alerts = generate_drift_alerts_task(drift_metrics, window_start)
    
    # Step 7: Save alerts (task)
    if alerts:
        logger.info(f"Step 7: Saving {len(alerts)} drift alerts")
        save_drift_alerts_task(alerts)
        logger.info(f"Generated and saved {len(alerts)} drift alerts")
    else:
        logger.info("Step 7: No drift alerts generated")
    
    logger.info("Drift detection flow completed")
    
    # Add alerts to metrics
    drift_metrics["alerts"] = alerts
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

