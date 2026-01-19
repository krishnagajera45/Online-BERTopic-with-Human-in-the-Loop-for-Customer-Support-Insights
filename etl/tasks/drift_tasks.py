"""Prefect tasks for drift detection."""
from prefect import task
from typing import Dict, Any, List
from src.drift import DriftDetector
from src.modeling import BERTopicOnlineWrapper
from src.utils import setup_logger, load_config
from src.storage import StorageManager

logger = setup_logger(__name__, "logs/prefect_tasks.log")


@task(name="calculate_drift", retries=1)
def calculate_drift_task(
    current_docs: List[str],
    previous_docs: List[str],
    window_start: str
) -> Dict[str, Any]:
    """
    Calculate drift between current and previous models.
    
    Args:
        current_docs: Current batch documents
        previous_docs: Previous batch documents
        window_start: Window start timestamp
        
    Returns:
        Dictionary with drift metrics
    """
    logger.info("Calculating topic drift")
    
    config = load_config()
    detector = DriftDetector(config)
    model_wrapper = BERTopicOnlineWrapper(config)
    
    # Load models
    current_model = model_wrapper.load_model(config.storage.current_model_path)
    previous_model = model_wrapper.load_model(config.storage.previous_model_path)
    
    # Calculate drift
    drift_metrics = detector.run_full_drift_detection(
        current_model=current_model,
        previous_model=previous_model,
        current_docs=current_docs[:1000],  # Sample for performance
        previous_docs=previous_docs[:1000] if previous_docs else [],
        window_start=window_start
    )
    
    logger.info(f"Drift detection complete. Drift metrics: {drift_metrics}")
    return drift_metrics


@task(name="generate_alerts")
def generate_alerts_task(drift_metrics: Dict[str, Any], window_start: str) -> List[Dict[str, Any]]:
    """
    Generate drift alerts from metrics.
    
    Args:
        drift_metrics: Drift metrics dictionary
        window_start: Window start timestamp
        
    Returns:
        List of alert dictionaries
    """
    logger.info("Generating drift alerts")
    
    config = load_config()
    detector = DriftDetector(config)
    
    alerts = detector.generate_drift_alerts(drift_metrics, window_start)
    
    logger.info(f"Generated {len(alerts)} alerts")
    return alerts


@task(name="save_alerts")
def save_alerts_task(alerts: List[Dict[str, Any]]) -> None:
    """
    Save drift alerts to storage.
    
    Args:
        alerts: List of alert dictionaries
    """
    if not alerts:
        logger.info("No alerts to save")
        return
    
    config = load_config()
    storage = StorageManager(config)
    storage.append_drift_alerts(alerts)
    
    logger.info(f"Saved {len(alerts)} alerts")

