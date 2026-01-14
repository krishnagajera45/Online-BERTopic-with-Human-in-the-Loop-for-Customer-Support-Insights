"""Prefect flow for model training (online learning)."""
from prefect import flow
from pathlib import Path
import pandas as pd
from etl.tasks.model_tasks import train_seed_model_task, update_model_online_task, archive_model_task
from src.utils import setup_logger, load_config
from src.storage import StorageManager

logger = setup_logger(__name__, "logs/prefect_flows.log")


@flow(name="model-training-flow", log_prints=True)
def model_training_flow(
    documents: list,
    batch_id: str,
    window_start: str,
    window_end: str,
    is_initial: bool = False
):
    """
    Prefect flow for model training (seed or online update).
    
    This flow:
    1. Checks if model exists
    2. Archives current model (if update)
    3. Trains/updates model
    4. Saves model and metadata
    5. Saves document assignments
    
    Args:
        documents: List of document texts
        batch_id: Batch identifier
        window_start: Window start date
        window_end: Window end date
        is_initial: Whether this is initial training
        
    Returns:
        Tuple of (topics, probabilities)
    """
    logger.info(f"Starting model training flow")
    logger.info(f"Batch: {batch_id}")
    logger.info(f"Window: {window_start} to {window_end}")
    logger.info(f"Documents: {len(documents)}")
    logger.info(f"Initial training: {is_initial}")
    
    config = load_config()
    storage = StorageManager(config)
    
    # Check if this is initial training or update
    model_exists = Path(config.storage.current_model_path).exists()
    
    if model_exists and not is_initial:
        # Archive current model before update
        logger.info("Archiving current model")
        archive_model_task()
        
        # Update model online
        topics, probs = update_model_online_task(
            documents=documents,
            batch_id=batch_id,
            window_start=window_start,
            window_end=window_end
        )
    else:
        # Train seed model
        logger.info("Training seed model")
        topics, probs = train_seed_model_task(
            documents=documents,
            batch_id=batch_id,
            window_start=window_start,
            window_end=window_end
        )
    
    # Save document assignments
    logger.info("Saving document assignments")
    # Create assignments DataFrame (doc_id would come from original data)
    assignments = pd.DataFrame({
        'doc_id': [f'doc_{i}' for i in range(len(documents))],
        'topic_id': topics,
        'timestamp': window_start,
        'batch_id': batch_id,
        'confidence': [p.max() if len(p) > 0 else 0.0 for p in probs]
    })
    storage.append_doc_assignments(assignments)
    
    logger.info(f"Model training flow completed")
    logger.info(f"Topics discovered: {len(set(topics))}")
    
    return topics, probs


if __name__ == "__main__":
    # Test the flow
    test_docs = [
        "I need help with billing",
        "My app keeps crashing",
        "How do I reset my password"
    ]
    
    topics, probs = model_training_flow(
        documents=test_docs,
        batch_id="test_batch",
        window_start="2024-01-01",
        window_end="2024-01-02",
        is_initial=True
    )
    print(f"Topics: {topics}")

