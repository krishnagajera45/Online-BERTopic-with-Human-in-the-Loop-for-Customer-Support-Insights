"""Prefect flow for model training (online learning)."""
from prefect import flow, get_run_logger
from pathlib import Path
import pandas as pd
import numpy as np
from etl.tasks.model_tasks import train_seed_model_task, update_model_online_task, archive_model_task
from src.utils import load_config
from src.utils import StorageManager


@flow(name="model-training-flow", log_prints=True)
def model_training_flow(
    documents: list,
    batch_id: str,
    window_start: str,
    window_end: str
):
    """
    Prefect flow for model training (automatically detects seed vs online update).
    
    This flow automatically:
    1. Checks if model exists
    2. If exists → Archives and updates (online learning)
    3. If not exists → Trains seed model
    4. Saves model and metadata
    5. Saves document assignments
    
    Args:
        documents: List of document texts
        batch_id: Batch identifier
        window_start: Window start date
        window_end: Window end date
        
    Returns:
        Tuple of (topics, probabilities)
    """
    logger = get_run_logger()
    
    logger.info(f"Starting model training flow")
    logger.info(f"Batch: {batch_id}")
    logger.info(f"Window: {window_start} to {window_end}")
    logger.info(f"Documents: {len(documents)}")
    
    config = load_config()
    storage = StorageManager(config)
    
    # Automatically detect if this is initial training or update
    model_exists = Path(config.storage.current_model_path).exists()
    
    if model_exists:
        # Model exists → Update with new data (online learning)
        logger.info("Existing model found → Performing online update")
        
        # Archive current model before update
        logger.info("Archiving current model as previous")
        archive_model_task()
        
        # Update model online
        topics, probs = update_model_online_task(
            documents=documents,
            batch_id=batch_id,
            window_start=window_start,
            window_end=window_end
        )
    else:
        # No model → Train seed model
        logger.info("No existing model found → Training seed model")
        topics, probs = train_seed_model_task(
            documents=documents,
            batch_id=batch_id,
            window_start=window_start,
            window_end=window_end
        )
    
    # Save document assignments
    logger.info("Saving document assignments")
    
    # Calculate confidence scores from probabilities
    # Handle different probability formats from BERTopic
    if probs is None:
        # No probabilities available
        confidence = [0.0] * len(topics)
    elif len(probs.shape) == 2:
        # 2D array: each row is probability distribution for a document
        confidence = [p.max() if len(p) > 0 else 0.0 for p in probs]
    else:
        # 1D array: single probability value per document
        confidence = probs.tolist()
    
    # Create assignments DataFrame (doc_id would come from original data)
    assignments = pd.DataFrame({
        'doc_id': [f'doc_{i}' for i in range(len(documents))],
        'topic_id': topics,
        'timestamp': window_start,
        'batch_id': batch_id,
        'confidence': confidence
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
        "How do I reset my password",
        "Help me fix my RJ-45 connector issue",
        "I have an issue in proxy/firewall in my machine",
        "Lost internet connection",
        "CHATGpt is not working"
    ]
    
    topics, probs = model_training_flow(
        documents=test_docs,
        batch_id="test_batch",
        window_start="2024-01-01",
        window_end="2024-01-02"
    )
    print(f"Topics: {topics}")

