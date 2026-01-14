"""Prefect tasks for model training."""
from prefect import task
from typing import List, Tuple
import numpy as np
from src.modeling import BERTopicOnlineWrapper
from src.utils import setup_logger, load_config

logger = setup_logger(__name__, "logs/prefect_tasks.log")


@task(name="train_seed_model", retries=1)
def train_seed_model_task(
    documents: List[str],
    batch_id: str,
    window_start: str,
    window_end: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train initial seed BERTopic model.
    
    Args:
        documents: List of document texts
        batch_id: Batch identifier
        window_start: Window start date
        window_end: Window end date
        
    Returns:
        Tuple of (topics, probabilities)
    """
    logger.info(f"Training seed model on {len(documents)} documents")
    
    config = load_config()
    model_wrapper = BERTopicOnlineWrapper(config)
    
    topics, probs = model_wrapper.train_seed_model(
        documents=documents,
        batch_id=batch_id,
        window_start=window_start,
        window_end=window_end
    )
    
    logger.info(f"Seed model trained with {len(set(topics))} topics")
    return topics, probs


@task(name="update_model_online", retries=1)
def update_model_online_task(
    documents: List[str],
    batch_id: str,
    window_start: str,
    window_end: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Update model with new batch (online learning).
    
    Args:
        documents: New documents
        batch_id: Batch identifier
        window_start: Window start
        window_end: Window end
        
    Returns:
        Tuple of (topics, probabilities)
    """
    logger.info(f"Updating model with {len(documents)} new documents")
    
    config = load_config()
    model_wrapper = BERTopicOnlineWrapper(config)
    
    topics, probs = model_wrapper.update_model_online(
        new_documents=documents,
        batch_id=batch_id,
        window_start=window_start,
        window_end=window_end
    )
    
    logger.info(f"Model updated with {len(set(topics))} topics")
    return topics, probs


@task(name="archive_model")
def archive_model_task() -> None:
    """Archive current model as previous version."""
    import shutil
    from pathlib import Path
    
    config = load_config()
    current_path = Path(config.storage.current_model_path)
    previous_path = Path(config.storage.previous_model_path)
    
    if current_path.exists():
        previous_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(current_path, previous_path)
        logger.info(f"Archived model: {current_path} -> {previous_path}")
    else:
        logger.warning("No current model to archive")

