"""Prefect flow for model training (batch retrain + merge_models)."""
from prefect import flow, get_run_logger
from pathlib import Path
import pandas as pd
import numpy as np
import json
from src.etl.tasks.model_tasks import (
    train_seed_model_task,
    train_batch_and_merge_models_task,
    archive_model_task
)
from src.utils import load_config
from src.utils import StorageManager


@flow(name="model-training-flow", log_prints=True)
def model_training_flow(
    documents: list,
    doc_ids: list,
    batch_id: str,
    window_start: str,
    window_end: str
):
    """
    Prefect flow for model training using batch retrain + merge_models approach.
    
    This flow implements the recommended pattern:
    1. If first run (no model exists):
       - Train seed model on batch
    2. If model exists:
       - Train fresh model on new batch
       - Merge with base model (which includes all HITL merges/splits)
       - Save merged as new base
    
    The base model is always the merged artifact that includes:
    - All HITL merges/splits from users
    - Topics from all previous batches
    
    Args:
        documents: List of document texts
        batch_id: Batch identifier
        window_start: Window start date
        window_end: Window end date
        
    Returns:
        Tuple of (topics, probabilities)
    """
    logger = get_run_logger()
    
    logger.info(f"Starting model training flow (batch retrain + merge_models)")
    logger.info(f"Batch: {batch_id}")
    logger.info(f"Window: {window_start} to {window_end}")
    logger.info(f"Documents: {len(documents)}")
    
    config = load_config()
    storage = StorageManager(config)
    
    # Automatically detect if this is initial training or batch retrain + merge
    model_exists = Path(config.storage.current_model_path).exists()
    
    if model_exists:
        # Model exists → Use batch retrain + merge_models pattern
        logger.info("Existing model found → Using batch retrain + merge_models approach")
        logger.info("This will train fresh on batch and merge with base (includes HITL changes)")
        
        topics, probs = train_batch_and_merge_models_task(
            documents=documents,
            batch_id=batch_id,
            window_start=window_start,
            window_end=window_end
            # min_similarity now loaded from config.model.min_similarity
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
    

    # Always require doc_ids and documents from processed batch file for alignment
    if not doc_ids or not documents:
        raise ValueError("doc_ids and documents must be provided from the processed batch file.")
    if len(doc_ids) != len(documents):
        logger.error(f"Mismatch: {len(doc_ids)} doc_ids vs {len(documents)} documents. Aborting assignments save.")
        raise ValueError("Length of doc_ids and documents must match for assignments.")
    if len(topics) != len(doc_ids):
        logger.error(f"Mismatch: {len(topics)} topics vs {len(doc_ids)} doc_ids. Aborting assignments save.")
        raise ValueError("Length of topics and doc_ids must match for assignments.")
    # Create assignments for current batch
    batch_assignments = pd.DataFrame({
        'doc_id': doc_ids,
        'topic_id': topics,
        'timestamp': window_start,
        'batch_id': batch_id,
        'confidence': confidence
    })
    
    # Maintain cumulative assignments and corpus for HITL operations
    # Both must contain ALL documents from ALL batches to match model.topics_ array
    assignments_path = "outputs/assignments/doc_assignments.csv"
    model_corpus_path = str(Path(config.storage.current_model_path).parent / (Path(config.storage.current_model_path).stem + "_corpus.json"))
    
    try:
        # Load and append to existing assignments
        if Path(assignments_path).exists():
            existing_assignments = pd.read_csv(assignments_path)
            cumulative_assignments = pd.concat([existing_assignments, batch_assignments], ignore_index=True)
            logger.info(f"Appended {len(batch_assignments)} new assignments to existing {len(existing_assignments)}")
        else:
            cumulative_assignments = batch_assignments
            logger.info(f"Created new assignments file with {len(batch_assignments)} rows")
        
        # Save cumulative assignments
        cumulative_assignments.to_csv(assignments_path, index=False)
        logger.info(f"Saved cumulative assignments: {len(cumulative_assignments)} total rows")
        
        # Load and append to existing corpus
        existing_corpus = []
        if Path(model_corpus_path).exists():
            with open(model_corpus_path, 'r') as f:
                existing_corpus = json.load(f)
            logger.info(f"Loaded existing corpus: {len(existing_corpus)} documents")
        
        # Append current batch documents to corpus
        cumulative_corpus = existing_corpus + documents
        
        # Save cumulative corpus
        with open(model_corpus_path, 'w') as f:
            json.dump(cumulative_corpus, f)
        logger.info(f"Saved cumulative corpus: {len(cumulative_corpus)} total documents ({len(documents)} new)")
        
        # Verify alignment between corpus, assignments, and model
        logger.info("=" * 70)
        logger.info("ALIGNMENT VERIFICATION")
        logger.info("=" * 70)
        logger.info(f"Cumulative corpus:      {len(cumulative_corpus)} documents")
        logger.info(f"Cumulative assignments: {len(cumulative_assignments)} rows")
        logger.info(f"Current batch:          {len(documents)} documents")
        
        if len(cumulative_corpus) == len(cumulative_assignments):
            logger.info(f"✅ PERFECT ALIGNMENT: Corpus and assignments match!")
            logger.info("   HITL operations (merge/split) will work correctly")
        else:
            error_msg = f"❌ ALIGNMENT MISMATCH: {len(cumulative_corpus)} corpus vs {len(cumulative_assignments)} assignments"
            logger.error(error_msg)
            logger.error("   HITL operations may fail! Run reconstruct_corpus.py to fix.")
            raise ValueError(error_msg)
        
        logger.info("=" * 70)
            
    except Exception as e:
        logger.error(f"Failed to save cumulative data: {e}", exc_info=True)
        raise
    
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
        doc_ids=[f"doc_{i}" for i in range(len(test_docs))],
        batch_id="test_batch",
        window_start="2024-01-01",
        window_end="2024-01-02"
    )
    print(f"Topics: {topics}")

