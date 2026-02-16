"""Prefect flow for LDA model training and comparison."""
from prefect import flow, get_run_logger
from typing import List, Dict, Any
import time
from datetime import datetime

from src.etl.tasks.lda_tasks import (
    preprocess_documents_for_lda_task,
    train_lda_model_task,
    calculate_coherence_task,
    calculate_diversity_task,
    calculate_silhouette_task,
    extract_lda_metadata_task,
    save_lda_metrics_task
)
from src.utils import load_config


@flow(name="lda-comparison-flow", log_prints=True)
def lda_comparison_flow(
    documents: List[str],
    num_topics: int,
    batch_id: str,
    window_start: str,
    window_end: str
) -> Dict[str, Any]:
    """
    Train LDA model and calculate evaluation metrics for comparison with BERTopic.
    
    This flow:
    1. Preprocesses documents (tokenization, stopwords, lemmatization)
    2. Trains LDA model with specified number of topics
    3. Calculates evaluation metrics (coherence, diversity, silhouette)
    4. Saves metrics for dashboard display
    
    Args:
        documents: List of document texts (from upstream data processing)
        num_topics: Number of topics to train (typically same as BERTopic)
        batch_id: Batch identifier
        window_start: Window start date
        window_end: Window end date
        
    Returns:
        Dictionary with LDA metrics
    """
    logger = get_run_logger()
    
    logger.info("=" * 80)
    logger.info("Starting LDA Comparison Flow")
    logger.info("=" * 80)
    logger.info(f"Batch: {batch_id}")
    logger.info(f"Window: {window_start} to {window_end}")
    logger.info(f"Documents: {len(documents)}")
    logger.info(f"Target Topics: {num_topics}")
    
    flow_start_time = time.time()
    
    try:
        config = load_config()
        
        # Step 1: Preprocess documents for LDA
        logger.info("Step 1: Preprocessing documents for LDA")
        step1_start = time.time()
        processed_docs, dictionary, corpus = preprocess_documents_for_lda_task(documents)
        step1_duration = time.time() - step1_start
        logger.info(f"Preprocessing completed in {step1_duration:.2f}s")
        
        # Check if we have enough data
        if len(corpus) < 10:
            logger.warning(f"Too few documents after preprocessing ({len(corpus)}). Skipping LDA training.")
            return {
                'status': 'skipped',
                'reason': 'insufficient_documents',
                'documents_processed': len(corpus)
            }
        
        # Step 2: Train LDA model
        logger.info("Step 2: Training LDA model")
        step2_start = time.time()
        
        # Get LDA configuration with safe defaults
        if hasattr(config, 'lda'):
            lda_passes = getattr(config.lda, 'passes', 10)
            lda_iterations = getattr(config.lda, 'iterations', 200)
        else:
            logger.warning("LDA config not found, using defaults")
            lda_passes = 10
            lda_iterations = 200
        
        lda_model = train_lda_model_task(
            corpus=corpus,
            dictionary=dictionary,
            num_topics=num_topics,
            passes=lda_passes,
            iterations=lda_iterations
        )
        step2_duration = time.time() - step2_start
        logger.info(f"LDA training completed in {step2_duration:.2f}s")
        
        # Step 3: Calculate evaluation metrics
        logger.info("Step 3: Calculating evaluation metrics")
        step3_start = time.time()
        
        # Calculate coherence (C_v)
        logger.info("Calculating coherence (C_v)...")
        coherence_cv = calculate_coherence_task(
            model=lda_model,
            texts=processed_docs,
            dictionary=dictionary,
            coherence_type='c_v'
        )
        
        # Calculate diversity
        logger.info("Calculating topic diversity...")
        diversity = calculate_diversity_task(
            model=lda_model,
            top_n=config.model.top_n_words
        )
        
        # Calculate silhouette score
        logger.info("Calculating silhouette score...")
        silhouette = calculate_silhouette_task(
            corpus=corpus,
            model=lda_model,
            dictionary=dictionary
        )
        
        step3_duration = time.time() - step3_start
        logger.info(f"Metrics calculation completed in {step3_duration:.2f}s")
        
        # Step 4: Extract model metadata
        logger.info("Step 4: Extracting model metadata")
        metadata = extract_lda_metadata_task(
            model=lda_model,
            corpus=corpus,
            top_n=config.model.top_n_words
        )
        
        # Step 5: Compile metrics
        flow_duration = time.time() - flow_start_time
        
        metrics = {
            'status': 'success',
            'batch_id': batch_id,
            'window_start': window_start,
            'window_end': window_end,
            'timestamp': datetime.now().isoformat(),
            
            # Model info
            'num_topics': num_topics,
            'num_documents': len(corpus),
            'vocabulary_size': len(dictionary),
            
            # Evaluation metrics
            'coherence_c_v': float(coherence_cv),
            'diversity': float(diversity),
            'silhouette_score': float(silhouette),
            
            # Timing
            'preprocessing_time_seconds': step1_duration,
            'training_time_seconds': step2_duration,
            'evaluation_time_seconds': step3_duration,
            'total_time_seconds': flow_duration,
            
            # Model metadata
            'topics': metadata['topics'],
            
            # Configuration
            'lda_config': {
                'passes': lda_passes,
                'iterations': lda_iterations,
                'top_n_words': config.model.top_n_words
            }
        }
        
        # Step 6: Save metrics
        logger.info("Step 5: Saving metrics")
        save_lda_metrics_task(metrics)
        
        # Log summary
        logger.info("=" * 80)
        logger.info("LDA Comparison Flow Completed Successfully")
        logger.info("=" * 80)
        logger.info(f"Total Duration: {flow_duration:.2f}s")
        logger.info(f"Topics: {num_topics}")
        logger.info(f"Coherence (C_v): {coherence_cv:.4f}")
        logger.info(f"Diversity: {diversity:.4f}")
        logger.info(f"Silhouette Score: {silhouette:.4f}")
        logger.info("=" * 80)
        
        return metrics
        
    except Exception as e:
        logger.error(f"LDA comparison flow failed: {e}", exc_info=True)
        
        # Return error metrics
        return {
            'status': 'error',
            'error_message': str(e),
            'batch_id': batch_id,
            'timestamp': datetime.now().isoformat()
        }


if __name__ == "__main__":
    # Test the flow
    test_docs = [
        "I need help with billing and payment issues",
        "My application keeps crashing on startup",
        "How do I reset my password and recover account",
        "Network connectivity problems with firewall",
        "Database connection timeout errors",
        "User interface is not responsive on mobile",
        "Email notifications are not being sent",
        "Login authentication failing repeatedly"
    ] * 10  # Duplicate to have enough docs
    
    metrics = lda_comparison_flow(
        documents=test_docs,
        num_topics=5,
        batch_id="test_batch",
        window_start="2024-01-01",
        window_end="2024-01-02"
    )
    
    print("\nLDA Metrics:")
    print(f"Coherence: {metrics.get('coherence_c_v', 'N/A')}")
    print(f"Diversity: {metrics.get('diversity', 'N/A')}")
    print(f"Silhouette: {metrics.get('silhouette_score', 'N/A')}")
