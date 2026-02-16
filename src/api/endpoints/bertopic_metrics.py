"""API endpoints for BERTopic model metrics."""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import json
from pathlib import Path

from src.utils import setup_logger

logger = setup_logger(__name__, "logs/api.log")

router = APIRouter()


@router.get("/", response_model=Dict[str, Any])
async def get_bertopic_metrics():
    """
    Get BERTopic model evaluation metrics.
    
    Returns:
        Dictionary with BERTopic metrics including:
        - coherence_c_v: Topic coherence score
        - diversity: Topic diversity score
        - silhouette_score: Clustering quality score
        - num_topics: Number of topics
        - training_time_seconds: Time taken to train
    """
    try:
        metrics_path = Path("outputs/metrics/bertopic_metrics.json")
        
        if not metrics_path.exists():
            logger.warning("BERTopic metrics file not found")
            return {
                "status": "not_available",
                "message": "BERTopic metrics not yet computed. Run pipeline to generate metrics.",
                "coherence_c_v": None,
                "diversity": None,
                "silhouette_score": None,
                "num_topics": None,
                "training_time_seconds": None
            }
        
        # Load metrics from file
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        logger.info(f"Retrieved BERTopic metrics: {metrics.get('num_topics', 0)} topics")
        return metrics
        
    except Exception as e:
        logger.error(f"Error retrieving BERTopic metrics: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve BERTopic metrics: {str(e)}"
        )
