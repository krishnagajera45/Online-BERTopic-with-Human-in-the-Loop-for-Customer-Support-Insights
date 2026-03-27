"""API endpoints for BERTopic model metrics."""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import json
from pathlib import Path

from src.utils import setup_logger

logger = setup_logger(__name__, "logs/api.log")

router = APIRouter()


def _load_bertopic_metrics() -> Dict[str, Any]:
    """Load BERTopic metrics from file, handling both legacy and new format."""
    metrics_path = Path("outputs/metrics/bertopic_metrics.json")
    if not metrics_path.exists():
        return {}
    with open(metrics_path, 'r') as f:
        return json.load(f)


@router.get("/", response_model=Dict[str, Any])
async def get_bertopic_metrics():
    """
    Get BERTopic model evaluation metrics (latest batch).
    
    Returns:
        Dictionary with BERTopic metrics including:
        - coherence_c_v: Topic coherence score
        - diversity: Topic diversity score
        - silhouette_score: Clustering quality score
        - num_topics: Number of topics
        - training_time_seconds: Time taken to train
    """
    try:
        data = _load_bertopic_metrics()
        
        if not data:
            return {
                "status": "not_available",
                "message": "BERTopic metrics not yet computed. Run pipeline to generate metrics.",
                "coherence_c_v": None,
                "diversity": None,
                "silhouette_score": None,
                "num_topics": None,
                "training_time_seconds": None
            }
        
        # New format: use latest; legacy: use root
        metrics = data.get("latest", data)
        if not metrics.get("coherence_c_v") and "batches" in data and data["batches"]:
            latest_batch = data["batches"][-1]
            metrics = {**metrics, **latest_batch}
        
        logger.info(f"Retrieved BERTopic metrics: {metrics.get('num_topics', 0)} topics")
        return metrics
        
    except Exception as e:
        logger.error(f"Error retrieving BERTopic metrics: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve BERTopic metrics: {str(e)}"
        )


@router.get("/history", response_model=Dict[str, Any])
async def get_bertopic_metrics_history():
    """
    Get BERTopic metrics history for temporal analysis (all batches).
    
    Returns:
        Dictionary with batches list for line charts over time.
    """
    try:
        data = _load_bertopic_metrics()
        batches = data.get("batches", [])
        # Migrate legacy flat format to batches
        if not batches and data.get("batch_id"):
            batches = [{
                "batch_id": data.get("batch_id"),
                "coherence_c_v": data.get("coherence_c_v"),
                "diversity": data.get("diversity"),
                "silhouette_score": data.get("silhouette_score"),
                "num_topics": data.get("num_topics"),
                "timestamp": data.get("timestamp"),
                "training_time_seconds": data.get("training_time_seconds"),
            }]
        return {"batches": batches, "status": "ok" if batches else "not_available"}
    except Exception as e:
        logger.error(f"Error retrieving BERTopic metrics history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
