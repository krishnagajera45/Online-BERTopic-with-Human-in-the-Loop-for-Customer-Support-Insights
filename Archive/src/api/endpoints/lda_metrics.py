"""API endpoints for LDA model metrics and comparison."""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import json
from pathlib import Path
from datetime import datetime

from src.utils import setup_logger

logger = setup_logger(__name__, "logs/api.log")

router = APIRouter()


def _load_lda_metrics() -> Dict[str, Any]:
    """Load LDA metrics from file."""
    metrics_path = Path("outputs/metrics/lda_metrics.json")
    if not metrics_path.exists():
        return {}
    with open(metrics_path, 'r') as f:
        return json.load(f)


@router.get("/", response_model=Dict[str, Any])
async def get_lda_metrics():
    """
    Get the latest LDA model evaluation metrics.
    
    Returns:
        Dictionary with LDA metrics including:
        - coherence_c_v: Topic coherence score
        - diversity: Topic diversity score
        - silhouette_score: Clustering quality score
        - num_topics: Number of topics
        - training_time_seconds: Time taken to train
        - topics: List of topics with top words
    """
    try:
        data = _load_lda_metrics()
        
        if not data:
            return {
                "status": "not_available",
                "message": "LDA metrics not yet computed. Run pipeline to generate metrics.",
                "coherence_c_v": 0.0,
                "diversity": 0.0,
                "silhouette_score": 0.0,
                "num_topics": 0,
                "training_time_seconds": 0.0
            }
        
        # New format: use latest; legacy: use root
        metrics = data.get("latest", data)
        if not metrics.get("coherence_c_v") and "batches" in data and data["batches"]:
            latest_batch = data["batches"][-1]
            metrics = {**metrics, **latest_batch}
        
        # Include topics from latest if present
        if "topics" in data.get("latest", {}):
            metrics["topics"] = data["latest"]["topics"]
        elif "topics" in data:
            metrics["topics"] = data["topics"]
        
        logger.info(f"Retrieved LDA metrics: {metrics.get('num_topics', 0)} topics")
        return metrics
        
    except Exception as e:
        logger.error(f"Error retrieving LDA metrics: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve LDA metrics: {str(e)}"
        )


@router.get("/history", response_model=Dict[str, Any])
async def get_lda_metrics_history():
    """
    Get LDA metrics history for temporal analysis (all batches).
    """
    try:
        data = _load_lda_metrics()
        batches = data.get("batches", [])
        # Migrate legacy format to batches
        if not batches and data.get("batch_id"):
            batches = [{
                "batch_id": data.get("batch_id"),
                "coherence_c_v": data.get("coherence_c_v"),
                "diversity": data.get("diversity"),
                "silhouette_score": data.get("silhouette_score"),
                "num_topics": data.get("num_topics"),
                "timestamp": data.get("timestamp"),
                "training_time_seconds": data.get("training_time_seconds") or data.get("total_time_seconds"),
            }]
        return {"batches": batches, "status": "ok" if batches else "not_available"}
    except Exception as e:
        logger.error(f"Error retrieving LDA metrics history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/comparison", response_model=Dict[str, Any])
async def get_model_comparison():
    """
    Get side-by-side comparison of BERTopic and LDA metrics.
    
    Returns:
        Dictionary with comparative metrics for both models
    """
    try:
        # Load LDA metrics (handle new format with "latest")
        lda_data = _load_lda_metrics()
        lda_metrics = lda_data.get("latest", lda_data) if lda_data else {}
        
        # Load BERTopic metrics (handle new format with "latest")
        from src.api.endpoints.bertopic_metrics import _load_bertopic_metrics
        bt_data = _load_bertopic_metrics()
        bertopic_metrics = bt_data.get("latest", bt_data) if bt_data else {}
        
        # Also load document count from topics metadata
        total_documents = 0
        topics_path = Path("outputs/topics/topics_metadata.json")
        
        if topics_path.exists():
            try:
                with open(topics_path, 'r') as f:
                    topics_data = json.load(f)
                    topics = topics_data.get('topics', [])
                    total_documents = sum(t.get('count', 0) for t in topics)
            except Exception as e:
                logger.warning(f"Could not load topics metadata: {e}")
        
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'lda': {
                'coherence_c_v': lda_metrics.get('coherence_c_v', 0.0),
                'diversity': lda_metrics.get('diversity', 0.0),
                'silhouette_score': lda_metrics.get('silhouette_score', 0.0),
                'num_topics': lda_metrics.get('num_topics', 0),
                'training_time_seconds': lda_metrics.get('training_time_seconds', 0.0),
                'status': lda_metrics.get('status', 'not_available')
            },
            'bertopic': {
                'coherence_c_v': bertopic_metrics.get('coherence_c_v', 0.0),
                'diversity': bertopic_metrics.get('diversity', 0.0),
                'silhouette_score': bertopic_metrics.get('silhouette_score', 0.0),
                'num_topics': bertopic_metrics.get('num_topics', 0),
                'training_time_seconds': bertopic_metrics.get('training_time_seconds', 0.0),
                'total_documents': total_documents
            }
        }
        
        logger.info(f"Retrieved comparison: LDA={comparison['lda']['num_topics']} topics, BERTopic={comparison['bertopic']['num_topics']} topics")
        return comparison
        
    except Exception as e:
        logger.error(f"Error generating model comparison: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate comparison: {str(e)}"
        )


@router.get("/topics", response_model=Dict[str, Any])
async def get_lda_topics():
    """
    Get LDA topic details with top words.
    
    Returns:
        Dictionary with LDA topics
    """
    try:
        metrics_path = Path("outputs/metrics/lda_metrics.json")
        
        if not metrics_path.exists():
            raise HTTPException(
                status_code=404,
                detail="LDA metrics not found. Run pipeline first."
            )
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        topics = metrics.get('topics', [])
        
        logger.info(f"Retrieved {len(topics)} LDA topics")
        return {
            'num_topics': len(topics),
            'topics': topics,
            'batch_id': metrics.get('batch_id'),
            'timestamp': metrics.get('timestamp')
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving LDA topics: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve LDA topics: {str(e)}"
        )
