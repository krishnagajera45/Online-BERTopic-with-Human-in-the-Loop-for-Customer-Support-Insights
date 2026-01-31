"""Topics API endpoints."""
from fastapi import APIRouter, HTTPException
from typing import List
from src.api.models.responses import TopicResponse
from src.utils import StorageManager
from src.utils import setup_logger

router = APIRouter()
logger = setup_logger(__name__)
storage = StorageManager()


@router.get("/current", response_model=List[TopicResponse])
async def get_current_topics():
    """Get current model topics with labels, keywords, and sizes."""
    try:
        logger.debug("Fetching current topics metadata")
        topics = storage.load_topics_metadata()
        logger.info(f"Retrieved {len(topics)} topics")
        return topics
    except Exception as e:
        logger.error(f"Error getting current topics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{topic_id}", response_model=TopicResponse)
async def get_topic_details(topic_id: int):
    """Get details for a specific topic."""
    try:
        logger.debug(f"Fetching details for topic_id={topic_id}")
        topics = storage.load_topics_metadata()
        
        for topic in topics:
            if topic['topic_id'] == topic_id:
                logger.info(f"Found topic {topic_id}: {topic.get('label', 'N/A')}")
                return topic
        
        logger.warning(f"Topic {topic_id} not found")
        raise HTTPException(status_code=404, detail=f"Topic {topic_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting topic details: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{topic_id}/examples")
async def get_topic_examples(topic_id: int, limit: int = 10):
    """Get example documents for a topic."""
    try:
        logger.debug(f"Fetching examples for topic_id={topic_id}, limit={limit}")
        assignments = storage.load_doc_assignments(topic_id=topic_id)
        
        if len(assignments) == 0:
            logger.info(f"No examples found for topic {topic_id}")
            return []
        
        # Return top examples sorted by confidence
        examples = assignments.nlargest(limit, 'confidence')
        logger.info(f"Retrieved {len(examples)} examples for topic {topic_id}")
        return examples.to_dict('records')
    
    except Exception as e:
        logger.error(f"Error getting topic examples: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

