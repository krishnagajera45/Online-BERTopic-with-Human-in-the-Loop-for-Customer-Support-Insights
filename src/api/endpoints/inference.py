"""Inference API endpoints."""
from fastapi import APIRouter, HTTPException
from pathlib import Path
from src.api.models.requests import InferRequest
from src.api.models.responses import InferResponse
from src.utils import setup_logger, clean_text, load_config, load_bertopic_model
from src.utils import StorageManager

router = APIRouter()
logger = setup_logger(__name__)
config = load_config()
storage = StorageManager(config)


@router.post("", response_model=InferResponse)
async def infer_topic(request: InferRequest):
    """Predict topic for input text."""
    try:
        logger.info(f"INFERENCE: Received text ({len(request.text)} chars)")
        logger.debug(f"INFERENCE: Text preview: {request.text[:100]}...")
        
        # Preprocess text
        cleaned_text = clean_text(request.text)
        
        if not cleaned_text:
            logger.warning("INFERENCE: Text is empty after cleaning")
            raise HTTPException(status_code=400, detail="Text is empty after cleaning")
        
        # Load current model
        model_path = Path(config.storage.current_model_path)
        if not model_path.exists():
            logger.error("INFERENCE: No trained model found")
            raise HTTPException(status_code=404, detail="No trained model found")
        
        logger.debug("INFERENCE: Loading model for transform")
        model = load_bertopic_model(str(model_path))
        
        # Transform
        topics, probs = model.transform([cleaned_text])
        topic_id = int(topics[0])
        
        # Get topic info
        topic_words = model.get_topic(topic_id)
        top_words = [word for word, _ in topic_words[:10]] if topic_words else []
        
        # Get topic label from metadata
        topics_metadata = storage.load_topics_metadata()
        topic_label = f"Topic {topic_id}"
        for topic in topics_metadata:
            if topic['topic_id'] == topic_id:
                topic_label = topic.get('custom_label', topic_label)
                break
        
        # Calculate confidence
        confidence = float(probs[0].max()) if len(probs[0]) > 0 else 0.0
        
        logger.info(f"INFERENCE: Result - Topic {topic_id} ({topic_label}), Confidence: {confidence:.3f}")
        
        return InferResponse(
            topic_id=topic_id,
            topic_label=topic_label,
            confidence=confidence,
            top_words=top_words
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in inference: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

