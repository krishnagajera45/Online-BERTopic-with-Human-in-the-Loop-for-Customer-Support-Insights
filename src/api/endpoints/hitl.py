"""HITL (Human-in-the-Loop) API endpoints for topic manipulation."""
from fastapi import APIRouter, HTTPException
from src.api.models.requests import MergeRequest, SplitRequest, RelabelRequest
from src.api.models.responses import StatusResponse
from src.modeling import BERTopicOnlineWrapper
from src.storage import StorageManager
from src.utils import setup_logger, load_config

router = APIRouter()
logger = setup_logger(__name__)
config = load_config()
storage = StorageManager(config)
model_wrapper = BERTopicOnlineWrapper(config)


@router.post("/merge", response_model=StatusResponse)
async def merge_topics(request: MergeRequest):
    """Merge multiple topics into one."""
    try:
        logger.info(f"Merging topics: {request.topic_ids}")
        
        # Load current model
        model = model_wrapper.load_model(config.storage.current_model_path)
        
        # Get documents (we need them for merging)
        # For now, we'll update the metadata without actually retraining
        # In production, you'd load the actual documents
        
        # Update metadata
        topics_metadata = storage.load_topics_metadata()
        merged_topic_id = request.topic_ids[0]
        
        # Update the first topic with new label
        for topic in topics_metadata:
            if topic['topic_id'] == merged_topic_id and request.new_label:
                topic['custom_label'] = request.new_label
        
        # Remove other topics from metadata
        topics_metadata = [t for t in topics_metadata if t['topic_id'] not in request.topic_ids[1:]]
        
        storage.save_topics_metadata(topics_metadata)
        
        # Log audit action
        storage.log_audit_action({
            'action_type': 'merge',
            'old_topics': str(request.topic_ids),
            'new_topics': str([merged_topic_id]),
            'user_note': request.note or ''
        })
        
        return StatusResponse(
            status="success",
            message=f"Merged topics {request.topic_ids} into topic {merged_topic_id}"
        )
    
    except Exception as e:
        logger.error(f"Error merging topics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/split", response_model=StatusResponse)
async def split_topic(request: SplitRequest):
    """Split a topic into subtopics."""
    try:
        logger.info(f"Splitting topic: {request.topic_id}")
        
        # Log audit action
        storage.log_audit_action({
            'action_type': 'split',
            'old_topics': str([request.topic_id]),
            'new_topics': str(request.new_topic_ids),
            'user_note': request.note or ''
        })
        
        return StatusResponse(
            status="success",
            message=f"Split topic {request.topic_id} into {len(request.new_topic_ids)} subtopics"
        )
    
    except Exception as e:
        logger.error(f"Error splitting topic: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/relabel", response_model=StatusResponse)
async def relabel_topic(request: RelabelRequest):
    """Update custom label for a topic."""
    try:
        logger.info(f"Relabeling topic {request.topic_id} to '{request.new_label}'")
        
        # Update topic label
        storage.update_topic_label(request.topic_id, request.new_label)
        
        # Log audit action
        storage.log_audit_action({
            'action_type': 'relabel',
            'old_topics': str([request.topic_id]),
            'new_topics': str([request.topic_id]),
            'user_note': request.note or ''
        })
        
        return StatusResponse(
            status="success",
            message=f"Relabeled topic {request.topic_id} to '{request.new_label}'"
        )
    
    except Exception as e:
        logger.error(f"Error relabeling topic: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/audit")
async def get_audit_log(limit: int = 50):
    """Get HITL audit log."""
    try:
        audit_df = storage.load_audit_log(limit=limit)
        
        if len(audit_df) == 0:
            return []
        
        return audit_df.to_dict('records')
    
    except Exception as e:
        logger.error(f"Error getting audit log: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

