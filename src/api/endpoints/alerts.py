"""Drift alerts API endpoints."""
from fastapi import APIRouter, HTTPException
from typing import List
from src.api.models.responses import AlertResponse
from src.storage import StorageManager
from src.utils import setup_logger

router = APIRouter()
logger = setup_logger(__name__)
storage = StorageManager()


@router.get("", response_model=List[AlertResponse])
async def get_alerts(limit: int = 50):
    """Get recent drift alerts."""
    try:
        alerts_df = storage.load_drift_alerts(limit=limit)
        
        if len(alerts_df) == 0:
            return []
        
        return alerts_df.to_dict('records')
    
    except Exception as e:
        logger.error(f"Error getting alerts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/latest", response_model=AlertResponse)
async def get_latest_alert():
    """Get the most recent drift alert."""
    try:
        alerts_df = storage.load_drift_alerts(limit=1)
        
        if len(alerts_df) == 0:
            raise HTTPException(status_code=404, detail="No alerts found")
        
        return alerts_df.iloc[0].to_dict()
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting latest alert: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

