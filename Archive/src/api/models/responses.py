"""Pydantic response models for the API."""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class TopicResponse(BaseModel):
    """Response model for topic information."""
    topic_id: int
    custom_label: str
    top_words: List[str]
    size: int
    created_at: str
    batch_id: str
    window_start: str
    window_end: str
    count: int
    gpt_label: Optional[str] = None
    gpt_summary: Optional[str] = None


class TrendResponse(BaseModel):
    """Response model for topic trends."""
    batch_id: str
    topic_id: int
    count: int
    timestamp: Optional[str] = None


class AlertResponse(BaseModel):
    """Response model for drift alerts."""
    alert_id: str
    topic_id: int
    window_start: str
    severity: str
    reason: str
    metrics_json: str
    created_at: str


class InferResponse(BaseModel):
    """Response model for topic inference."""
    topic_id: int
    topic_label: str
    confidence: float
    top_words: List[str]


class ModelMetadataResponse(BaseModel):
    """Response model for model metadata."""
    model_version: str
    last_updated: str
    num_topics: int
    embedding_model: str
    batch_id: str


class StatusResponse(BaseModel):
    """Generic status response."""
    status: str
    message: str


class PipelineStatusResponse(BaseModel):
    """Response model for pipeline status."""
    last_run: Optional[str]
    last_batch_id: Optional[str]
    documents_processed: Optional[int]
    status: str
    next_scheduled_run: Optional[str] = None

