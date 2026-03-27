"""Pydantic request models for the API."""
from pydantic import BaseModel, Field
from typing import List, Optional


class InferRequest(BaseModel):
    """Request model for topic inference."""
    text: str = Field(..., description="Text to infer topic for")


class MergeRequest(BaseModel):
    """Request model for merging topics."""
    topic_ids: List[int] = Field(..., description="List of topic IDs to merge")
    new_label: Optional[str] = Field(None, description="New label for merged topic")
    note: Optional[str] = Field(None, description="User note about the merge")


class SplitRequest(BaseModel):
    """Request model for splitting a topic."""
    topic_id: int = Field(..., description="Topic ID to split")
    new_topic_ids: List[int] = Field(..., description="New topic IDs after split")
    note: Optional[str] = Field(None, description="User note about the split")


class RelabelRequest(BaseModel):
    """Request model for relabeling a topic."""
    topic_id: int = Field(..., description="Topic ID to relabel")
    new_label: str = Field(..., description="New label for the topic")
    note: Optional[str] = Field(None, description="User note about the relabel")

