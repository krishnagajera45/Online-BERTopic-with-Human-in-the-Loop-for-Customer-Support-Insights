"""Prefect flows package."""
from .data_ingestion import data_ingestion_flow
from .model_training import model_training_flow
from .drift_detection import drift_detection_flow
from .complete_pipeline import complete_pipeline_flow

__all__ = [
    'data_ingestion_flow',
    'model_training_flow',
    'drift_detection_flow',
    'complete_pipeline_flow'
]

