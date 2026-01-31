"""Model utilities for loading and working with BERTopic models."""
from bertopic import BERTopic
from pathlib import Path
from typing import List, Tuple
import numpy as np


def load_bertopic_model(model_path: str) -> BERTopic:
    """
    Load BERTopic model from disk.
    
    Args:
        model_path: Path to saved model file
        
    Returns:
        Loaded BERTopic model
        
    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    return BERTopic.load(model_path)


def predict_topics(model: BERTopic, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict topics for new texts.
    
    Args:
        model: Fitted BERTopic model
        texts: List of text documents
        
    Returns:
        Tuple of (topics, probabilities)
    """
    return model.transform(texts)
