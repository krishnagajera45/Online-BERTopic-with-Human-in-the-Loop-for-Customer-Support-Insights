"""Prefect tasks for drift detection (granular task-level logic)."""
from prefect import task, get_run_logger
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine, jensenshannon
from sklearn.metrics.pairwise import cosine_similarity
from bertopic import BERTopic

from src.utils import load_config, load_drift_thresholds, generate_alert_id, normalize_weights, StorageManager


@task(name="calculate-prevalence-change", retries=1)
def calculate_prevalence_change_task(
    current_model: BERTopic,
    previous_model: BERTopic
) -> Dict[str, Any]:
    """
    Calculate change in topic distribution (prevalence) between models.
    
    Logic moved from DriftDetector.calculate_prevalence_change()
    
    Args:
        current_model: Current BERTopic model
        previous_model: Previous BERTopic model
        
    Returns:
        Dictionary with prevalence change metrics
    """
    logger = get_run_logger()
    logger.info("Calculating topic prevalence change")
    
    try:
        current_topics = current_model.get_topic_info()
        previous_topics = previous_model.get_topic_info()
        
        # Remove outlier topic (-1)
        current_topics = current_topics[current_topics['Topic'] != -1]
        previous_topics = previous_topics[previous_topics['Topic'] != -1]
        
        # Normalize counts to get distributions
        current_dist = current_topics.set_index('Topic')['Count']
        previous_dist = previous_topics.set_index('Topic')['Count']
        
        current_dist = current_dist / current_dist.sum()
        previous_dist = previous_dist / previous_dist.sum()
        
        # Align indices (topics may have different IDs)
        all_topics = sorted(set(current_dist.index) | set(previous_dist.index))
        current_dist = current_dist.reindex(all_topics, fill_value=0)
        previous_dist = previous_dist.reindex(all_topics, fill_value=0)
        
        # Calculate total variation distance
        # TVD = (1/2) × Σ |P(topic_i) - Q(topic_i)|
        prevalence_change = np.abs(current_dist - previous_dist).sum() / 2
        
        # Calculate per-topic changes
        topic_changes = {}
        for topic_id in all_topics:
            change = abs(current_dist[topic_id] - previous_dist[topic_id])
            if change > 0.01:  # Only report significant changes
                topic_changes[int(topic_id)] = {
                    'current_prevalence': float(current_dist[topic_id]),
                    'previous_prevalence': float(previous_dist[topic_id]),
                    'change': float(change)
                }
        
        result = {
            'prevalence_change': float(prevalence_change),
            'topic_changes': topic_changes,
            'num_current_topics': len(current_topics),
            'num_previous_topics': len(previous_topics)
        }
        
        logger.info(f"Prevalence change: {prevalence_change:.4f}")
        return result
    
    except Exception as e:
        logger.error(f"Error calculating prevalence change: {e}", exc_info=True)
        return {'prevalence_change': 0.0, 'topic_changes': {}}


@task(name="calculate-centroid-shift", retries=1)
def calculate_centroid_shift_task(
    current_model: BERTopic,
    previous_model: BERTopic,
    current_docs: List[str],
    previous_docs: List[str]
) -> Dict[int, Dict[str, float]]:
    """
    Calculate centroid shift for each topic in embedding space.
    
    Logic moved from DriftDetector.calculate_centroid_shift()
    
    Args:
        current_model: Current BERTopic model
        previous_model: Previous BERTopic model
        current_docs: Current documents
        previous_docs: Previous documents (can be empty)
        
    Returns:
        Dictionary mapping topic_id to shift metrics (empty dict if no previous docs)
    """
    logger = get_run_logger()
    logger.info("Calculating topic centroid shifts")
    
    # Handle empty previous_docs gracefully
    if not previous_docs or len(previous_docs) == 0:
        logger.warning("No previous documents available for centroid shift calculation. Skipping.")
        return {}
    
    if not current_docs or len(current_docs) == 0:
        logger.warning("No current documents available for centroid shift calculation. Skipping.")
        return {}
    
    try:
        centroid_shifts = {}
        
        # Get topics from current model
        current_topics_info = current_model.get_topic_info()
        current_topics_info = current_topics_info[current_topics_info['Topic'] != -1]
        
        # Get embeddings
        current_embeddings = current_model._extract_embeddings(current_docs)
        current_topic_assignments, _ = current_model.transform(current_docs)
        
        previous_embeddings = previous_model._extract_embeddings(previous_docs)
        previous_topic_assignments, _ = previous_model.transform(previous_docs)
        
        # Calculate centroid for each topic in current model
        for topic_id in current_topics_info['Topic']:
            # Get embeddings for documents in this topic
            current_mask = np.array(current_topic_assignments) == topic_id
            previous_mask = np.array(previous_topic_assignments) == topic_id
            
            if current_mask.sum() > 0 and previous_mask.sum() > 0:
                current_centroid = current_embeddings[current_mask].mean(axis=0)
                previous_centroid = previous_embeddings[previous_mask].mean(axis=0)
                
                # Calculate cosine similarity
                # similarity = (A · B) / (||A|| × ||B||)
                similarity = cosine_similarity(
                    [current_centroid],
                    [previous_centroid]
                )[0][0]
                
                shift = 1 - similarity  # Convert to distance
                
                centroid_shifts[int(topic_id)] = {
                    'centroid_shift': float(shift),
                    'similarity': float(similarity),
                    'current_docs': int(current_mask.sum()),
                    'previous_docs': int(previous_mask.sum())
                }
        
        logger.info(f"Calculated centroid shifts for {len(centroid_shifts)} topics")
        return centroid_shifts
    
    except Exception as e:
        logger.error(f"Error calculating centroid shift: {e}", exc_info=True)
        return {}


@task(name="calculate-keyword-divergence", retries=1)
def calculate_keyword_divergence_task(
    current_model: BERTopic,
    previous_model: BERTopic
) -> Dict[int, Dict[str, Any]]:
    """
    Calculate Jensen-Shannon divergence on topic keyword distributions.
    
    Logic moved from DriftDetector.calculate_keyword_divergence()
    
    Args:
        current_model: Current BERTopic model
        previous_model: Previous BERTopic model
        
    Returns:
        Dictionary mapping topic_id to divergence metrics
    """
    logger = get_run_logger()
    logger.info("Calculating keyword divergence")
    
    try:
        keyword_divergences = {}
        
        # Get topics from current model
        current_topics_info = current_model.get_topic_info()
        current_topics_info = current_topics_info[current_topics_info['Topic'] != -1]
        
        for topic_id in current_topics_info['Topic']:
            try:
                # Get keyword distributions
                current_words = current_model.get_topic(topic_id)
                previous_words = previous_model.get_topic(topic_id)
                
                if current_words and previous_words:
                    # Normalize weights to probability distributions
                    current_weights = normalize_weights(current_words)
                    previous_weights = normalize_weights(previous_words)
                    
                    # Ensure same length
                    max_len = max(len(current_weights), len(previous_weights))
                    current_weights += [0] * (max_len - len(current_weights))
                    previous_weights += [0] * (max_len - len(previous_weights))
                    
                    # Calculate Jensen-Shannon divergence
                    # JS(P||Q) = 0.5 × KL(P||M) + 0.5 × KL(Q||M)
                    # Where M = (P + Q) / 2 (midpoint)
                    js_div = jensenshannon(current_weights, previous_weights)
                    
                    keyword_divergences[int(topic_id)] = {
                        'js_divergence': float(js_div),
                        'current_top_words': [w for w, _ in current_words[:10]],
                        'previous_top_words': [w for w, _ in previous_words[:10]]
                    }
            
            except Exception as e:
                logger.warning(f"Could not calculate divergence for topic {topic_id}: {e}")
                continue
        
        logger.info(f"Calculated keyword divergence for {len(keyword_divergences)} topics")
        return keyword_divergences
    
    except Exception as e:
        logger.error(f"Error calculating keyword divergence: {e}", exc_info=True)
        return {}


@task(name="detect-topic-changes", retries=1)
def detect_topic_changes_task(
    current_model: BERTopic,
    previous_model: BERTopic
) -> Dict[str, List[int]]:
    """
    Detect new and disappeared topics.
    
    Logic moved from DriftDetector.detect_new_and_disappeared_topics()
    
    Args:
        current_model: Current BERTopic model
        previous_model: Previous BERTopic model
        
    Returns:
        Dictionary with 'new_topics' and 'disappeared_topics' lists
    """
    logger = get_run_logger()
    logger.info("Detecting new and disappeared topics")
    
    try:
        current_topics = set(current_model.get_topic_info()['Topic'])
        previous_topics = set(previous_model.get_topic_info()['Topic'])
        
        # Remove outlier topic
        current_topics.discard(-1)
        previous_topics.discard(-1)
        
        new_topics = list(current_topics - previous_topics)  # In current, not in previous
        disappeared_topics = list(previous_topics - current_topics)  # In previous, not in current
        
        logger.info(f"Found {len(new_topics)} new topics, {len(disappeared_topics)} disappeared")
        
        return {
            'new_topics': new_topics,
            'disappeared_topics': disappeared_topics
        }
    
    except Exception as e:
        logger.error(f"Error detecting topic changes: {e}", exc_info=True)
        return {'new_topics': [], 'disappeared_topics': []}


@task(name="generate-drift-alerts", retries=1)
def generate_drift_alerts_task(
    drift_metrics: Dict[str, Any],
    window_start: str
) -> List[Dict[str, Any]]:
    """
    Generate alerts when drift exceeds thresholds.
    
    Logic moved from DriftDetector.generate_drift_alerts()
    
    Args:
        drift_metrics: Dictionary with all drift metrics
        window_start: Window start timestamp
        
    Returns:
        List of alert dictionaries
    """
    logger = get_run_logger()
    logger.info("Generating drift alerts")
    
    # Load thresholds
    drift_config = load_drift_thresholds()
    thresholds = drift_config['thresholds']
    severity_levels = drift_config['severity_levels']
    
    def determine_severity(metric_type: str, value: float) -> str:
        """Determine alert severity based on metric value."""
        if value >= severity_levels['high'][metric_type]:
            return 'high'
        elif value >= severity_levels['medium'][metric_type]:
            return 'medium'
        else:
            return 'low'
    
    alerts = []
    timestamp = datetime.now().isoformat()
    
    # Alert on prevalence change
    prevalence_change = drift_metrics.get('prevalence_change', {}).get('prevalence_change', 0)
    if prevalence_change > thresholds['prevalence_threshold']:
        severity = determine_severity('prevalence_change', prevalence_change)
        
        alerts.append({
            'alert_id': generate_alert_id(),
            'topic_id': -1,  # Global alert
            'window_start': window_start,
            'severity': severity,
            'reason': 'Topic prevalence changed significantly',
            'metrics_json': str({
                'prevalence_change': prevalence_change,
                'threshold': thresholds['prevalence_threshold']
            }),
            'created_at': timestamp
        })
    
    # Alert on centroid shifts
    centroid_shifts = drift_metrics.get('centroid_shifts', {})
    for topic_id, metrics in centroid_shifts.items():
        shift = metrics['centroid_shift']
        if shift > thresholds['centroid_threshold']:
            severity = determine_severity('centroid_shift', shift)
            
            alerts.append({
                'alert_id': generate_alert_id(),
                'topic_id': topic_id,
                'window_start': window_start,
                'severity': severity,
                'reason': 'Topic centroid shifted in embedding space',
                'metrics_json': str(metrics),
                'created_at': timestamp
            })
    
    # Alert on keyword divergence
    keyword_divergences = drift_metrics.get('keyword_divergences', {})
    for topic_id, metrics in keyword_divergences.items():
        js_div = metrics['js_divergence']
        if js_div > thresholds['js_divergence_threshold']:
            severity = determine_severity('js_divergence', js_div)
            
            alerts.append({
                'alert_id': generate_alert_id(),
                'topic_id': topic_id,
                'window_start': window_start,
                'severity': severity,
                'reason': 'Topic keyword distribution shifted significantly',
                'metrics_json': str(metrics),
                'created_at': timestamp
            })
    
    # Alert on new/disappeared topics
    topic_changes = drift_metrics.get('topic_changes', {})
    new_topics = topic_changes.get('new_topics', [])
    disappeared_topics = topic_changes.get('disappeared_topics', [])
    
    if len(new_topics) > thresholds['new_topic_threshold']:
        alerts.append({
            'alert_id': generate_alert_id(),
            'topic_id': -1,
            'window_start': window_start,
            'severity': 'medium',
            'reason': f'{len(new_topics)} new topics appeared',
            'metrics_json': str({'new_topics': new_topics}),
            'created_at': timestamp
        })
    
    if len(disappeared_topics) > thresholds['disappeared_topic_threshold']:
        alerts.append({
            'alert_id': generate_alert_id(),
            'topic_id': -1,
            'window_start': window_start,
            'severity': 'medium',
            'reason': f'{len(disappeared_topics)} topics disappeared',
            'metrics_json': str({'disappeared_topics': disappeared_topics}),
            'created_at': timestamp
        })
    
    logger.info(f"Generated {len(alerts)} drift alerts")
    return alerts


@task(name="save-drift-alerts", retries=1)
def save_drift_alerts_task(alerts: List[Dict[str, Any]]) -> None:
    """
    Save drift alerts to storage.
    
    Args:
        alerts: List of alert dictionaries
    """
    logger = get_run_logger()
    
    if not alerts:
        logger.info("No alerts to save")
        return
    
    config = load_config()
    storage = StorageManager(config)
    storage.append_drift_alerts(alerts)
    
    logger.info(f"Saved {len(alerts)} alerts")


@task(name="load-bertopic-models", retries=1)
def load_bertopic_models_task(
    current_model_path: str,
    previous_model_path: str
) -> tuple:
    """
    Load current and previous BERTopic models.
    
    Args:
        current_model_path: Path to current model
        previous_model_path: Path to previous model
        
    Returns:
        Tuple of (current_model, previous_model)
    """
    logger = get_run_logger()
    logger.info("Loading BERTopic models for drift detection")
    
    from pathlib import Path
    
    if not Path(current_model_path).exists():
        raise FileNotFoundError(f"Current model not found: {current_model_path}")
    
    if not Path(previous_model_path).exists():
        raise FileNotFoundError(f"Previous model not found: {previous_model_path}")
    
    current_model = BERTopic.load(current_model_path)
    previous_model = BERTopic.load(previous_model_path)
    
    logger.info(f"Loaded models from {current_model_path} and {previous_model_path}")
    return current_model, previous_model
