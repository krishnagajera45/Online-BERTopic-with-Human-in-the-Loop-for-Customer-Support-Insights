"""
Drift Detection Module for Topic Modeling.

Detects topic drift using:
1. Topic prevalence change (distribution shift)
2. Topic centroid shift in embedding space (cosine similarity)
3. Keyword distribution shift (Jensen-Shannon divergence)
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from scipy.spatial.distance import cosine, jensenshannon
from sklearn.metrics.pairwise import cosine_similarity

from bertopic import BERTopic
from src.utils import setup_logger, load_drift_thresholds, generate_alert_id, normalize_weights
from src.storage import StorageManager

logger = setup_logger(__name__, "logs/drift.log")


class DriftDetector:
    """Detect and analyze topic drift between model versions."""
    
    def __init__(self, config: Any = None):
        """
        Initialize drift detector.
        
        Args:
            config: Configuration object
        """
        self.thresholds = load_drift_thresholds()['thresholds']
        self.severity_levels = load_drift_thresholds()['severity_levels']
        self.storage = StorageManager(config)
        logger.info("DriftDetector initialized")
    
    def calculate_prevalence_change(
        self,
        current_model: BERTopic,
        previous_model: BERTopic
    ) -> Dict[str, Any]:
        """
        Calculate change in topic distribution (prevalence) between models.
        
        Args:
            current_model: Current BERTopic model
            previous_model: Previous BERTopic model
            
        Returns:
            Dictionary with prevalence change metrics
        """
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
            prevalence_change = np.abs(current_dist - previous_dist).sum() / 2
            
            #TVD = (1/2) × Σ |P(topic_i) - Q(topic_i)|
            

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
    
    def calculate_centroid_shift(
        self,
        current_model: BERTopic,
        previous_model: BERTopic,
        current_docs: List[str],
        previous_docs: List[str]
    ) -> Dict[int, Dict[str, float]]:
        """
        Calculate centroid shift for each topic in embedding space.
        
        Args:
            current_model: Current BERTopic model
            previous_model: Previous BERTopic model
            current_docs: Current documents
            previous_docs: Previous documents
            
        Returns:
            Dictionary mapping topic_id to shift metrics
        """
        logger.info("Calculating topic centroid shifts")
        
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
                    #similarity = (A · B) / (||A|| × ||B||)
                    """Where:
                        A · B = dot product
                        ||A|| = magnitude of A"""

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
    
    def calculate_keyword_divergence(
        self,
        current_model: BERTopic,
        previous_model: BERTopic
    ) -> Dict[int, Dict[str, Any]]:
        """
        Calculate Jensen-Shannon divergence on topic keyword distributions.
        
        Args:
            current_model: Current BERTopic model
            previous_model: Previous BERTopic model
            
        Returns:
            Dictionary mapping topic_id to divergence metrics
        """
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
                        
                        """_summary_JS(P||Q) = 0.5 × KL(P||M) + 0.5 × KL(Q||M)
                            Where:
                            M = (P + Q) / 2  (midpoint)
                            KL = Kullback-Leibler divergence"""
                    
                        # Calculate Jensen-Shannon divergence
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
    
    def detect_new_and_disappeared_topics(
        self,
        current_model: BERTopic,
        previous_model: BERTopic
    ) -> Dict[str, List[int]]:
        """
        Detect new and disappeared topics.
        
        Args:
            current_model: Current BERTopic model
            previous_model: Previous BERTopic model
            
        Returns:
            Dictionary with 'new_topics' and 'disappeared_topics' lists
        """
        logger.info("Detecting new and disappeared topics")
        
        try:
            current_topics = set(current_model.get_topic_info()['Topic'])
            previous_topics = set(previous_model.get_topic_info()['Topic'])
            
            # Remove outlier topic
            current_topics.discard(-1)
            previous_topics.discard(-1)
            
            new_topics = list(current_topics - previous_topics)# In current, not in previous
            disappeared_topics = list(previous_topics - current_topics)# In previous, not in current
            
            logger.info(f"Found {len(new_topics)} new topics, {len(disappeared_topics)} disappeared")
            
            return {
                'new_topics': new_topics,
                'disappeared_topics': disappeared_topics
            }
        
        except Exception as e:
            logger.error(f"Error detecting topic changes: {e}", exc_info=True)
            return {'new_topics': [], 'disappeared_topics': []}
    
    def generate_drift_alerts(
        self,
        drift_metrics: Dict[str, Any],
        window_start: str
    ) -> List[Dict[str, Any]]:
        """
        Generate alerts when drift exceeds thresholds.
        
        Args:
            drift_metrics: Dictionary with all drift metrics
            window_start: Window start timestamp
            
        Returns:
            List of alert dictionaries
        """
        logger.info("Generating drift alerts")
        
        alerts = []
        timestamp = datetime.now().isoformat()
        
        # Alert on prevalence change
        prevalence_change = drift_metrics.get('prevalence_change', {}).get('prevalence_change', 0)
        if prevalence_change > self.thresholds['prevalence_threshold']:
            severity = self._determine_severity(
                'prevalence_change',
                prevalence_change
            )
            
            alerts.append({
                'alert_id': generate_alert_id(),
                'topic_id': -1,  # Global alert
                'window_start': window_start,
                'severity': severity,
                'reason': 'Topic prevalence changed significantly',
                'metrics_json': str({
                    'prevalence_change': prevalence_change,
                    'threshold': self.thresholds['prevalence_threshold']
                }),
                'created_at': timestamp
            })
        
        # Alert on centroid shifts
        centroid_shifts = drift_metrics.get('centroid_shifts', {})
        for topic_id, metrics in centroid_shifts.items():
            shift = metrics['centroid_shift']
            if shift > self.thresholds['centroid_threshold']:
                severity = self._determine_severity('centroid_shift', shift)
                
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
            if js_div > self.thresholds['js_divergence_threshold']:
                severity = self._determine_severity('js_divergence', js_div)
                
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
        
        if len(new_topics) > self.thresholds['new_topic_threshold']:
            alerts.append({
                'alert_id': generate_alert_id(),
                'topic_id': -1,
                'window_start': window_start,
                'severity': 'medium',
                'reason': f'{len(new_topics)} new topics appeared',
                'metrics_json': str({'new_topics': new_topics}),
                'created_at': timestamp
            })
        
        if len(disappeared_topics) > self.thresholds['disappeared_topic_threshold']:
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
    
    def _determine_severity(self, metric_type: str, value: float) -> str:
        """Determine alert severity based on metric value."""
        levels = self.severity_levels
        
        if value >= levels['high'][metric_type]:
            return 'high'
        elif value >= levels['medium'][metric_type]:
            return 'medium'
        else:
            return 'low'
    
    def run_full_drift_detection(
        self,
        current_model: BERTopic,
        previous_model: BERTopic,
        current_docs: List[str],
        previous_docs: List[str],
        window_start: str
    ) -> Dict[str, Any]:
        """
        Run complete drift detection pipeline.
        
        Args:
            current_model: Current BERTopic model
            previous_model: Previous BERTopic model
            current_docs: Current documents
            previous_docs: Previous documents
            window_start: Window start timestamp
            
        Returns:
            Drift metrics dictionary
        """
        logger.info("Running full drift detection")
        
        # Calculate all drift metrics
        drift_metrics = {
            'prevalence_change': self.calculate_prevalence_change(current_model, previous_model),
            'centroid_shifts': self.calculate_centroid_shift(
                current_model, previous_model, current_docs, previous_docs
            ),
            'keyword_divergences': self.calculate_keyword_divergence(current_model, previous_model),
            'topic_changes': self.detect_new_and_disappeared_topics(current_model, previous_model),
            'window_start': window_start,
            'timestamp': datetime.now().isoformat()
        }

        logger.info("Drift detection complete")
        return drift_metrics


if __name__ == "__main__":
    # Example usage
    from src.modeling import BERTopicOnlineWrapper
    
    wrapper = BERTopicOnlineWrapper()
    detector = DriftDetector()
    
    # Load models (assuming they exist)
    current_model = wrapper.load_model("models/current/bertopic_model.pkl")
    previous_model = wrapper.load_model("models/previous/bertopic_model.pkl")
    
    # Run drift detection
    drift_metrics, alerts = detector.run_full_drift_detection(
        current_model=current_model,
        previous_model=previous_model,
        current_docs=[],  # Would need actual documents
        previous_docs=[],
        window_start="2017-11-01"
    )
    
    print(f"Drift metrics: {drift_metrics}")
    print(f"Alerts generated: {len(alerts)}")

