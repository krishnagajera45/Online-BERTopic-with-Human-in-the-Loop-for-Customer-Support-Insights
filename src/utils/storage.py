"""
Storage Layer for TwCS Topic Modeling System.

Handles persistence of:
- Topic metadata (JSON)
- Document assignments (CSV)
- Drift alerts (CSV)
- HITL audit logs (CSV)
- Processing state (JSON)
"""
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from src.utils.logging_config import setup_logger
from src.utils.config import load_config

logger = setup_logger(__name__, "logs/storage.log")


class StorageManager:
    """Centralized storage manager for all data persistence."""
    
    def __init__(self, config: Any = None):
        """
        Initialize storage manager.
        
        Args:
            config: Configuration object (will load if not provided)
        """
        if config is None:
            config = load_config()
        
        self.config = config
        self.topics_path = Path(config.storage.topics_metadata_path)
        self.assignments_path = Path(config.storage.doc_assignments_path)
        self.alerts_path = Path(config.storage.alerts_path)
        self.audit_path = Path(config.storage.audit_log_path)
        self.state_path = Path(config.storage.state_file)
        
        # Ensure directories exist
        self._ensure_dirs()
    
    def _ensure_dirs(self):
        """Create storage directories if they don't exist."""
        for path in [self.topics_path, self.assignments_path, self.alerts_path, 
                     self.audit_path, self.state_path]:
            path.parent.mkdir(parents=True, exist_ok=True)
    
    # ========== Topic Metadata ==========
    
    def save_topics_metadata(self, topics: List[Dict[str, Any]], metadata: Dict[str, Any] = None):
        """
        Save topic metadata to JSON.
        
        Args:
            topics: List of topic dictionaries
            metadata: Additional metadata (optional)
        """
        try:
            data = {
                'topics': topics,
                'last_updated': datetime.now().isoformat(),
                'num_topics': len(topics)
            }
            
            if metadata:
                data['metadata'] = metadata
            
            with open(self.topics_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(topics)} topics to {self.topics_path}")
        
        except Exception as e:
            logger.error(f"Error saving topics metadata: {e}", exc_info=True)
            raise
    
    def load_topics_metadata(self) -> List[Dict[str, Any]]:
        """Load topic metadata from JSON."""
        if not self.topics_path.exists():
            logger.warning(f"Topics metadata file not found: {self.topics_path}")
            return []
        
        try:
            with open(self.topics_path, 'r') as f:
                data = json.load(f)
            
            topics = data.get('topics', [])
            logger.info(f"Loaded {len(topics)} topics from {self.topics_path}")
            return topics
        
        except Exception as e:
            logger.error(f"Error loading topics metadata: {e}", exc_info=True)
            return []
    
    def update_topic_label(self, topic_id: int, new_label: str):
        """Update custom label for a topic."""
        topics = self.load_topics_metadata()
        
        for topic in topics:
            if topic['topic_id'] == topic_id:
                topic['custom_label'] = new_label
                topic['last_modified'] = datetime.now().isoformat()
                break
        
        self.save_topics_metadata(topics)
        logger.info(f"Updated label for topic {topic_id} to '{new_label}'")
    
    # ========== Document Assignments ==========
    
    def append_doc_assignments(self, assignments: pd.DataFrame):
        """
        Append document-topic assignments to CSV.
        
        Args:
            assignments: DataFrame with doc_id, topic_id, timestamp, batch_id, confidence
        """
        try:
            file_exists = self.assignments_path.exists()
            
            assignments.to_csv(
                self.assignments_path,
                mode='a',
                header=not file_exists,
                index=False
            )
            
            logger.info(f"Appended {len(assignments)} document assignments")
        
        except Exception as e:
            logger.error(f"Error appending document assignments: {e}", exc_info=True)
            raise
    
    def load_doc_assignments(self, topic_id: Optional[int] = None) -> pd.DataFrame:
        """Load document assignments, optionally filtered by topic."""
        if not self.assignments_path.exists():
            logger.warning(f"Document assignments file not found: {self.assignments_path}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(self.assignments_path)
            
            if topic_id is not None:
                df = df[df['topic_id'] == topic_id]
            
            logger.info(f"Loaded {len(df)} document assignments")
            return df
        
        except Exception as e:
            logger.error(f"Error loading document assignments: {e}", exc_info=True)
            return pd.DataFrame()
    
    # ========== Drift Alerts ==========
    
    def append_drift_alerts(self, alerts: List[Dict[str, Any]]):
        """
        Append drift alerts to CSV.
        
        Args:
            alerts: List of alert dictionaries
        """
        if not alerts:
            return
        
        try:
            df = pd.DataFrame(alerts)
            file_exists = self.alerts_path.exists()
            
            df.to_csv(
                self.alerts_path,
                mode='a',
                header=not file_exists,
                index=False
            )
            
            logger.info(f"Appended {len(alerts)} drift alerts")
        
        except Exception as e:
            logger.error(f"Error appending drift alerts: {e}", exc_info=True)
            raise
    
    def load_drift_alerts(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Load drift alerts, optionally limited to most recent."""
        if not self.alerts_path.exists():
            logger.warning(f"Drift alerts file not found: {self.alerts_path}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(self.alerts_path)
            
            if limit:
                df = df.tail(limit)
            
            logger.info(f"Loaded {len(df)} drift alerts")
            return df
        
        except Exception as e:
            logger.error(f"Error loading drift alerts: {e}", exc_info=True)
            return pd.DataFrame()
    
    # ========== HITL Audit Log ==========
    
    def log_audit_action(self, action: Dict[str, Any]):
        """
        Log a HITL action to audit log.
        
        Args:
            action: Dictionary with action_type, old_topics, new_topics, timestamp, user_note
        """
        try:
            action['timestamp'] = datetime.now().isoformat()
            df = pd.DataFrame([action])
            file_exists = self.audit_path.exists()
            
            df.to_csv(
                self.audit_path,
                mode='a',
                header=not file_exists,
                index=False
            )
            
            logger.info(f"Logged audit action: {action['action_type']}")
        
        except Exception as e:
            logger.error(f"Error logging audit action: {e}", exc_info=True)
            raise
    
    def load_audit_log(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Load HITL audit log."""
        # Define expected columns for empty DataFrame
        expected_columns = ['timestamp', 'action_type', 'old_topics', 'new_topics', 'user_note', 'archived_model_timestamp']
        
        if not self.audit_path.exists():
            logger.info(f"Audit log file not found (no actions yet): {self.audit_path}")
            return pd.DataFrame(columns=expected_columns)
        
        try:
            df = pd.read_csv(self.audit_path)
            
            # Validate DataFrame has content
            if df.empty:
                logger.info("Audit log file is empty")
                return pd.DataFrame(columns=expected_columns)
            
            # Ensure all expected columns exist (fill missing ones with empty string)
            for col in expected_columns:
                if col not in df.columns:
                    df[col] = ''
            
            if limit and limit > 0:
                df = df.tail(limit)
            
            logger.info(f"Loaded {len(df)} audit log entries")
            return df
        
        except pd.errors.EmptyDataError:
            logger.warning(f"Audit log file is empty or corrupted: {self.audit_path}")
            return pd.DataFrame(columns=expected_columns)
        except Exception as e:
            logger.error(f"Error loading audit log: {e}", exc_info=True)
            return pd.DataFrame(columns=expected_columns)
    
    # ========== Processing State ==========
    
    def save_processing_state(self, state: Dict[str, Any]):
        """
        Save processing state (last processed timestamp, batch info, etc.).
        
        Args:
            state: Dictionary with processing state information
        """
        try:
            state['updated_at'] = datetime.now().isoformat()
            
            with open(self.state_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"Saved processing state to {self.state_path}")
        
        except Exception as e:
            logger.error(f"Error saving processing state: {e}", exc_info=True)
            raise
    
    def load_processing_state(self) -> Dict[str, Any]:
        """Load processing state."""
        if not self.state_path.exists():
            logger.warning(f"Processing state file not found: {self.state_path}")
            return {}
        
        try:
            with open(self.state_path, 'r') as f:
                state = json.load(f)
            
            logger.info(f"Loaded processing state from {self.state_path}")
            return state
        
        except Exception as e:
            logger.error(f"Error loading processing state: {e}", exc_info=True)
            return {}
    
    def get_last_processed_timestamp(self) -> Optional[str]:
        """Get the last processed timestamp from state."""
        state = self.load_processing_state()
        return state.get('last_processed_timestamp')
    
    def update_last_processed_timestamp(self, timestamp: str):
        """Update the last processed timestamp in state."""
        state = self.load_processing_state()
        state['last_processed_timestamp'] = timestamp
        self.save_processing_state(state)


if __name__ == "__main__":
    # Example usage
    storage = StorageManager()
    
    # Test saving topics
    topics = [
        {
            'topic_id': 0,
            'custom_label': 'Billing Issues',
            'top_words': ['charge', 'bill', 'payment'],
            'size': 100,
            'created_at': datetime.now().isoformat(),
            'batch_id': 'batch_001'
        }
    ]
    storage.save_topics_metadata(topics)
    
    # Test loading
    loaded_topics = storage.load_topics_metadata()
    print(f"Loaded topics: {loaded_topics}")

