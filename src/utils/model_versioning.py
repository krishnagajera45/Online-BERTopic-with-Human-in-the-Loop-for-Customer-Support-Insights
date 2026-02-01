"""
Model versioning and archiving utilities for BERTopic.

Handles:
- Versioned model storage (models/current/, models/previous/, models/archive/ts/)
- Model merging with min_similarity tuning
- Rollback capabilities
- Archiving old versions
"""
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__, "logs/model_versioning.log")


class ModelVersionManager:
    """Manages model versioning, archiving, and rollback."""
    
    def __init__(self, base_dir: str = "models"):
        """
        Initialize model version manager.
        
        Args:
            base_dir: Base directory for model storage
        """
        self.base_dir = Path(base_dir)
        self.current_dir = self.base_dir / "current"
        self.previous_dir = self.base_dir / "previous"
        self.archive_dir = self.base_dir / "archive"
        
        # Ensure directories exist
        for dir_path in [self.current_dir, self.previous_dir, self.archive_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_current_model_path(self) -> str:
        """Get path to current model."""
        return str(self.current_dir / "bertopic_model.pkl")
    
    def get_previous_model_path(self) -> str:
        """Get path to previous model."""
        return str(self.previous_dir / "bertopic_model.pkl")
    
    def get_model_metadata_path(self, model_path: str) -> str:
        """
        Get metadata file path for a model.
        
        Args:
            model_path: Model file path
            
        Returns:
            Metadata file path (same directory, .json suffix)
        """
        model_path_obj = Path(model_path)
        return str(model_path_obj.parent / f"{model_path_obj.stem}_metadata.json")
    
    def archive_current_as_previous(self) -> Tuple[str, str]:
        """
        Archive current model as previous (with timestamp).
        
        Returns:
            Tuple of (archived_path, timestamp)
        """
        current_path = Path(self.get_current_model_path())
        
        if not current_path.exists():
            logger.warning(f"No current model to archive at {current_path}")
            return str(current_path), datetime.now().isoformat()
        
        # Archive to models/archive/timestamp/
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archived_dir = self.archive_dir / timestamp
        archived_dir.mkdir(parents=True, exist_ok=True)
        
        archived_path = archived_dir / "bertopic_model.pkl"
        shutil.copy2(current_path, archived_path)
        
        # Also copy metadata if exists
        metadata_path = Path(self.get_model_metadata_path(str(current_path)))
        if metadata_path.exists():
            shutil.copy2(metadata_path, archived_dir / "model_metadata.json")
        
        # Copy current → previous
        previous_path = Path(self.get_previous_model_path())
        previous_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(current_path, previous_path)
        
        # Copy metadata to previous
        if metadata_path.exists():
            previous_metadata_path = Path(self.get_model_metadata_path(str(previous_path)))
            shutil.copy2(metadata_path, previous_metadata_path)
        
        logger.info(f"Archived model: {current_path} → {archived_path}")
        logger.info(f"Also copied to previous: {previous_path}")
        
        return str(archived_path), timestamp
    
    def save_model_metadata(self, model_path: str, metadata: Dict[str, Any]) -> str:
        """
        Save model metadata (merge info, version info, etc.).
        
        Args:
            model_path: Model file path
            metadata: Metadata dictionary
            
        Returns:
            Metadata file path
        """
        metadata_path = Path(self.get_model_metadata_path(model_path))
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp if not present
        if 'saved_at' not in metadata:
            metadata['saved_at'] = datetime.now().isoformat()
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved model metadata to {metadata_path}")
        return str(metadata_path)
    
    def load_model_metadata(self, model_path: str) -> Dict[str, Any]:
        """
        Load model metadata.
        
        Args:
            model_path: Model file path
            
        Returns:
            Metadata dictionary (empty dict if not found)
        """
        metadata_path = Path(self.get_model_metadata_path(model_path))
        
        if not metadata_path.exists():
            return {}
        
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading metadata from {metadata_path}: {e}")
            return {}
    
    def cleanup_old_versions(self, keep_count: int = 5) -> int:
        """
        Clean up old archived versions, keeping only the most recent.
        
        Args:
            keep_count: Number of versions to keep
            
        Returns:
            Number of versions deleted
        """
        if not self.archive_dir.exists():
            return 0
        
        # Get all timestamped directories sorted by name (reverse = newest first)
        versions = sorted(self.archive_dir.iterdir(), reverse=True)
        
        deleted_count = 0
        for version_dir in versions[keep_count:]:
            if version_dir.is_dir():
                shutil.rmtree(version_dir)
                deleted_count += 1
                logger.info(f"Deleted archived version: {version_dir}")
        
        return deleted_count
    
    def get_version_history(self) -> list:
        """
        Get list of archived versions with metadata.
        
        Returns:
            List of version info dictionaries (sorted by timestamp, newest first)
        """
        if not self.archive_dir.exists():
            return []
        
        versions = []
        for version_dir in sorted(self.archive_dir.iterdir(), reverse=True):
            if version_dir.is_dir():
                model_path = version_dir / "bertopic_model.pkl"
                if model_path.exists():
                    metadata_path = version_dir / "model_metadata.json"
                    metadata = {}
                    if metadata_path.exists():
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                        except Exception:
                            pass
                    
                    versions.append({
                        'timestamp': version_dir.name,
                        'path': str(model_path),
                        'size_mb': model_path.stat().st_size / (1024 * 1024),
                        'metadata': metadata
                    })
        
        return versions
