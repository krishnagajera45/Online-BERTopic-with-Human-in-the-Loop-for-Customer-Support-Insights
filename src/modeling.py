"""
BERTopic Model Training and Online Learning Module.

This module handles:
- Initial seed model training
- Online model updates (rolling window approach)
- Model persistence with MLflow
- Topic extraction and metadata generation
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import joblib
import mlflow
import mlflow.sklearn

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer

from src.utils import setup_logger, load_config, generate_batch_id
from src.storage import StorageManager

logger = setup_logger(__name__, "logs/modeling.log")


class BERTopicOnlineWrapper:
    """Wrapper for BERTopic with online learning capabilities."""
    
    def __init__(self, config: Any = None):
        """
        Initialize BERTopic model with configuration.
        
        Args:
            config: Configuration object (will load if not provided)
        """
        if config is None:
            config = load_config()
        
        self.config = config
        self.model = None
        self.storage = StorageManager(config)
        self._init_model_components()
    
    def _init_model_components(self):
        """Initialize BERTopic sub-models with configuration."""
        logger.info("Initializing BERTopic components")
        
        # Sentence transformer for embeddings
        self.sentence_model = SentenceTransformer(self.config.model.embedding_model)
        logger.info(f"Loaded embedding model: {self.config.model.embedding_model}")
        
        # UMAP for dimensionality reduction
        self.umap_model = UMAP(
            n_neighbors=self.config.model.umap_n_neighbors,
            n_components=self.config.model.umap_n_components,
            min_dist=self.config.model.umap_min_dist,
            metric=self.config.model.umap_metric,
            random_state=42
        )
        logger.info(f"Initialized UMAP with {self.config.model.umap_n_components} components")
        
        # HDBSCAN for clustering
        self.hdbscan_model = HDBSCAN(
            min_cluster_size=self.config.model.min_cluster_size,
            min_samples=self.config.model.min_samples,
            metric=self.config.model.hdbscan_metric,
            prediction_data=True
        )
        logger.info(f"Initialized HDBSCAN with min_cluster_size={self.config.model.min_cluster_size}")
        
        # CountVectorizer for bag-of-words representation
        self.vectorizer_model = CountVectorizer(
            stop_words='english',
            min_df=self.config.model.min_df,
            max_df=self.config.model.max_df,
            ngram_range=tuple(self.config.model.ngram_range)
        )
        logger.info(f"Initialized CountVectorizer with ngram_range={self.config.model.ngram_range}")
        
        # C-TF-IDF for topic representation c-TF-IDF = (word frequency in topic) × log(total topics / topics containing word)
        self.ctfidf_model = ClassTfidfTransformer()
        logger.info("Initialized C-TF-IDF transformer")
        
        # Create BERTopic model
        self.model = BERTopic(
            embedding_model=self.sentence_model,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            vectorizer_model=self.vectorizer_model,
            ctfidf_model=self.ctfidf_model,
            calculate_probabilities=True,
            verbose=True
        )
        logger.info("BERTopic model initialized")
    
    def train_seed_model(
        self,
        documents: List[str],
        batch_id: str,
        window_start: str,
        window_end: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train initial seed model on historical data.
        
        Args:
            documents: List of document texts
            batch_id: Batch identifier
            window_start: Window start date
            window_end: Window end date
            
        Returns:
            Tuple of (topics, probabilities)
        """
        logger.info(f"Training seed model on {len(documents)} documents")
        logger.info(f"Batch: {batch_id}, Window: {window_start} to {window_end}")
        
        try:
            # Train model
            topics, probs = self.model.fit_transform(documents)
            logger.info(f"Model training complete. Found {len(set(topics))} topics")
            
            # Save model with MLflow
            self._save_model_with_mlflow(batch_id, window_start, window_end, len(documents))
            
            # Save model locally
            self.save_model(self.config.storage.current_model_path)
            
            # Extract and save topic metadata
            self._save_topic_metadata(batch_id, window_start, window_end)
            
            return topics, probs
        
        except Exception as e:
            logger.error(f"Error training seed model: {e}", exc_info=True)
            raise
    
    def update_model_online(
        self,
        new_documents: List[str],
        batch_id: str,
        window_start: str,
        window_end: str,
        rolling_window_size: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update model with new batch of documents (rolling window approach).
        
        Args:
            new_documents: List of new document texts
            batch_id: Batch identifier
            window_start: Window start date
            window_end: Window end date
            rolling_window_size: Size of rolling window (None = use all historical data)
            
        Returns:
            Tuple of (topics, probabilities)
        """
        logger.info(f"Updating model with {len(new_documents)} new documents")
        logger.info(f"Batch: {batch_id}")
        
        try:
            # Archive current model as previous
            self._archive_current_model()
            

            # For simplicity, we'll use update_topics which updates topic representations
            # without retraining the entire model
            #Uses existing UMAP and HDBSCAN models,Embeds new documents with BERT,Projects into existing 5D space,,Assigns to existing or new clusters
            topics, probs = self.model.transform(new_documents)
            
            # Optionally update topic representations with new documents
            """Re-calculates bag-of-words for topics
                Re-ranks keywords with c-TF-IDF
                Updates topic labels"""
                
            self.model.update_topics(new_documents, vectorizer_model=self.vectorizer_model)
            
            logger.info(f"Model update complete")
            
            # Save updated model
            self._save_model_with_mlflow(batch_id, window_start, window_end, len(new_documents))
            # Saves to: models/current/bertopic_model.pkl
            self.save_model(self.config.storage.current_model_path)
            
            # Update topic metadata
            # Saves to: outputs/topics/topics_metadata.json
            self._save_topic_metadata(batch_id, window_start, window_end)
            
            return topics, probs
        
        except Exception as e:
            logger.error(f"Error updating model: {e}", exc_info=True)
            raise
    
    def save_model(self, model_path: str):
        """Save model to disk."""
        try:
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            self.model.save(model_path, serialization="pickle")
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}", exc_info=True)
            raise
    
    def load_model(self, model_path: str) -> BERTopic:
        """Load model from disk."""
        try:
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            self.model = BERTopic.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            return self.model
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            raise
    
    def _save_model_with_mlflow(
        self,
        batch_id: str,
        window_start: str,
        window_end: str,
        num_documents: int
    ):
        """Save model and log to MLflow."""
        try:
            mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)
            mlflow.set_experiment(self.config.mlflow.experiment_name)
            
            with mlflow.start_run(run_name=f"batch_{batch_id}"):
                # Log parameters
                mlflow.log_param("batch_id", batch_id)
                mlflow.log_param("window_start", window_start)
                mlflow.log_param("window_end", window_end)
                mlflow.log_param("num_documents", num_documents)
                mlflow.log_param("embedding_model", self.config.model.embedding_model)
                mlflow.log_param("min_cluster_size", self.config.model.min_cluster_size)
                
                # Log metrics
                topic_info = self.model.get_topic_info()
                num_topics = len(topic_info) - 1  # Exclude outlier topic (-1)
                mlflow.log_metric("num_topics", num_topics)
                mlflow.log_metric("avg_topic_size", topic_info['Count'].mean())
                
                # Log model artifact
                model_path = self.config.storage.current_model_path
                mlflow.log_artifact(model_path)
                
                logger.info(f"Logged model to MLflow: {num_topics} topics")
        
        except Exception as e:
            logger.error(f"Error logging to MLflow: {e}", exc_info=True)
    
    def _archive_current_model(self):
        """Archive current model as previous version."""
        current_path = Path(self.config.storage.current_model_path)
        previous_path = Path(self.config.storage.previous_model_path)
        
        if current_path.exists():
            previous_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy current to previous
            import shutil
            shutil.copy2(current_path, previous_path)
            logger.info(f"Archived current model to {previous_path}")
    
    def _save_topic_metadata(self, batch_id: str, window_start: str, window_end: str):
        """Extract and save topic metadata."""
        try:
            topic_info = self.model.get_topic_info()
            
            topics_metadata = []
            for _, row in topic_info.iterrows():
                topic_id = int(row['Topic'])
                
                # Skip outlier topic
                if topic_id == -1:
                    continue
                
                # Get top words for this topic
                topic_words = self.model.get_topic(topic_id)
                top_words = [word for word, _ in topic_words[:10]] if topic_words else []
                
                # Generate custom label from top 3 words
                custom_label = ", ".join([word for word, _ in topic_words[:3]]) if topic_words else f"Topic {topic_id}"
                
                topic_metadata = {
                    'topic_id': topic_id,
                    'custom_label': custom_label,
                    'top_words': top_words,
                    'size': int(row['Count']),
                    'created_at': datetime.now().isoformat(),
                    'batch_id': batch_id,
                    'window_start': window_start,
                    'window_end': window_end,
                    'count': int(row['Count'])
                }
                topics_metadata.append(topic_metadata)
            
            # Save to storage
            self.storage.save_topics_metadata(topics_metadata)
            logger.info(f"Saved metadata for {len(topics_metadata)} topics")
        
        except Exception as e:
            logger.error(f"Error saving topic metadata: {e}", exc_info=True)
            raise
    
    def extract_topics_metadata(self) -> List[Dict[str, Any]]:
        """Extract current topic metadata."""
        return self.storage.load_topics_metadata()
    
    def merge_topics(
        self,
        documents: List[str],
        topics_to_merge: List[int],
        new_label: Optional[str] = None
    ):
        """
        Merge multiple topics into one.
        
        Args:
            documents: Original documents
            topics_to_merge: List of topic IDs to merge
            new_label: Custom label for merged topic (optional)
        """
        try:
            logger.info(f"Merging topics: {topics_to_merge}")
            
            # Use BERTopic's merge_topics functionality
            self.model.merge_topics(documents, topics_to_merge)
            
            # Update metadata
            if new_label:
                self.storage.update_topic_label(topics_to_merge[0], new_label)
            
            # Save updated model
            self.save_model(self.config.storage.current_model_path)
            logger.info(f"Topics merged successfully")
        
        except Exception as e:
            logger.error(f"Error merging topics: {e}", exc_info=True)
            raise
    
    def get_topic_info(self) -> pd.DataFrame:
        """Get topic information DataFrame."""
        return self.model.get_topic_info()
    
    def transform(self, documents: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Transform new documents to get topic assignments."""
        return self.model.transform(documents)


if __name__ == "__main__":
    # Example usage
    from src.etl import load_processed_data
    
    config = load_config()
    wrapper = BERTopicOnlineWrapper(config)
    
    # Load sample data
    df = load_processed_data("data/processed/twcs_sample.parquet")
    documents = df['text_cleaned'].tolist()[:1000]  # First 1000 docs
    
    # Train seed model
    topics, probs = wrapper.train_seed_model(
        documents=documents,
        batch_id="batch_001",
        window_start="2017-10-01",
        window_end="2017-10-31"
    )
    
    print(f"Trained model with {len(set(topics))} topics")
    print(wrapper.get_topic_info())

