"""Tasks for calculating BERTopic evaluation metrics."""
from prefect import task, get_run_logger
from typing import List, Tuple, Dict, Any
import numpy as np
from pathlib import Path
import json

from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils import load_config


@task(name="calculate-bertopic-coherence", retries=1)
def calculate_bertopic_coherence_task(
    model,
    documents: List[str],
    topics: np.ndarray
) -> Dict[str, float]:
    """
    Calculate coherence metrics for BERTopic model.
    
    Args:
        model: BERTopic model
        documents: Document texts
        topics: Topic assignments
        
    Returns:
        Dictionary with coherence scores
    """
    logger = get_run_logger()
    logger.info("Calculating BERTopic coherence metrics")
    
    try:
        # Preprocess documents for coherence calculation
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        import nltk
        
        # Download required NLTK data if not present
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        stop_words = set(stopwords.words('english'))
        
        # Tokenize documents
        texts = []
        for doc in documents:
            tokens = [w.lower() for w in word_tokenize(doc) if w.isalnum() and w.lower() not in stop_words and len(w) > 2]
            if tokens:
                texts.append(tokens)
        
        # Create dictionary
        dictionary = Dictionary(texts)
        dictionary.filter_extremes(no_below=2, no_above=0.5)
        
        # Get topics from BERTopic model (exclude outlier -1)
        topic_words_list = []
        topic_info = model.get_topic_info()
        
        for _, row in topic_info.iterrows():
            topic_id = int(row['Topic'])
            if topic_id == -1:  # Skip outlier topic
                continue
                
            topic_words = model.get_topic(topic_id)
            if topic_words:
                # Get just the words (not probabilities) and filter valid words
                words = []
                for word, _ in topic_words[:10]:  # Top 10 words
                    # Only include words that exist in our tokenized texts
                    if word in dictionary.token2id:
                        words.append(word)
                
                if words:  # Only add if we have valid words
                    topic_words_list.append(words)
        
        if not topic_words_list or len(topic_words_list) < 2:
            logger.warning(f"Not enough valid topics for coherence calculation: {len(topic_words_list)}")
            return {'coherence_c_v': 0.0}
        
        # Calculate C_v coherence
        logger.info(f"Calculating coherence for {len(topic_words_list)} topics")
        try:
            coherence_model = CoherenceModel(
                topics=topic_words_list,
                texts=texts,
                dictionary=dictionary,
                coherence='c_v'
            )
            coherence_cv = coherence_model.get_coherence()
            logger.info(f"BERTopic Coherence (C_v): {coherence_cv:.4f}")
        except Exception as coh_err:
            logger.error(f"Coherence calculation error: {coh_err}")
            # Fallback to simpler u_mass coherence
            try:
                corpus_bow = [dictionary.doc2bow(text) for text in texts]
                coherence_model = CoherenceModel(
                    topics=topic_words_list,
                    corpus=corpus_bow,
                    dictionary=dictionary,
                    coherence='u_mass'
                )
                coherence_cv = coherence_model.get_coherence()
                # Normalize u_mass (typically -14 to 14) to 0-1 range
                coherence_cv = (coherence_cv + 14) / 28
                logger.info(f"BERTopic Coherence (u_mass normalized): {coherence_cv:.4f}")
            except Exception as fallback_err:
                logger.error(f"Fallback coherence failed: {fallback_err}")
                coherence_cv = 0.0
        
        return {
            'coherence_c_v': float(coherence_cv),
            'num_topics': len(topic_words_list)
        }
        
    except Exception as e:
        logger.error(f"Error calculating coherence: {e}", exc_info=True)
        return {'coherence_c_v': 0.0}


@task(name="calculate-bertopic-silhouette", retries=1)
def calculate_bertopic_silhouette_task(
    model_or_embeddings,
    documents_or_topics,
    topics=None,
    sample_size: int = 1000
) -> float:
    """
    Calculate silhouette score for BERTopic clustering.
    
    Args:
        model_or_embeddings: Either BERTopic model or precomputed embeddings
        documents_or_topics: Either documents (if model passed) or topics (if embeddings passed)
        topics: Topic assignments (if model passed)
        sample_size: Max number of documents to sample
        
    Returns:
        Silhouette score
    """
    logger = get_run_logger()
    logger.info("Calculating BERTopic silhouette score")
    
    try:
        # Determine if we got a model or embeddings
        if hasattr(model_or_embeddings, 'embedding_model'):
            # We got a BERTopic model - extract embeddings
            model = model_or_embeddings
            documents = documents_or_topics
            
            # Try different methods to get embeddings
            try:
                if hasattr(model.embedding_model, 'encode'):
                    embeddings = model.embedding_model.encode(documents)
                elif hasattr(model.embedding_model, 'embed'):
                    embeddings = model.embedding_model.embed(documents)
                else:
                    # Fallback: create new SentenceTransformer
                    from sentence_transformers import SentenceTransformer
                    encoder = SentenceTransformer('all-MiniLM-L6-v2')
                    embeddings = encoder.encode(documents)
            except Exception as e:
                logger.error(f"Could not extract embeddings: {e}")
                return 0.0
        else:
            # We got embeddings directly
            embeddings = model_or_embeddings
            topics = documents_or_topics
        
        # Convert to numpy array if needed
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        
        if not isinstance(topics, np.ndarray):
            topics = np.array(topics)
        
        # Filter out outlier topic -1
        mask = topics != -1
        filtered_embeddings = embeddings[mask]
        filtered_topics = topics[mask]
        
        # Check if we have enough topics
        unique_topics = np.unique(filtered_topics)
        if len(unique_topics) < 2:
            logger.warning(f"Not enough topics for silhouette: {len(unique_topics)}")
            return 0.0
        
        # Sample if too many documents
        if len(filtered_topics) > sample_size:
            indices = np.random.choice(len(filtered_topics), sample_size, replace=False)
            filtered_embeddings = filtered_embeddings[indices]
            filtered_topics = filtered_topics[indices]
        
        logger.info(f"Computing silhouette for {len(filtered_topics)} docs, {len(unique_topics)} topics")
        
        # Calculate silhouette score
        silhouette = silhouette_score(
            filtered_embeddings,
            filtered_topics,
            metric='cosine',
            sample_size=None
        )
        
        logger.info(f"BERTopic Silhouette Score: {silhouette:.4f}")
        return float(silhouette)
        
    except Exception as e:
        logger.error(f"Error calculating silhouette: {e}", exc_info=True)
        return 0.0


@task(name="save-bertopic-metrics", retries=1)
def save_bertopic_metrics_task(
    metrics: Dict[str, Any],
    output_path: str = "outputs/metrics/bertopic_metrics.json"
) -> str:
    """
    Save BERTopic metrics to file. Appends per-batch metrics for temporal analysis.
    
    Structure: {"batches": [{batch_id, coherence_c_v, diversity, ...}, ...], "latest": {...}}
    
    Args:
        metrics: Dictionary with BERTopic metrics (must include batch_id)
        output_path: Path to save metrics
        
    Returns:
        Path to saved file
    """
    logger = get_run_logger()
    logger.info(f"Saving BERTopic metrics to {output_path}")
    
    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing data
    data = {"batches": [], "latest": {}}
    if Path(output_path).exists():
        try:
            with open(output_path, 'r') as f:
                data = json.load(f)
            if "batches" not in data:
                data["batches"] = []
        except Exception:
            data = {"batches": [], "latest": {}}
    
    # Append this batch's metrics to history (avoid duplicates by batch_id)
    batch_id = metrics.get("batch_id")
    batch_record = {
        "batch_id": batch_id,
        "coherence_c_v": metrics.get("coherence_c_v", 0.0),
        "diversity": metrics.get("diversity", 0.0),
        "silhouette_score": metrics.get("silhouette_score", 0.0),
        "num_topics": metrics.get("num_topics", 0),
        "timestamp": metrics.get("timestamp", ""),
        "training_time_seconds": metrics.get("training_time_seconds", 0.0),
    }
    
    # Replace existing batch record if same batch_id, else append
    existing_ids = [b.get("batch_id") for b in data["batches"]]
    if batch_id in existing_ids:
        idx = existing_ids.index(batch_id)
        data["batches"][idx] = batch_record
    else:
        data["batches"].append(batch_record)
    
    # Keep latest for backward compatibility
    data["latest"] = {k: v for k, v in metrics.items() if k != "batches"}
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"BERTopic metrics saved ({len(data['batches'])} batches in history)")
    return output_path
