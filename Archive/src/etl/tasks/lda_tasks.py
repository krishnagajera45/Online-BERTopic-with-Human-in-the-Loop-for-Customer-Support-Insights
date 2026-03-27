"""Prefect tasks for LDA model training and evaluation."""
from prefect import task, get_run_logger
from typing import List, Tuple, Dict, Any
import numpy as np
from pathlib import Path
import time
import json

from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel
from gensim.utils import simple_preprocess
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK data is available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


@task(name="preprocess-for-lda", retries=1)
def preprocess_documents_for_lda_task(documents: List[str]) -> Tuple[List[List[str]], Dictionary, List]:
    """
    Preprocess documents for LDA (tokenization, stopwords, lemmatization).
    
    Args:
        documents: List of document texts
        
    Returns:
        Tuple of (tokenized_docs, dictionary, corpus)
    """
    logger = get_run_logger()
    logger.info(f"Preprocessing {len(documents)} documents for LDA")
    
    # Initialize preprocessing tools
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # Tokenize and preprocess
    processed_docs = []
    for doc in documents:
        # Simple tokenization (lowercase, alphanumeric only, min length 3)
        tokens = simple_preprocess(doc, deacc=True, min_len=3)
        
        # Remove stopwords and lemmatize
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        
        # Filter tokens that are too short after lemmatization
        tokens = [token for token in tokens if len(token) >= 3]
        
        if tokens:  # Only add non-empty documents
            processed_docs.append(tokens)
    
    logger.info(f"Preprocessed {len(processed_docs)} non-empty documents")
    
    # Create dictionary
    dictionary = Dictionary(processed_docs)
    
    # Filter extremes to reduce vocabulary size
    # Remove words that appear in less than 5 docs or more than 50% of docs
    dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=10000)
    
    logger.info(f"Dictionary size: {len(dictionary)} unique tokens")
    
    # Create bag-of-words corpus
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    
    return processed_docs, dictionary, corpus


@task(name="train-lda-model", retries=1)
def train_lda_model_task(
    corpus: List,
    dictionary: Dictionary,
    num_topics: int,
    passes: int = 10,
    iterations: int = 200,
    random_state: int = 42
) -> LdaModel:
    """
    Train LDA model on preprocessed corpus.
    
    Args:
        corpus: Bag-of-words corpus
        dictionary: Gensim dictionary
        num_topics: Number of topics to discover
        passes: Number of passes through corpus
        iterations: Number of iterations per pass
        random_state: Random seed for reproducibility
        
    Returns:
        Trained LDA model
    """
    logger = get_run_logger()
    logger.info(f"Training LDA model with {num_topics} topics")
    logger.info(f"Corpus size: {len(corpus)} documents")
    logger.info(f"Passes: {passes}, Iterations: {iterations}")
    
    start_time = time.time()
    
    # Train LDA model
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=random_state,
        update_every=1,
        chunksize=100,
        passes=passes,
        iterations=iterations,
        alpha='auto',  # Learn document-topic distribution
        eta='auto',    # Learn topic-word distribution
        per_word_topics=True
    )
    
    training_time = time.time() - start_time
    logger.info(f"LDA training completed in {training_time:.2f} seconds")
    
    return lda_model


@task(name="calculate-lda-coherence", retries=1)
def calculate_coherence_task(
    model: LdaModel,
    texts: List[List[str]],
    dictionary: Dictionary,
    coherence_type: str = 'c_v'
) -> float:
    """
    Calculate topic coherence for LDA model.
    
    Args:
        model: Trained LDA model
        texts: Preprocessed tokenized documents
        dictionary: Gensim dictionary
        coherence_type: Type of coherence ('c_v', 'u_mass', 'c_uci', 'c_npmi')
        
    Returns:
        Coherence score
    """
    logger = get_run_logger()
    logger.info(f"Calculating {coherence_type} coherence")
    
    coherence_model = CoherenceModel(
        model=model,
        texts=texts,
        dictionary=dictionary,
        coherence=coherence_type
    )
    
    coherence_score = coherence_model.get_coherence()
    logger.info(f"Coherence ({coherence_type}): {coherence_score:.4f}")
    
    return coherence_score


@task(name="calculate-lda-diversity", retries=1)
def calculate_diversity_task(model: LdaModel, top_n: int = 10) -> float:
    """
    Calculate topic diversity (proportion of unique words across topics).
    
    Args:
        model: Trained LDA model
        top_n: Number of top words per topic
        
    Returns:
        Diversity score (0-1)
    """
    logger = get_run_logger()
    logger.info(f"Calculating topic diversity (top {top_n} words)")
    
    # Get top words for each topic
    all_words = []
    for topic_id in range(model.num_topics):
        topic_words = model.show_topic(topic_id, topn=top_n)
        words = [word for word, _ in topic_words]
        all_words.extend(words)
    
    # Calculate diversity
    unique_words = len(set(all_words))
    total_words = len(all_words)
    diversity = unique_words / total_words if total_words > 0 else 0.0
    
    logger.info(f"Diversity: {diversity:.4f} ({unique_words}/{total_words} unique)")
    
    return diversity


@task(name="calculate-lda-silhouette", retries=1)
def calculate_silhouette_task(
    corpus: List,
    model: LdaModel,
    dictionary: Dictionary
) -> float:
    """
    Calculate silhouette score for LDA topic assignments.
    
    Args:
        corpus: Bag-of-words corpus
        model: Trained LDA model
        dictionary: Gensim dictionary
        
    Returns:
        Silhouette score (-1 to 1)
    """
    logger = get_run_logger()
    logger.info("Calculating silhouette score")
    
    try:
        # Get topic assignments for each document
        topic_assignments = []
        for doc in corpus:
            topic_dist = model.get_document_topics(doc)
            if topic_dist:
                # Assign to topic with highest probability
                main_topic = max(topic_dist, key=lambda x: x[1])[0]
                topic_assignments.append(main_topic)
            else:
                topic_assignments.append(-1)  # No topic assigned
        
        # Check if we have enough topics for silhouette calculation
        unique_topics = set(t for t in topic_assignments if t != -1)
        if len(unique_topics) < 2:
            logger.warning(f"Not enough topics for silhouette calculation: {len(unique_topics)}")
            return 0.0
        
        # Convert corpus to dense TF-IDF vectors for distance calculation
        # Use TF-IDF vectorizer on original BoW representation
        vectorizer = TfidfVectorizer(max_features=1000)
        
        # Reconstruct documents from BoW for vectorization
        reconstructed_docs = []
        for doc_bow in corpus:
            words = []
            for word_id, count in doc_bow:
                word = dictionary[word_id]
                words.extend([word] * int(count))
            reconstructed_docs.append(' '.join(words))
        
        # Filter out -1 labels (unassigned docs) — consistent with BERTopic silhouette
        valid_indices = [i for i, t in enumerate(topic_assignments) if t != -1]
        if len(valid_indices) < 2:
            logger.warning("Not enough assigned documents for silhouette after filtering -1")
            return 0.0
        filtered_docs = [reconstructed_docs[i] for i in valid_indices]
        filtered_labels = [topic_assignments[i] for i in valid_indices]

        # Create TF-IDF matrix
        tfidf_matrix = vectorizer.fit_transform(filtered_docs)
        
        # Calculate silhouette score
        silhouette = silhouette_score(
            tfidf_matrix,
            filtered_labels,
            metric='cosine',
            sample_size=min(1000, len(filtered_labels))  # Sample for efficiency
        )
        
        logger.info(f"Silhouette score: {silhouette:.4f}")
        return silhouette
        
    except Exception as e:
        logger.warning(f"Could not calculate silhouette score: {e}")
        return 0.0


@task(name="extract-lda-metadata", retries=1)
def extract_lda_metadata_task(
    model: LdaModel,
    corpus: List,
    top_n: int = 10
) -> Dict[str, Any]:
    """
    Extract metadata from LDA model.
    
    Args:
        model: Trained LDA model
        corpus: Bag-of-words corpus
        top_n: Number of top words per topic
        
    Returns:
        Dictionary with LDA metadata
    """
    logger = get_run_logger()
    logger.info("Extracting LDA model metadata")
    
    # Get topic information
    topics = []
    for topic_id in range(model.num_topics):
        topic_words = model.show_topic(topic_id, topn=top_n)
        words = [word for word, _ in topic_words]
        word_probs = [float(prob) for _, prob in topic_words]
        
        # Count documents assigned to this topic
        doc_count = 0
        for doc in corpus:
            topic_dist = model.get_document_topics(doc)
            if topic_dist:
                main_topic = max(topic_dist, key=lambda x: x[1])[0]
                if main_topic == topic_id:
                    doc_count += 1
        
        topics.append({
            'topic_id': topic_id,
            'top_words': words,
            'word_probabilities': word_probs,
            'document_count': doc_count
        })
    
    metadata = {
        'num_topics': model.num_topics,
        'num_documents': len(corpus),
        'topics': topics
    }
    
    logger.info(f"Extracted metadata for {model.num_topics} topics")
    return metadata


@task(name="save-lda-metrics", retries=1)
def save_lda_metrics_task(
    metrics: Dict[str, Any],
    output_path: str = "outputs/metrics/lda_metrics.json"
) -> str:
    """
    Save LDA metrics to file. Appends per-batch metrics for temporal analysis.
    
    Structure: {"batches": [{batch_id, coherence_c_v, diversity, ...}, ...], "latest": {...}}
    
    Args:
        metrics: Dictionary with LDA metrics (must include batch_id)
        output_path: Path to save metrics
        
    Returns:
        Path to saved file
    """
    logger = get_run_logger()
    logger.info(f"Saving LDA metrics to {output_path}")
    
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
    
    # Append this batch's metrics to history
    batch_id = metrics.get("batch_id")
    batch_record = {
        "batch_id": batch_id,
        "coherence_c_v": metrics.get("coherence_c_v", 0.0),
        "diversity": metrics.get("diversity", 0.0),
        "silhouette_score": metrics.get("silhouette_score", 0.0),
        "num_topics": metrics.get("num_topics", 0),
        "timestamp": metrics.get("timestamp", ""),
        "training_time_seconds": metrics.get("training_time_seconds") or metrics.get("total_time_seconds", 0.0),
    }
    
    existing_ids = [b.get("batch_id") for b in data["batches"]]
    if batch_id in existing_ids:
        idx = existing_ids.index(batch_id)
        data["batches"][idx] = batch_record
    else:
        data["batches"].append(batch_record)
    
    # Keep latest + full metrics for backward compatibility
    data["latest"] = {k: v for k, v in metrics.items() if k != "batches"}
    data["status"] = metrics.get("status", "success")
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"LDA metrics saved ({len(data['batches'])} batches in history)")
    return output_path
