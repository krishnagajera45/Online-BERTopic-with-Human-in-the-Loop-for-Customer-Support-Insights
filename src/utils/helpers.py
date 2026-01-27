"""Helper utilities for the TwCS Topic Modeling system."""
import hashlib
import uuid
from datetime import datetime
from typing import List, Dict, Any
import html
import unicodedata


def generate_batch_id(start_date: str, end_date: str) -> str:
    """
    Generate a unique batch ID from date range.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        Batch ID string
    """
    start = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y%m%d')
    end = datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y%m%d')
    return f"batch_{start}_to_{end}"
import re


def generate_alert_id() -> str:
    """Generate a unique alert ID."""
    return f"alert_{uuid.uuid4().hex[:8]}"


def generate_batch_id(window_start: str, window_end: str = None) -> str:
    """
    Generate a batch ID from window dates.
    
    Args:
        window_start: Start date string
        window_end: End date string (optional)
        
    Returns:
        Batch ID string
    """
    if window_end:
        return f"batch_{window_start}_to_{window_end}"
    return f"batch_{window_start}"


def clean_text(text: str) -> str:
    """
    Clean tweet text by removing URLs, mentions, and extra whitespace.
    
    Args:
        text: Raw tweet text
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Decode HTML entities (e.g., &amp; -> &)
    text = html.unescape(text)
    
    # Normalize unicode (fix curly quotes, odd apostrophes, etc.)
    text = unicodedata.normalize("NFKC", text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove @mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags (keep the word)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove outbound agent signatures (e.g., ^MM, /AY) at end of message
    text = re.sub(r'(\s*[\^/][A-Za-z]{1,3})+$', '', text)
    
    # Mask phone-like numbers (privacy + noise reduction)
    text = re.sub(r'\b(?:\d[\s\-\.\(\)]*){7,}\d\b', '<PHONE>', text)
    
    # Normalize version-like strings (e.g., 8.4.22 -> <VERSION>)
    text = re.sub(r'\b\d+(?:\.\d+){1,}\b', '<VERSION>', text)
    
    
    # Remove emojis and pictographs (reduce topic noise)
    text = re.sub(
        r'[\U0001F300-\U0001F6FF\U0001F700-\U0001FAFF\U00002700-\U000027BF\U0001F1E6-\U0001F1FF]+',
        '',
        text
    )
    
    # Normalize repeated punctuation (e.g., !!! -> !, ??? -> ?)
    text = re.sub(r'([!?\.]){2,}', r'\1', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    # Convert to lowercase
    text = text.lower()
    
    return text


def parse_timestamp(timestamp_str: str) -> datetime:
    """Parse timestamp string to datetime object."""
    try:
        return datetime.fromisoformat(timestamp_str)
    except:
        try:
            return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        except:
            return datetime.strptime(timestamp_str, "%Y-%m-%d")


def format_timestamp(dt: datetime) -> str:
    """Format datetime to ISO string."""
    return dt.isoformat()


def normalize_weights(word_weight_list: List[tuple]) -> List[float]:
    """
    Normalize word weights to probability distribution.
    
    Args:
        word_weight_list: List of (word, weight) tuples
        
    Returns:
        List of normalized weights
    """
    weights = [w for _, w in word_weight_list]
    total = sum(weights)
    if total == 0:
        return [1.0 / len(weights)] * len(weights)
    return [w / total for w in weights]


def calculate_percentage_change(current: float, previous: float) -> float:
    """Calculate percentage change between two values."""
    if previous == 0:
        return 100.0 if current > 0 else 0.0
    return ((current - previous) / previous) * 100


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to maximum length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

