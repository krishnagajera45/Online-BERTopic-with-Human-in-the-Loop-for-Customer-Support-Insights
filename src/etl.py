"""
ETL Pipeline for TwCS Data Processing.

This module handles:
- Loading pre-processed TwCS data from CSV (already filtered to inbound, sorted, with synthetic timestamps)
- Cleaning text (remove URLs, mentions, extra whitespace)
- Filtering by date windows for batch processing
- Saving processed data as Parquet

NOTE: The raw data (twcs_cleaned.csv) has already been preprocessed in notebooks/data_preprocess.ipynb:
  - Filtered to inbound=True (customer messages only)
  - Empty/NaN text removed
  - Synthetic timestamps created (1-second intervals starting 2017-10-01)
  - Sorted chronologically by created_at
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List
from src.utils import clean_text, setup_logger, load_config

logger = setup_logger(__name__, "logs/etl.log")


def load_twcs_data(
    csv_path: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    nrows: Optional[int] = None
) -> pd.DataFrame:
    """
    Load pre-processed TwCS data from CSV with optional date filtering.
    
    NOTE: The CSV is expected to be already preprocessed with:
    - Inbound messages only (customer tweets)
    - Valid timestamps in datetime format
    - Chronologically sorted
    - No empty/NaN text values
    
    Args:
        csv_path: Path to preprocessed TwCS CSV file
        start_date: Start date for filtering (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
        end_date: End date for filtering (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
        nrows: Number of rows to read (for testing)
        
    Returns:
        DataFrame with TwCS data
    """
    logger.info(f"Loading pre-processed TwCS data from {csv_path}")
    
    try:
        # Read CSV - timestamps are already in datetime format from preprocessing
        df = pd.read_csv(csv_path, nrows=nrows, parse_dates=['created_at'])
        logger.info(f"Loaded {len(df)} rows from CSV")
        
        # Ensure timezone-aware datetimes (should already be UTC from preprocessing)
        if 'created_at' in df.columns and df['created_at'].dt.tz is None:
            df['created_at'] = df['created_at'].dt.tz_localize('UTC')
            logger.info("Applied UTC timezone to timestamps")
        
        # Filter by date range if specified (for batch processing)
        if start_date and 'created_at' in df.columns:
            start_dt = pd.to_datetime(start_date, utc=True)
            df = df[df['created_at'] >= start_dt]
            logger.info(f"Filtered from {start_date}: {len(df)} rows remaining")
        
        if end_date and 'created_at' in df.columns:
            end_dt = pd.to_datetime(end_date, utc=True)
            df = df[df['created_at'] <= end_dt]
            logger.info(f"Filtered until {end_date}: {len(df)} rows remaining")
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading TwCS data: {e}", exc_info=True)
        raise


def clean_tweet_text(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    """
    Clean tweet text column.
    
    Args:
        df: DataFrame with tweet data
        text_column: Name of text column
        
    Returns:
        DataFrame with cleaned text column
    """
    logger.info(f"Cleaning text in column '{text_column}'")
    
    # Create cleaned text column
    df['text_cleaned'] = df[text_column].fillna('').apply(clean_text)
    
    # Remove empty texts
    original_len = len(df)
    df = df[df['text_cleaned'].str.len() > 0]
    logger.info(f"Removed {original_len - len(df)} empty texts")
    
    # Drop ultra-short documents (token count threshold)
    min_tokens = 5
    token_counts = df['text_cleaned'].str.split().str.len()
    original_len = len(df)
    df = df[token_counts >= min_tokens]
    logger.info(f"Removed {original_len - len(df)} short texts (< {min_tokens} tokens)")
    
    # Deduplicate on cleaned text
    original_len = len(df)
    df = df.drop_duplicates(subset=['text_cleaned'])
    logger.info(f"Removed {original_len - len(df)} duplicate cleaned texts")
    
    return df


# NOTE: filter_customer_tweets() removed - already done in preprocessing notebook
# The twcs_cleaned.csv already contains only inbound=True (customer messages)


def add_doc_id(df: pd.DataFrame) -> pd.DataFrame:
    """Add unique document ID to each row."""
    if 'tweet_id' in df.columns:
        df['doc_id'] = 'tweet_' + df['tweet_id'].astype(str)
    else:
        df['doc_id'] = 'doc_' + pd.Series(range(len(df))).astype(str)
    
    return df


def preprocess_batch(
    csv_path: str,
    output_parquet: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    nrows: Optional[int] = None,
    filter_inbound: bool = True  # Kept for API compatibility but not used
) -> pd.DataFrame:
    """
    Simplified ETL pipeline for pre-processed data: load → clean → save Parquet.
    
    NOTE: The input CSV is expected to be already preprocessed:
    - Already filtered to inbound=True
    - Already sorted chronologically
    - Already has valid timestamps
    
    Args:
        csv_path: Path to preprocessed input CSV file
        output_parquet: Path to output Parquet file
        start_date: Start date for filtering (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
        end_date: End date for filtering (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
        nrows: Number of rows to read (for testing)
        filter_inbound: Deprecated - kept for backward compatibility
        
    Returns:
        Processed DataFrame
    """
    logger.info(f"Starting ETL pipeline: {csv_path} -> {output_parquet}")
    
    try:
        # Step 1: Load pre-processed data with date filtering
        df = load_twcs_data(csv_path, start_date, end_date, nrows)
        
        # Step 2: Clean text (remove URLs, mentions, extra whitespace, duplicates)
        df = clean_tweet_text(df, text_column='text')
        
        # Step 3: Add document IDs
        df = add_doc_id(df)
        
        # Step 4: Save to Parquet
        output_path = Path(output_parquet)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_parquet, index=False)
        logger.info(f"Saved {len(df)} processed rows to {output_parquet}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error in ETL pipeline: {e}", exc_info=True)
        raise


def load_processed_data(parquet_path: str) -> pd.DataFrame:
    """
    Load processed data from Parquet file.
    
    Args:
        parquet_path: Path to Parquet file
        
    Returns:
        DataFrame with processed data
    """
    logger.info(f"Loading processed data from {parquet_path}")
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df)} rows")
    return df


def get_date_range(df: pd.DataFrame, date_column: str = 'created_at') -> tuple:
    """
    Get date range from DataFrame.
    
    Args:
        df: DataFrame with date column
        date_column: Name of date column
        
    Returns:
        Tuple of (min_date, max_date)
    """
    if date_column not in df.columns:
        return None, None
    
    min_date = df[date_column].min()
    max_date = df[date_column].max()
    return min_date, max_date


if __name__ == "__main__":
    # Example usage
    config = load_config()
    
    # Process sample data
    df = preprocess_batch(
        csv_path=config.data.sample_csv_path,
        output_parquet="data/processed/twcs_sample.parquet",
        nrows=50000,
        filter_inbound=True
    )
    
    print(f"Processed {len(df)} documents")
    print(f"Date range: {get_date_range(df)}")
    print(f"Sample text: {df['text_cleaned'].iloc[0]}")

