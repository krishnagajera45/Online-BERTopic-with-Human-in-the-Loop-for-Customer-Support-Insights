"""
ETL Pipeline for TwCS Data Processing.

This module handles:
- Loading TwCS data from CSV
- Parsing timestamps
- Cleaning text (remove URLs, mentions, extra whitespace)
- Filtering to inbound customer tweets
- Saving processed data as Parquet
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
    Load TwCS data from CSV with optional date filtering.
    
    Args:
        csv_path: Path to TwCS CSV file
        start_date: Start date for filtering (YYYY-MM-DD)
        end_date: End date for filtering (YYYY-MM-DD)
        nrows: Number of rows to read (for testing)
        
    Returns:
        DataFrame with TwCS data
    """
    logger.info(f"Loading TwCS data from {csv_path}")
    
    try:
        # Read CSV
        df = pd.read_csv(csv_path, nrows=nrows)
        logger.info(f"Loaded {len(df)} rows from CSV")
        
        # Parse timestamps
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(
                df['created_at'],
                errors='coerce',
                utc=True
            )
            logger.info("Parsed 'created_at' timestamps")
        
        # Filter by date range if specified
        if start_date and 'created_at' in df.columns:
            start_dt = pd.to_datetime(start_date, utc=True)
            df = df[df['created_at'] >= start_dt]
            logger.info(f"Filtered data from {start_date}: {len(df)} rows remaining")
        
        if end_date and 'created_at' in df.columns:
            end_dt = pd.to_datetime(end_date, utc=True)
            df = df[df['created_at'] <= end_dt]
            logger.info(f"Filtered data until {end_date}: {len(df)} rows remaining")
        
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
    
    return df


def filter_customer_tweets(df: pd.DataFrame, inbound_column: str = 'inbound') -> pd.DataFrame:
    """
    Filter to inbound customer tweets only.
    
    Args:
        df: DataFrame with tweet data
        inbound_column: Name of inbound indicator column
        
    Returns:
        DataFrame with only customer tweets
    """
    if inbound_column not in df.columns:
        logger.warning(f"Column '{inbound_column}' not found, skipping filter")
        return df
    
    original_len = len(df)
    df_filtered = df[df[inbound_column] == True].copy()
    logger.info(f"Filtered to inbound tweets: {len(df_filtered)} of {original_len}")
    
    return df_filtered


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
    filter_inbound: bool = True
) -> pd.DataFrame:
    """
    Full ETL pipeline: load → clean → filter → save Parquet.
    
    Args:
        csv_path: Path to input CSV file
        output_parquet: Path to output Parquet file
        start_date: Start date for filtering (YYYY-MM-DD)
        end_date: End date for filtering (YYYY-MM-DD)
        nrows: Number of rows to read (for testing)
        filter_inbound: Whether to filter to inbound tweets only
        
    Returns:
        Processed DataFrame
    """
    logger.info(f"Starting ETL pipeline: {csv_path} -> {output_parquet}")
    
    try:
        # Step 1: Load data
        df = load_twcs_data(csv_path, start_date, end_date, nrows)
        
        # Step 2: Clean text
        df = clean_tweet_text(df, text_column='text')
        
        # Step 3: Filter to customer tweets (optional)
        if filter_inbound:
            df = filter_customer_tweets(df, inbound_column='inbound')
        
        # Step 4: Add document IDs
        df = add_doc_id(df)
        
        # Step 5: Save to Parquet
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

