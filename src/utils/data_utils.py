"""Data utility functions for loading and processing data."""
import pandas as pd
from typing import Optional
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__, "logs/data_utils.log")


def load_twcs_data(
    csv_path: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
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
        
    Returns:
        DataFrame with TwCS data
    """
    logger.info(f"Loading pre-processed TwCS data from {csv_path}")
    
    try:
        # Read CSV - timestamps are already in datetime format from preprocessing
        df = pd.read_csv(csv_path, parse_dates=['created_at'])
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


def load_processed_data(parquet_path: str) -> pd.DataFrame:
    """
    Load processed data from Parquet file.
    
    Args:
        parquet_path: Path to processed Parquet file
        
    Returns:
        DataFrame with processed data
    """
    logger.info(f"Loading processed data from {parquet_path}")
    return pd.read_parquet(parquet_path)
