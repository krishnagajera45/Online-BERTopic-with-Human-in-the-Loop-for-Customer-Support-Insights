"""Prefect tasks for data processing."""
from prefect import task
from pathlib import Path
import pandas as pd
from src.etl import preprocess_batch, load_twcs_data
from src.utils import setup_logger

logger = setup_logger(__name__, "logs/prefect_tasks.log")


@task(name="read_batch", retries=2, retry_delay_seconds=10)
def read_batch_task(
    csv_path: str,
    start_date: str = None,
    end_date: str = None,
    batch_size: int = None
) -> pd.DataFrame:
    """
    Read a batch of data from CSV.
    
    Args:
        csv_path: Path to CSV file
        start_date: Start date for filtering
        end_date: End date for filtering
        batch_size: Number of rows to read
        
    Returns:
        DataFrame with loaded data
    """
    logger.info(f"Reading batch from {csv_path}")
    logger.info(f"Date range: {start_date} to {end_date}")
    
    df = load_twcs_data(
        csv_path=csv_path,
        start_date=start_date,
        end_date=end_date,
        nrows=batch_size
    )
    
    logger.info(f"Loaded {len(df)} rows")
    return df


@task(name="preprocess_data", retries=2)
def preprocess_data_task(
    csv_path: str,
    output_parquet: str,
    start_date: str = None,
    end_date: str = None,
    filter_inbound: bool = True
) -> pd.DataFrame:
    """
    Preprocess data batch.
    
    Args:
        csv_path: Path to input CSV
        output_parquet: Path to output Parquet
        start_date: Start date
        end_date: End date
        filter_inbound: Whether to filter to inbound tweets
        
    Returns:
        Processed DataFrame
    """
    logger.info(f"Preprocessing data: {csv_path} -> {output_parquet}")
    
    df = preprocess_batch(
        csv_path=csv_path,
        output_parquet=output_parquet,
        start_date=start_date,
        end_date=end_date,
        filter_inbound=filter_inbound
    )
    
    logger.info(f"Preprocessed {len(df)} documents")
    return df


@task(name="validate_data")
def validate_data_task(df: pd.DataFrame, min_docs: int = 10) -> bool:
    """
    Validate that data meets minimum requirements.
    
    Args:
        df: DataFrame to validate
        min_docs: Minimum number of documents required
        
    Returns:
        True if valid, raises error otherwise
    """
    if len(df) < min_docs:
        raise ValueError(f"Insufficient data: {len(df)} < {min_docs} documents")
    
    required_cols = ['doc_id', 'text_cleaned', 'created_at']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    logger.info(f"Data validation passed: {len(df)} documents")
    return True

