"""Prefect flow for data ingestion and ETL."""
from typing import Optional
from prefect import flow, get_run_logger
from datetime import datetime
from etl.tasks.data_tasks import preprocess_data_task, validate_data_task


@flow(name="data-ingestion-flow", log_prints=True)
def data_ingestion_flow(
    csv_path: str,
    output_parquet: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    filter_inbound: bool = True,
    min_docs: int = 10
):
    """
    Prefect flow for data ingestion and preprocessing.
    
    This flow:
    1. Reads data from CSV
    2. Preprocesses (clean, filter)
    3. Validates data quality
    4. Saves to Parquet
    
    Args:
        csv_path: Path to input CSV
        output_parquet: Path to output Parquet
        start_date: Start date for filtering
        end_date: End date for filtering
        filter_inbound: Whether to filter to inbound tweets
        min_docs: Minimum documents required
        
    Returns:
        Processed DataFrame
    """
    logger = get_run_logger()
    
    logger.info(f"Starting data ingestion flow")
    logger.info(f"Input: {csv_path}")
    logger.info(f"Output: {output_parquet}")
    logger.info(f"Date range: {start_date} to {end_date}")
    
    # Step 1: Preprocess data
    df = preprocess_data_task(
        csv_path=csv_path,
        output_parquet=output_parquet,
        start_date=start_date,
        end_date=end_date,
        filter_inbound=filter_inbound
    )
    
    # Step 2: Validate data
    validate_data_task(df, min_docs=min_docs)
    
    logger.info(f"Data ingestion flow completed: {len(df)} documents")
    return df


if __name__ == "__main__":
    # Test the flow
    df = data_ingestion_flow(
        csv_path="data/sample/twcs_sample.csv",
        output_parquet="data/processed/test.parquet",
        filter_inbound=True
    )
    print(f"Processed {len(df)} documents")

