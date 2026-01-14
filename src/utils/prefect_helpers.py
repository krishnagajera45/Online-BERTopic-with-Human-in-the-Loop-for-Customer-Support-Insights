"""Utility functions for Prefect flows."""
from datetime import datetime


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


def format_timestamp(dt: datetime = None) -> str:
    """
    Format datetime for logging.
    
    Args:
        dt: Datetime to format (defaults to now)
        
    Returns:
        Formatted timestamp string
    """
    if dt is None:
        dt = datetime.now()
    return dt.strftime('%Y-%m-%d %H:%M:%S')

