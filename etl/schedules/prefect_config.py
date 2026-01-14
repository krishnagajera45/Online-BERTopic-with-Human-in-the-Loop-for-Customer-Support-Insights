"""Prefect configuration file."""
from pathlib import Path

# Prefect Configuration
PREFECT_API_URL = "http://127.0.0.1:4200/api"
PREFECT_LOGGING_LEVEL = "INFO"

# Work Queue Configuration
DEFAULT_WORK_QUEUE = "default"
WORK_POOL_NAME = "default-agent-pool"

# Flow Run Configuration
FLOW_RUN_NAME_TEMPLATE = "pipeline-{date}"
TASK_RETRY_DELAY_SECONDS = 10
MAX_RETRIES = 2

# Storage Configuration
PREFECT_STORAGE_PATH = Path("data/prefect")
PREFECT_STORAGE_PATH.mkdir(parents=True, exist_ok=True)

