"""Prefect deployment configuration for pipeline scheduling."""
from pathlib import Path
import sys

from prefect.schedules import Cron

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from etl.flows.complete_pipeline import complete_pipeline_flow
from src.utils import load_config
from etl.schedules.prefect_config import WORK_POOL_NAME


config = load_config()


if __name__ == "__main__":
    # Apply the deployment
    deployment = complete_pipeline_flow.to_deployment(
        name="half-hour-pipeline",
        schedule=Cron(config.scheduler.schedule_cron, timezone="America/Los_Angeles"),
        parameters={"is_initial": False},
        work_pool_name=WORK_POOL_NAME,
        job_variables={
            "working_dir": str(PROJECT_ROOT),
            "env": {"PYTHONPATH": str(PROJECT_ROOT)}
        },
        tags=["production", "half-hour", "topic-modeling"]
    )
    deployment.apply()
    print("✅ 30-minute deployment created")

