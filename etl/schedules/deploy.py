"""Prefect deployment configuration for pipeline scheduling."""
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule
from etl.flows.complete_pipeline import complete_pipeline_flow


# Daily deployment - runs at 2 AM every day
daily_deployment = Deployment.build_from_flow(
    flow=complete_pipeline_flow,
    name="daily-pipeline",
    schedule=CronSchedule(cron="0 2 * * *", timezone="America/New_York"),
    parameters={
        "is_initial": False
    },
    work_queue_name="default",
    tags=["production", "daily", "topic-modeling"]
)


# Weekly deployment - runs at 3 AM every Sunday
weekly_deployment = Deployment.build_from_flow(
    flow=complete_pipeline_flow,
    name="weekly-pipeline",
    schedule=CronSchedule(cron="0 3 * * 0", timezone="America/New_York"),
    parameters={
        "is_initial": False
    },
    work_queue_name="default",
    tags=["production", "weekly", "topic-modeling"]
)


if __name__ == "__main__":
    # Apply the deployment
    daily_deployment.apply()
    print("✅ Daily deployment created")
    
    weekly_deployment.apply()
    print("✅ Weekly deployment created")

