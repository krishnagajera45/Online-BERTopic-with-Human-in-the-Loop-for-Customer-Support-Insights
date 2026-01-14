#!/bin/bash
# Deploy Prefect flows and create schedules

echo "📦 Deploying Prefect flows..."

cd "$(dirname "$0")/.." || exit

# Activate virtual environment
source venv/bin/activate

# Deploy flows
python etl/schedules/deploy.py

echo ""
echo "✅ Flows deployed successfully!"
echo "📊 View deployments: http://127.0.0.1:4200/deployments"
echo ""
echo "Scheduled deployments:"
echo "  - daily-pipeline: Runs at 2 AM every day"
echo "  - weekly-pipeline: Runs at 3 AM every Sunday"
echo ""

