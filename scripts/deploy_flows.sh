#!/bin/bash
# Deploy Prefect flows and create schedules

echo "📦 Deploying Prefect flows..."

cd "$(dirname "$0")/.." || exit

# Activate virtual environment
source venv/bin/activate

# Prefect API URL for local worker and deploy
export PREFECT_API_URL="http://127.0.0.1:4200/api"

# Ensure work pool exists (Prefect v3 requires it)
WORK_POOL_NAME="default-agent-pool"
if ! prefect work-pool inspect "$WORK_POOL_NAME" >/dev/null 2>&1; then
  echo "🛠️  Creating work pool: $WORK_POOL_NAME"
  prefect work-pool create "$WORK_POOL_NAME" -t process
fi

# Deploy flows
python src/etl/schedules/deploy.py

echo ""
echo "✅ Flows deployed successfully!"
echo "📊 View deployments: http://127.0.0.1:4200/deployments"
echo ""

