#!/bin/bash
# Start Prefect server and worker

echo "🚀 Starting Prefect infrastructure..."

# Start Prefect server in background
echo "📊 Starting Prefect server..."
prefect server start &
PREFECT_SERVER_PID=$!
echo "Prefect server started (PID: $PREFECT_SERVER_PID)"

# Wait for server to be ready
echo "⏳ Waiting for Prefect server to be ready..."
sleep 10

# Check server health
until curl -s http://127.0.0.1:4200/api/health > /dev/null 2>&1; do
    echo "Waiting for Prefect server..."
    sleep 2
done

echo "✅ Prefect server is ready!"

# Configure Prefect API URL for worker
export PREFECT_API_URL="http://127.0.0.1:4200/api"

# Ensure work pool exists (Prefect v3 requires it)
WORK_POOL_NAME="default-agent-pool"
if ! prefect work-pool inspect "$WORK_POOL_NAME" >/dev/null 2>&1; then
    echo "🛠️  Creating work pool: $WORK_POOL_NAME"
    prefect work-pool create "$WORK_POOL_NAME" -t process
fi

# Start Prefect worker in background
echo "🤖 Starting Prefect worker..."
prefect worker start -p "$WORK_POOL_NAME" -q default &
PREFECT_WORKER_PID=$!
echo "Prefect worker started (PID: $PREFECT_WORKER_PID)"

# Save PIDs for later cleanup
mkdir -p .prefect
echo $PREFECT_SERVER_PID > .prefect/server.pid
echo $PREFECT_WORKER_PID > .prefect/worker.pid

echo ""
echo "✅ Prefect infrastructure is running!"
echo "📊 Prefect UI: http://127.0.0.1:4200"
echo "🛑 To stop: ./scripts/stop_prefect.sh"
echo ""

