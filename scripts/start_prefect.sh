#!/bin/bash
# Start Prefect server and agent

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

# Start Prefect agent in background
echo "🤖 Starting Prefect agent..."
prefect agent start -q default &
PREFECT_AGENT_PID=$!
echo "Prefect agent started (PID: $PREFECT_AGENT_PID)"

# Save PIDs for later cleanup
mkdir -p .prefect
echo $PREFECT_SERVER_PID > .prefect/server.pid
echo $PREFECT_AGENT_PID > .prefect/agent.pid

echo ""
echo "✅ Prefect infrastructure is running!"
echo "📊 Prefect UI: http://127.0.0.1:4200"
echo "🛑 To stop: ./scripts/stop_prefect.sh"
echo ""

