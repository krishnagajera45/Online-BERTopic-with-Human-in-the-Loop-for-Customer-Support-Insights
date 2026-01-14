#!/bin/bash
# Stop Prefect server and agent

echo "🛑 Stopping Prefect infrastructure..."

# Read PIDs
if [ -f .prefect/server.pid ]; then
    SERVER_PID=$(cat .prefect/server.pid)
    echo "Stopping Prefect server (PID: $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null || echo "Server already stopped"
fi

if [ -f .prefect/agent.pid ]; then
    AGENT_PID=$(cat .prefect/agent.pid)
    echo "Stopping Prefect agent (PID: $AGENT_PID)..."
    kill $AGENT_PID 2>/dev/null || echo "Agent already stopped"
fi

# Clean up PID files
rm -rf .prefect

echo "✅ Prefect infrastructure stopped"

