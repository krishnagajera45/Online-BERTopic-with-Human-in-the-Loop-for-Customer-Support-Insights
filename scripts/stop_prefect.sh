#!/bin/bash
# Stop Prefect server and worker

echo "🛑 Stopping Prefect infrastructure..."

# Read PIDs
if [ -f .prefect/server.pid ]; then
    SERVER_PID=$(cat .prefect/server.pid)
    echo "Stopping Prefect server (PID: $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null || echo "Server already stopped"
fi

if [ -f .prefect/worker.pid ]; then
    WORKER_PID=$(cat .prefect/worker.pid)
    echo "Stopping Prefect worker (PID: $WORKER_PID)..."
    kill $WORKER_PID 2>/dev/null || echo "Worker already stopped"
fi

# Clean up PID files
rm -rf .prefect

echo "✅ Prefect infrastructure stopped"

