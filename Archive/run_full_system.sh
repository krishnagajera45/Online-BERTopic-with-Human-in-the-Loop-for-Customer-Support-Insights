#!/bin/bash

# TwCS Topic Modeling System - Full System Startup Script
# Automatically starts: Prefect Server, Prefect Worker, MLflow, Ollama, API, and Dashboard

set -e

echo "=================================================="
echo "TwCS Topic Modeling System - Full System Startup"
echo "=================================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${BLUE}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate

# Install dependencies
echo -e "${BLUE}Installing dependencies...${NC}"
if command -v pip &> /dev/null; then
    pip install -q -r requirements.txt
else
    pip3 install -q -r requirements.txt
fi
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Create necessary directories
echo -e "${BLUE}Creating directories...${NC}"
mkdir -p data/raw data/processed data/sample data/state outputs/state
mkdir -p models/current models/previous
mkdir -p outputs/topics outputs/assignments outputs/alerts outputs/audit
mkdir -p logs mlruns
echo -e "${GREEN}✓ Directories created${NC}"

# Check if sample data exists
if [ ! -f "data/sample/twcs_sample.csv" ] && [ ! -f "data/raw/twcs_cleaned.csv" ]; then
    echo -e "${YELLOW}Warning: No data found!${NC}"
    echo "Please place your data at:"
    echo "  - data/raw/twcs_cleaned.csv (full dataset), OR"
    echo "  - data/sample/twcs_sample.csv (sample dataset)"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# # Check if model exists, train if not
# if [ ! -f "models/current/bertopic_model.pkl" ]; then
#     echo -e "${YELLOW}No trained model found. Running initial training...${NC}"
#     echo "This may take a few minutes depending on your data size."
#     echo ""
    
#     if [ -f "data/raw/twcs_cleaned.csv" ] || [ -f "data/sample/twcs_sample.csv" ]; then
#         # Pipeline auto-detects that no model exists and trains seed model
#         python -m src.etl.flows.complete_pipeline
#         echo -e "${GREEN}✓ Initial model trained${NC}"
#     else
#         echo -e "${YELLOW}Skipping model training - no data available${NC}"
#     fi
# fi

echo ""
echo "=================================================="
echo "Setup Complete! Starting All Services..."
echo "=================================================="
echo ""

# Start Prefect Server
echo -e "${BLUE}[1/5] Starting Prefect server...${NC}"
prefect server start > logs/prefect_server.log 2>&1 &
PREFECT_SERVER_PID=$!
echo "Prefect server started (PID: $PREFECT_SERVER_PID)"

# Wait for Prefect server to be ready
echo "⏳ Waiting for Prefect server..."
sleep 10

until curl -s http://127.0.0.1:4200/api/health > /dev/null 2>&1; do
    echo "Still waiting for Prefect server..."
    sleep 2
done
echo -e "${GREEN}✓ Prefect server ready!${NC}"
echo ""

# Start Prefect Worker
echo -e "${BLUE}[2/5] Starting Prefect worker...${NC}"
export PREFECT_API_URL="http://127.0.0.1:4200/api"
WORK_POOL_NAME="default-agent-pool"

# Create work pool if it doesn't exist
if ! prefect work-pool inspect "$WORK_POOL_NAME" >/dev/null 2>&1; then
    echo "🛠️  Creating work pool: $WORK_POOL_NAME"
    prefect work-pool create "$WORK_POOL_NAME" -t process
fi

prefect worker start -p "$WORK_POOL_NAME" -q default > logs/prefect_worker.log 2>&1 &
PREFECT_WORKER_PID=$!
echo "Prefect worker started (PID: $PREFECT_WORKER_PID)"
sleep 2
echo -e "${GREEN}✓ Prefect worker ready!${NC}"
echo ""

# Start MLflow Server
echo -e "${BLUE}[3/5] Starting MLflow server...${NC}"
mlflow server \
    --backend-store-uri mlruns \
    --default-artifact-root mlruns \
    --host 127.0.0.1 \
    --port 5000 \
    > logs/mlflow.log 2>&1 &
MLFLOW_PID=$!
echo "MLflow server started (PID: $MLFLOW_PID)"
sleep 3
echo -e "${GREEN}✓ MLflow server ready!${NC}"
echo "   UI: http://127.0.0.1:5000"
echo ""

# Start Ollama Server (if available)
echo -e "${BLUE}[4/5] Starting Ollama server...${NC}"
if command -v ollama >/dev/null 2>&1; then
    if ! curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
        ollama serve > logs/ollama.log 2>&1 &
        OLLAMA_PID=$!
        echo "Ollama server started (PID: $OLLAMA_PID)"
        
        # Wait for Ollama to be ready
        for _ in {1..10}; do
            if curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
                echo -e "${GREEN}✓ Ollama server ready!${NC}"
                break
            fi
            sleep 1
        done
    else
        echo -e "${GREEN}✓ Ollama already running!${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Ollama not installed (skipping)${NC}"
fi
echo ""

# Save PIDs for cleanup
mkdir -p .prefect
echo $PREFECT_SERVER_PID > .prefect/server.pid
echo $PREFECT_WORKER_PID > .prefect/worker.pid

# Start FastAPI Backend
echo -e "${BLUE}[5/5] Starting FastAPI & Dashboard...${NC}"
python -m src.api.main > logs/api.log 2>&1 &
API_PID=$!
echo "API started (PID: $API_PID)"
sleep 5
echo -e "${GREEN}✓ API ready!${NC}"
echo ""

echo ""
echo "=================================================="
echo "🎉 FULL SYSTEM RUNNING!"
echo "=================================================="
echo ""
echo "Access Points:"
echo "  📊 Prefect UI:  http://127.0.0.1:4200"
echo "  📒 MLflow UI:   http://127.0.0.1:5000"
echo "  🔧 API:         http://localhost:8000"
echo "  📝 API Docs:    http://localhost:8000/docs"
echo "  🤖 Ollama:      http://localhost:11434"
echo "  📈 Dashboard:   http://localhost:8501 (launching...)"
echo ""
echo "Logs Directory: logs/"
echo "  - Prefect Server: logs/prefect_server.log"
echo "  - Prefect Worker: logs/prefect_worker.log"
echo "  - MLflow:         logs/mlflow.log"
echo "  - API:            logs/api.log"
echo "  - Ollama:         logs/ollama.log"
echo ""
echo "Deploy flows: ./scripts/deploy_flows.sh"
echo ""
echo "Press Ctrl+C to stop all services"
echo "=================================================="
echo ""

# Start Streamlit Dashboard (foreground)
streamlit run src/dashboard/app.py

# Cleanup on exit (when Ctrl+C is pressed)
echo ""
echo "Stopping all services..."
kill $API_PID 2>/dev/null || true
kill $PREFECT_WORKER_PID 2>/dev/null || true
kill $PREFECT_SERVER_PID 2>/dev/null || true
kill $MLFLOW_PID 2>/dev/null || true
if [ -n "${OLLAMA_PID:-}" ]; then
    kill $OLLAMA_PID 2>/dev/null || true
fi
rm -rf .prefect

echo -e "${GREEN}✓ All services stopped${NC}"
echo "Goodbye! 👋"
