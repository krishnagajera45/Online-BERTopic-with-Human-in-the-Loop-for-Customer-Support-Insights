#!/bin/bash

# TwCS Topic Modeling System - Complete Startup Script
# This script sets up and runs the FULL system including Prefect

set -e

echo "=================================================="
echo "TwCS Topic Modeling System - FULL Startup"
echo "=================================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Generate unified debug log filename in debug folder
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p logs/debug
UNIFIED_DEBUG_LOG="logs/debug/run_${TIMESTAMP}.log"
# Export so Python logging uses the same file
export UNIFIED_DEBUG_LOG

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
# Use pip if available in venv, otherwise fall back to pip3
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
mkdir -p logs/debug mlruns
echo -e "${GREEN}✓ Directories created${NC}"

# Initialize unified debug log
echo -e "${CYAN}Initializing unified debug log...${NC}"
echo "================================================================================" > "$UNIFIED_DEBUG_LOG"
echo "UNIFIED DEBUG LOG - TwCS Topic Modeling System" >> "$UNIFIED_DEBUG_LOG"
echo "Started at: $(date)" >> "$UNIFIED_DEBUG_LOG"
echo "================================================================================" >> "$UNIFIED_DEBUG_LOG"
echo "" >> "$UNIFIED_DEBUG_LOG"
echo -e "${GREEN}✓ Unified debug log: ${CYAN}${UNIFIED_DEBUG_LOG}${NC}"

# Check if sample data exists
if [ ! -f "data/sample/twcs_sample.csv" ]; then
    echo -e "${YELLOW}Warning: No sample data found at data/sample/twcs_sample.csv${NC}"
    echo "Please place your TwCS sample data file there before running the model training."
    echo ""
    echo "You can create a sample from your full dataset:"
    echo "  head -n 50001 data/raw/twcs.csv > data/sample/twcs_sample.csv"
    echo ""
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if model exists
if [ ! -f "models/current/bertopic_model.pkl" ]; then
    echo -e "${YELLOW}No trained model found. Running initial training...${NC}"
    echo "This may take a few minutes depending on your data size."
    echo ""
    
    if [ -f "data/raw/twcs_cleaned.csv" ] || [ -f "data/sample/twcs_sample.csv" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] MODEL: Starting initial model training..." >> "$UNIFIED_DEBUG_LOG"
        # Pipeline auto-detects that no model exists and trains seed model
        python -m etl.flows.complete_pipeline 2>&1 | tee -a "$UNIFIED_DEBUG_LOG"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] MODEL: Initial model training complete" >> "$UNIFIED_DEBUG_LOG"
        echo -e "${GREEN}✓ Initial model trained${NC}"
    else
        echo -e "${YELLOW}Skipping model training - no data available${NC}"
        echo "Please place data at: data/raw/twcs_cleaned.csv or data/sample/twcs_sample.csv"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] MODEL: Skipped - no data available" >> "$UNIFIED_DEBUG_LOG"
    fi
fi

echo ""
echo "=================================================="
echo "Setup Complete! 🎉"
echo "=================================================="
echo ""

# Ask which components to start
echo "Which components would you like to start?"
echo ""
echo "1) API + Dashboard only (Simple)"
echo "2) API + Dashboard + Prefect (Full System)"
echo "3) Prefect only (for scheduling)"
echo "4) Exit (manual startup)"
echo ""
read -p "Enter your choice (1-4): " -n 1 -r
echo ""
echo ""

case $REPLY in
    1)
        echo -e "${GREEN}Starting API + Dashboard...${NC}"
        echo "Press Ctrl+C to stop all services"
        echo ""
        
        # Log to unified debug
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] STARTUP: API + Dashboard mode selected" >> "$UNIFIED_DEBUG_LOG"
        
        # Start API in background
        echo -e "${BLUE}Starting API...${NC}"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] STARTUP: Starting FastAPI server..." >> "$UNIFIED_DEBUG_LOG"
        python -m src.api.main 2>&1 | tee -a "$UNIFIED_DEBUG_LOG" &
        API_PID=$!
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] STARTUP: API started (PID: $API_PID)" >> "$UNIFIED_DEBUG_LOG"
        
        # Wait for API
        sleep 3
        
        # Start Dashboard
        echo -e "${BLUE}Starting Dashboard...${NC}"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] STARTUP: Starting Streamlit Dashboard..." >> "$UNIFIED_DEBUG_LOG"
        
        echo ""
        echo -e "${CYAN}📋 Unified Debug Log: ${UNIFIED_DEBUG_LOG}${NC}"
        echo -e "${CYAN}   View live: tail -f ${UNIFIED_DEBUG_LOG}${NC}"
        echo ""
        
        streamlit run src/dashboard/app.py 2>&1 | tee -a "$UNIFIED_DEBUG_LOG"
        
        # Cleanup on exit
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] SHUTDOWN: Stopping services..." >> "$UNIFIED_DEBUG_LOG"
        kill $API_PID 2>/dev/null || true
        ;;
        
    2)
        echo -e "${GREEN}Starting FULL SYSTEM (API + Dashboard + Prefect)...${NC}"
        echo ""
        
        # Log to unified debug
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] STARTUP: FULL SYSTEM mode selected" >> "$UNIFIED_DEBUG_LOG"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========== STARTING ALL SERVICES ==========" >> "$UNIFIED_DEBUG_LOG"
        
        # Start Prefect
        echo -e "${BLUE}Step 1/5: Starting Prefect server...${NC}"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] PREFECT: Starting Prefect server..." >> "$UNIFIED_DEBUG_LOG"
        prefect server start 2>&1 | tee -a logs/prefect_server.log >> "$UNIFIED_DEBUG_LOG" &
        PREFECT_SERVER_PID=$!
        echo "Prefect server started (PID: $PREFECT_SERVER_PID)"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] PREFECT: Server started (PID: $PREFECT_SERVER_PID)" >> "$UNIFIED_DEBUG_LOG"
        
        # Wait for Prefect server
        echo "⏳ Waiting for Prefect server to be ready..."
        sleep 10
        
        until curl -s http://127.0.0.1:4200/api/health > /dev/null 2>&1; do
            echo "Still waiting for Prefect server..."
            sleep 2
        done
        echo -e "${GREEN}✓ Prefect server ready!${NC}"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] PREFECT: Server ready!" >> "$UNIFIED_DEBUG_LOG"
        echo ""
        
        # Start Prefect worker
        echo -e "${BLUE}Step 2/5: Starting Prefect worker...${NC}"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] PREFECT: Starting Prefect worker..." >> "$UNIFIED_DEBUG_LOG"
        export PREFECT_API_URL="http://127.0.0.1:4200/api"
        WORK_POOL_NAME="default-agent-pool"
        if ! prefect work-pool inspect "$WORK_POOL_NAME" >/dev/null 2>&1; then
            echo "🛠️  Creating work pool: $WORK_POOL_NAME"
            prefect work-pool create "$WORK_POOL_NAME" -t process
        fi
        prefect worker start -p "$WORK_POOL_NAME" -q default 2>&1 | tee -a logs/prefect_worker.log >> "$UNIFIED_DEBUG_LOG" &
        PREFECT_WORKER_PID=$!
        echo "Prefect worker started (PID: $PREFECT_WORKER_PID)"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] PREFECT: Worker started (PID: $PREFECT_WORKER_PID)" >> "$UNIFIED_DEBUG_LOG"
        sleep 2
        echo -e "${GREEN}✓ Prefect agent ready!${NC}"
        echo ""

        # Start MLflow server for experiment tracking
        echo -e "${BLUE}Step 3/5: Starting MLflow server...${NC}"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] MLFLOW: Starting MLflow server..." >> "$UNIFIED_DEBUG_LOG"
        mlflow server \
            --backend-store-uri mlruns \
            --default-artifact-root mlruns \
            --host 127.0.0.1 \
            --port 5000 \
            2>&1 | tee -a logs/mlflow.log >> "$UNIFIED_DEBUG_LOG" &
        MLFLOW_PID=$!
        echo "MLflow server started (PID: $MLFLOW_PID)"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] MLFLOW: Server started (PID: $MLFLOW_PID)" >> "$UNIFIED_DEBUG_LOG"
        sleep 3
        echo -e "${GREEN}✓ MLflow server ready!${NC}"
        echo "   UI: http://127.0.0.1:5000"
        echo ""
        
        # Start Ollama server for local LLM labeling
        echo -e "${BLUE}Step 4/5: Starting Ollama server (gemma3:1b)...${NC}"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] OLLAMA: Starting Ollama server..." >> "$UNIFIED_DEBUG_LOG"
        if command -v ollama >/dev/null 2>&1; then
            if ! curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
                ollama serve >> logs/ollama.log 2>&1 &
                OLLAMA_PID=$!
                echo "Ollama server started (PID: $OLLAMA_PID)"
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] OLLAMA: Server started (PID: $OLLAMA_PID)" >> "$UNIFIED_DEBUG_LOG"

                # Wait briefly for Ollama to become ready (non-blocking)
                for _ in {1..10}; do
                    if curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
                        echo "Ollama server ready"
                        echo "[$(date '+%Y-%m-%d %H:%M:%S')] OLLAMA: Server ready" >> "$UNIFIED_DEBUG_LOG"
                        break
                    fi
                    sleep 1
                done
            else
                echo "Ollama server already running"
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] OLLAMA: Server already running" >> "$UNIFIED_DEBUG_LOG"
            fi
        else
            echo -e "${YELLOW}Ollama not found. Skipping local LLM labeling.${NC}"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] OLLAMA: Not installed, skipping" >> "$UNIFIED_DEBUG_LOG"
        fi
        echo ""

        # Save PIDs
        mkdir -p .prefect
        echo $PREFECT_SERVER_PID > .prefect/server.pid
        echo $PREFECT_WORKER_PID > .prefect/worker.pid
        
        # Start API
        echo -e "${BLUE}Step 5/5: Starting API & Dashboard...${NC}"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] API: Starting FastAPI server..." >> "$UNIFIED_DEBUG_LOG"
        python -m src.api.main 2>&1 | tee -a logs/api.log >> "$UNIFIED_DEBUG_LOG" &
        API_PID=$!
        echo "API started (PID: $API_PID)"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] API: Server started (PID: $API_PID)" >> "$UNIFIED_DEBUG_LOG"
        sleep 15
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
        echo "  📈 Dashboard:   http://localhost:8501"
        echo ""
        echo "Logs:"
        echo -e "  ${CYAN}📋 UNIFIED DEBUG: ${UNIFIED_DEBUG_LOG}${NC}"
        echo "  - Prefect Server: logs/prefect_server.log"
        echo "  - Prefect Worker: logs/prefect_worker.log"
        echo "  - MLflow Server:  logs/mlflow.log"
        echo "  - API:            logs/api.log"
        echo "  - Ollama:         logs/ollama.log"
        echo ""
        echo -e "${CYAN}View unified log live: tail -f ${UNIFIED_DEBUG_LOG}${NC}"
        echo ""
        echo "To deploy flows: ./scripts/deploy_flows.sh"
        echo ""
        echo "Press Ctrl+C to stop all services"
        echo "=================================================="
        echo ""
        
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] DASHBOARD: Starting Streamlit Dashboard..." >> "$UNIFIED_DEBUG_LOG"
        
        # Start Dashboard (foreground)

        #wait for 5 seconds
        streamlit run src/dashboard/app.py 2>&1 | tee -a "$UNIFIED_DEBUG_LOG"
        
        # Cleanup on exit
        echo ""
        echo "Stopping all services..."
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========== SHUTTING DOWN ALL SERVICES ==========" >> "$UNIFIED_DEBUG_LOG"
        kill $API_PID 2>/dev/null || true
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] SHUTDOWN: API stopped" >> "$UNIFIED_DEBUG_LOG"
        kill $PREFECT_WORKER_PID 2>/dev/null || true
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] SHUTDOWN: Prefect worker stopped" >> "$UNIFIED_DEBUG_LOG"
        kill $PREFECT_SERVER_PID 2>/dev/null || true
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] SHUTDOWN: Prefect server stopped" >> "$UNIFIED_DEBUG_LOG"
        kill $MLFLOW_PID 2>/dev/null || true
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] SHUTDOWN: MLflow stopped" >> "$UNIFIED_DEBUG_LOG"
        if [ -n "${OLLAMA_PID:-}" ]; then
            kill $OLLAMA_PID 2>/dev/null || true
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] SHUTDOWN: Ollama stopped" >> "$UNIFIED_DEBUG_LOG"
        fi
        rm -rf .prefect
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] SHUTDOWN: All services stopped successfully" >> "$UNIFIED_DEBUG_LOG"
        echo "All services stopped."
        ;;
        
    3)
        echo -e "${GREEN}Starting Prefect only...${NC}"
        ./scripts/start_prefect.sh
        
        echo ""
        echo "Prefect is running!"
        echo "To deploy flows: ./scripts/deploy_flows.sh"
        echo "To stop: ./scripts/stop_prefect.sh"
        echo ""
        echo "Note: You'll need to start API + Dashboard separately."
        ;;
        
    4)
        echo ""
        echo "=================================================="
        echo "Manual Startup Instructions"
        echo "=================================================="
        echo ""
        echo -e "${GREEN}Terminal 1 - Prefect (Optional):${NC}"
        echo "  ./scripts/start_prefect.sh"
        echo "  Then: ./scripts/deploy_flows.sh"
        echo ""
        echo -e "${GREEN}Terminal 2 - API:${NC}"
        echo "  source venv/bin/activate"
        echo "  python -m src.api.main"
        echo ""
        echo -e "${GREEN}Terminal 3 - Dashboard:${NC}"
        echo "  source venv/bin/activate"
        echo "  streamlit run src/dashboard/app.py"
        echo ""
        echo "Access Points:"
        echo "  - Prefect UI: http://127.0.0.1:4200"
        echo "  - API: http://localhost:8000"
        echo "  - Dashboard: http://localhost:8501"
        echo ""
        ;;
        
    *)
        echo -e "${RED}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac

