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
pip install -q -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Create necessary directories
echo -e "${BLUE}Creating directories...${NC}"
mkdir -p data/raw data/processed data/sample data/state outputs/state
mkdir -p models/current models/previous
mkdir -p outputs/topics outputs/assignments outputs/alerts outputs/audit
mkdir -p logs mlruns
echo -e "${GREEN}✓ Directories created${NC}"

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
    
    if [ -f "data/sample/twcs_sample.csv" ]; then
        python src/scheduler/run_window.py --init
        echo -e "${GREEN}✓ Initial model trained${NC}"
    else
        echo -e "${YELLOW}Skipping model training - no data available${NC}"
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
        
        # Start API in background
        echo -e "${BLUE}Starting API...${NC}"
        python -m src.api.main &
        API_PID=$!
        
        # Wait for API
        sleep 3
        
        # Start Dashboard
        echo -e "${BLUE}Starting Dashboard...${NC}"
        streamlit run src/dashboard/app.py
        
        # Cleanup on exit
        kill $API_PID 2>/dev/null || true
        ;;
        
    2)
        echo -e "${GREEN}Starting FULL SYSTEM (API + Dashboard + Prefect)...${NC}"
        echo ""
        
        # Start Prefect
        echo -e "${BLUE}Step 1/3: Starting Prefect server...${NC}"
        prefect server start > logs/prefect_server.log 2>&1 &
        PREFECT_SERVER_PID=$!
        echo "Prefect server started (PID: $PREFECT_SERVER_PID)"
        
        # Wait for Prefect server
        echo "⏳ Waiting for Prefect server to be ready..."
        sleep 10
        
        until curl -s http://127.0.0.1:4200/api/health > /dev/null 2>&1; do
            echo "Still waiting for Prefect server..."
            sleep 2
        done
        echo -e "${GREEN}✓ Prefect server ready!${NC}"
        echo ""
        
        # Start Prefect agent
        echo -e "${BLUE}Step 2/3: Starting Prefect agent...${NC}"
        prefect agent start -q default > logs/prefect_agent.log 2>&1 &
        PREFECT_AGENT_PID=$!
        echo "Prefect agent started (PID: $PREFECT_AGENT_PID)"
        sleep 2
        echo -e "${GREEN}✓ Prefect agent ready!${NC}"
        echo ""
        
        # Save PIDs
        mkdir -p .prefect
        echo $PREFECT_SERVER_PID > .prefect/server.pid
        echo $PREFECT_AGENT_PID > .prefect/agent.pid
        
        # Start API
        echo -e "${BLUE}Step 3/3: Starting API & Dashboard...${NC}"
        python -m src.api.main > logs/api.log 2>&1 &
        API_PID=$!
        echo "API started (PID: $API_PID)"
        sleep 3
        echo -e "${GREEN}✓ API ready!${NC}"
        echo ""
        
        echo ""
        echo "=================================================="
        echo "🎉 FULL SYSTEM RUNNING!"
        echo "=================================================="
        echo ""
        echo "Access Points:"
        echo "  📊 Prefect UI:  http://127.0.0.1:4200"
        echo "  🔧 API:         http://localhost:8000"
        echo "  📝 API Docs:    http://localhost:8000/docs"
        echo "  📈 Dashboard:   http://localhost:8501"
        echo ""
        echo "Logs:"
        echo "  - Prefect Server: logs/prefect_server.log"
        echo "  - Prefect Agent:  logs/prefect_agent.log"
        echo "  - API:            logs/api.log"
        echo ""
        echo "To deploy flows: ./scripts/deploy_flows.sh"
        echo ""
        echo "Press Ctrl+C to stop all services"
        echo "=================================================="
        echo ""
        
        # Start Dashboard (foreground)
        streamlit run src/dashboard/app.py
        
        # Cleanup on exit
        echo ""
        echo "Stopping all services..."
        kill $API_PID 2>/dev/null || true
        kill $PREFECT_AGENT_PID 2>/dev/null || true
        kill $PREFECT_SERVER_PID 2>/dev/null || true
        rm -rf .prefect
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

