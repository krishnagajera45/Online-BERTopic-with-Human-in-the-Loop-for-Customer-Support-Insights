#!/bin/bash
# Run the complete pipeline manually

echo "▶️ Running complete pipeline manually..."

cd "$(dirname "$0")/.." || exit

# Activate virtual environment
source venv/bin/activate

# Run the pipeline
python -m src.etl.flows.complete_pipeline

echo ""
echo "✅ Pipeline completed!"

