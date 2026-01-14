#!/bin/bash
# Run the complete pipeline manually

echo "▶️ Running complete pipeline manually..."

cd "$(dirname "$0")/.." || exit

# Activate virtual environment
source venv/bin/activate

# Run the pipeline
python -c "from etl.flows.complete_pipeline import complete_pipeline_flow; complete_pipeline_flow(is_initial=False)"

echo ""
echo "✅ Pipeline completed!"

