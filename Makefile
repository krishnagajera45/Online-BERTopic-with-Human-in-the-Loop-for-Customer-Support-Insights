.PHONY: help install clean run-api run-dashboard run-all init-model test

help:
	@echo "TwCS Topic Modeling System - Makefile Commands"
	@echo "================================================"
	@echo "install          Install dependencies"
	@echo "clean            Clean up generated files"
	@echo "init-model       Initialize with sample data and train seed model"
	@echo "run-api          Start FastAPI backend"
	@echo "run-dashboard    Start Streamlit dashboard"
	@echo "run-all          Run API and dashboard (requires tmux)"
	@echo "run-pipeline     Run complete pipeline (Prefect flow)"
	@echo "test             Run tests"
	@echo "lint             Run linting"

install:
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache
	@echo "🧹 Cleaned up generated files"

init-model:
	python -m etl.flows.complete_pipeline
	@echo "✅ Initial model trained (pipeline auto-detected first run)"

run-api:
	python -m src.api.main

run-dashboard:
	streamlit run src/dashboard/app.py

run-all:
	@echo "Starting API and Dashboard..."
	@echo "API: http://localhost:8000"
	@echo "Dashboard: http://localhost:8501"
	@tmux new-session -d -s twcs-api 'python -m src.api.main'
	@tmux new-session -d -s twcs-dashboard 'streamlit run src/dashboard/app.py'
	@echo "✅ Services started in tmux sessions"
	@echo "To attach: tmux attach -t twcs-api or tmux attach -t twcs-dashboard"
	@echo "To stop: make stop"

stop:
	@tmux kill-session -t twcs-api || true
	@tmux kill-session -t twcs-dashboard || true
	@echo "⏹️  Services stopped"

run-pipeline:
	python -m etl.flows.complete_pipeline

test:
	pytest tests/ -v

lint:
	flake8 src/ --max-line-length=120
	black src/ --check

format:
	black src/
	isort src/

setup-sample:
	@echo "Creating sample data directory..."
	mkdir -p data/sample
	@echo "Please place your TwCS sample data at data/sample/twcs_sample.csv"

logs:
	tail -f logs/*.log

