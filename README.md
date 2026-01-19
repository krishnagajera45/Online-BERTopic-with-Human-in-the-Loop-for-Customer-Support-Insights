# TwCS Online Topic Modeling System

An end-to-end online topic modeling system for Twitter Customer Support (TwCS) conversations using BERTopic, with a FastAPI backend and Streamlit dashboard.

## Features

- 📊 **Online Topic Modeling**: Train and update BERTopic models incrementally
- 🔄 **Drift Detection**: Monitor topic evolution with prevalence, centroid shift, and keyword divergence metrics
- 🎯 **Real-time Inference**: Predict topics for new customer support messages
- ✏️ **Human-in-the-Loop**: Merge, split, and relabel topics with full audit trail
- 📈 **Interactive Dashboard**: Visualize topics, trends, and drift alerts
- 🤖 **MLflow Integration**: Track model versions and experiments
- ⚙️ **Automated Pipeline**: Scheduled batch processing with state management

## Architecture

```
┌─────────────────┐
│   Data Layer    │  TwCS CSV → Processed Parquet
└────────┬────────┘
         │
┌────────▼────────┐
│   ETL Pipeline  │  Clean → Filter → Transform
└────────┬────────┘
         │
┌────────▼────────┐
│  Modeling Layer │  Seed Model → Online Updates
└────────┬────────┘
         │
┌────────▼────────┐
│ Drift Detection │  Prevalence → Centroid → Keywords
└────────┬────────┘
         │
┌────────▼────────┐
│  FastAPI Backend│  REST API + Model Serving
└────────┬────────┘
         │
┌────────▼────────┐
│Streamlit Dashboard│  UI + Visualizations
└─────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.9+
- TwCS dataset (CSV format)

### One-Command Startup (NEW!)

The easiest way to start the **complete system** with all components:

```bash
# Single script to start everything (API + Dashboard + Prefect)
./run_full_system.sh

# Choose from:
# 1) API + Dashboard only (Simple)
# 2) API + Dashboard + Prefect (Full System) ⭐ Recommended
# 3) Prefect only (for scheduling)
# 4) Exit (manual startup)
```

### Manual Installation

1. **Clone the repository**
```bash
cd "/Users/krishnagajera/Project/Final Year Project/Master Project-BERTopic"
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Place TwCS dataset**
```bash
# Place your TwCS dataset in data/raw/twcs.csv
# Or create a sample:
cp your_twcs_data.csv data/raw/twcs.csv

# For testing, create a sample (first 50k rows):
head -n 50001 data/raw/twcs.csv > data/sample/twcs_sample.csv
```

### Running the System

#### 1. Train Initial Model (One-time Setup)

```bash
# Run initial training with sample data
python src/scheduler/run_window.py --init
```

This will:
- Process the first batch of data
- Train the seed BERTopic model
- Save model artifacts and metadata
- Initialize processing state

#### 2. Start FastAPI Backend

```bash
# In terminal 1
python -m src.api.main
```

API will be available at: `http://localhost:8000`
API docs at: `http://localhost:8000/docs`

#### 3. Start Streamlit Dashboard

```bash
# In terminal 2
streamlit run src/dashboard/app.py
```

Dashboard will open at: `http://localhost:8501`

#### 4. Run Scheduled Updates with Prefect

**Option A: Using Prefect (Recommended for Production)**

```bash
# Start Prefect server and agent
./scripts/start_prefect.sh

# Deploy flows (one-time)
./scripts/deploy_flows.sh

# View Prefect UI
open http://127.0.0.1:4200

# Run pipeline manually through Prefect
./scripts/run_pipeline_manual.sh

# Stop Prefect when done
./scripts/stop_prefect.sh
```

**Option B: Using Simple Scheduler (Quick Testing)**

```bash
# Run window pipeline manually
python src/scheduler/run_window.py --window

# Or set up cron job for daily updates:
crontab -e
# Add: 0 2 * * * cd /path/to/project && /path/to/venv/bin/python src/scheduler/run_window.py --window
```

## Project Structure

```
Master-Project-BERTopic/
├── src/
│   ├── etl.py                    # Data ingestion and preprocessing
│   ├── modeling.py               # BERTopic training and updates
│   ├── drift.py                  # Drift detection algorithms
│   ├── storage.py                # Data persistence layer
│   ├── api/                      # FastAPI backend
│   │   ├── main.py              # API application
│   │   ├── endpoints/           # API routes
│   │   └── models/              # Pydantic schemas
│   ├── dashboard/               # Streamlit frontend
│   │   ├── app.py              # Main dashboard
│   │   ├── pages/              # Multi-page app
│   │   └── utils/              # API client
│   ├── scheduler/              # Simple batch processing
│   │   └── run_window.py      # Basic scheduler (cron)
│   └── utils/                  # Shared utilities
├── etl/                        # Prefect orchestration
│   ├── tasks/                 # Prefect tasks
│   │   ├── data_tasks.py     # ETL tasks
│   │   ├── model_tasks.py    # Model training tasks
│   │   └── drift_tasks.py    # Drift detection tasks
│   ├── flows/                 # Prefect flows
│   │   ├── data_ingestion.py        # Data pipeline
│   │   ├── model_training.py        # Training pipeline
│   │   ├── drift_detection.py       # Drift pipeline
│   │   └── complete_pipeline.py     # Master flow
│   └── schedules/             # Deployment configs
│       ├── deploy.py         # Deploy flows
│       └── prefect_config.py # Prefect settings
├── scripts/                    # Helper scripts
│   ├── start_prefect.sh      # Start Prefect
│   ├── stop_prefect.sh       # Stop Prefect
│   ├── deploy_flows.sh       # Deploy flows
│   └── run_pipeline_manual.sh # Manual run
├── data/
│   ├── raw/                    # Raw CSV files
│   ├── processed/              # Processed Parquet
│   └── sample/                 # Sample data for testing
├── models/                     # Trained models
│   ├── current/               # Active model
│   └── previous/              # Previous version
├── outputs/                    # Results
│   ├── topics/                # Topic metadata (JSON)
│   ├── assignments/           # Doc-topic assignments (CSV)
│   ├── alerts/                # Drift alerts (CSV)
│   └── audit/                 # HITL audit log (CSV)
├── logs/                       # Application logs
├── config/                     # Configuration files
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## API Endpoints

### Topics
- `GET /api/v1/topics/current` - Get all current topics
- `GET /api/v1/topics/{topic_id}` - Get topic details
- `GET /api/v1/topics/{topic_id}/examples` - Get example documents

### Trends
- `GET /api/v1/trends` - Get topic trends over time

### Alerts
- `GET /api/v1/alerts` - Get drift alerts
- `GET /api/v1/alerts/latest` - Get latest alert

### Inference
- `POST /api/v1/infer` - Predict topic for text

### HITL (Human-in-the-Loop)
- `POST /api/v1/hitl/merge` - Merge topics
- `POST /api/v1/hitl/relabel` - Relabel topic
- `GET /api/v1/hitl/audit` - Get audit log

### Pipeline
- `GET /api/v1/pipeline/status` - Get pipeline status

## Dashboard Pages

1. **Home** - Overview and quick statistics
2. **Topics & Trends** - View all topics and their evolution
3. **Topic Drill-down** - Deep dive into specific topics
4. **Drift Alerts** - Monitor topic drift and changes
5. **HITL Editor** - Merge, relabel topics with audit trail
6. **Inference** - Predict topics for new text

## Configuration

Edit `config/config.yaml` to customize:
- Data paths
- Model hyperparameters
- API/Dashboard settings
- Batch processing parameters
- MLflow tracking

Edit `config/drift_thresholds.yaml` for drift detection sensitivity.

## Prefect Orchestration

The system includes two orchestration options:

### 1. Prefect (Recommended for Production)

**Features:**
- Web UI for monitoring at `http://127.0.0.1:4200`
- Task retry logic and error handling
- Schedule management (daily/weekly)
- Flow run history and logging
- Distributed task execution (with Dask)

**Setup:**
```bash
# Install Prefect (already in requirements.txt)
pip install prefect>=2.14.0

# Start Prefect server + agent
./scripts/start_prefect.sh

# Deploy flows with schedules
./scripts/deploy_flows.sh
```

**Scheduled Deployments:**
- `daily-pipeline`: Runs at 2 AM every day
- `weekly-pipeline`: Runs at 3 AM every Sunday

**Manual Execution:**
```bash
# Run through Prefect UI or CLI
prefect deployment run complete-pipeline-flow/daily-pipeline

# Or use helper script
./scripts/run_pipeline_manual.sh
```

**Monitoring:**
Visit `http://127.0.0.1:4200` to:
- View flow runs and task status
- Check logs and errors
- Monitor schedules
- Trigger manual runs

**Stop Prefect:**
```bash
./scripts/stop_prefect.sh
```

### 2. Simple Scheduler (For Testing/Development)

If you don't need Prefect's advanced features, use the simple scheduler:

```bash
# Manual run
python src/scheduler/run_window.py --window

# Or via cron
crontab -e
# Add: 0 2 * * * cd /path/to/project && /path/to/venv/bin/python src/scheduler/run_window.py --window
```

## Workflow

### Initial Training
1. Load historical TwCS data
2. Clean and filter customer tweets
3. Train seed BERTopic model
4. Extract and save topic metadata

### Online Updates
1. Scheduler fetches new batch of tweets
2. ETL pipeline processes data
3. Model updates with new batch
4. Drift detection compares models
5. Alerts generated if thresholds exceeded
6. Dashboard auto-updates

### HITL Refinement
1. User reviews topics in dashboard
2. Merge similar topics
3. Relabel with meaningful names
4. Changes logged in audit trail

## Development

### Run Tests
```bash
pytest tests/
```

### Code Structure
- **ETL**: Handles data ingestion, cleaning, filtering
- **Modeling**: BERTopic wrapper with online learning
- **Drift**: Multi-metric drift detection
- **Storage**: Centralized data persistence
- **API**: RESTful backend with FastAPI
- **Dashboard**: Interactive Streamlit UI

## Troubleshooting

### API Connection Error
- Ensure FastAPI is running: `python -m src.api.main`
- Check port 8000 is not in use

### No Topics Found
- Run initial setup: `python src/scheduler/run_window.py --init`
- Check `models/current/` for model file

### Import Errors
- Activate virtual environment: `source venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`

### Memory Issues
- Reduce batch size in `config/config.yaml`
- Use sample data for testing

## Technologies

- **BERTopic**: Topic modeling with transformers
- **FastAPI**: Modern Python web framework
- **Streamlit**: Interactive dashboards
- **MLflow**: Experiment tracking
- **Pandas**: Data processing
- **Plotly**: Interactive visualizations
- **scikit-learn**: ML utilities
- **UMAP**: Dimensionality reduction
- **HDBSCAN**: Density-based clustering

## License

MIT License

## Citation

If you use this system in your research, please cite:

```bibtex
@software{twcs_topic_modeling_2024,
  title={TwCS Online Topic Modeling System},
  author={Krishna Gajera},
  year={2024},
  url={https://github.com/yourusername/twcs-topic-modeling}
}
```

## Contact

For questions or issues, please open an issue on GitHub or contact Krishna Gajera.

---

**Note**: This is a prototype system. For production deployment, consider:
- Database instead of CSV/JSON storage
- Proper authentication and authorization
- Load balancing for API
- Containerization with Docker
- CI/CD pipeline
- Monitoring and alerting
