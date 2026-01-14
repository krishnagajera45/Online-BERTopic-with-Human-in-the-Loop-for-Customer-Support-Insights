"""Configuration management for TwCS Topic Modeling system."""
import yaml
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Data-related configuration."""
    raw_csv_path: str
    processed_parquet_dir: str
    sample_csv_path: str
    timestamp_column: str = "created_at"
    text_column: str = "text"
    inbound_column: str = "inbound"


@dataclass
class ModelConfig:
    """Model hyperparameters configuration."""
    embedding_model: str = "all-MiniLM-L6-v2"
    min_cluster_size: int = 15
    min_samples: int = 5
    umap_n_components: int = 5
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.0
    umap_metric: str = "cosine"
    hdbscan_metric: str = "euclidean"
    min_df: int = 5
    max_df: float = 0.95
    ngram_range: list = field(default_factory=lambda: [1, 2])


@dataclass
class StorageConfig:
    """Storage paths configuration."""
    topics_metadata_path: str
    doc_assignments_path: str
    alerts_path: str
    audit_log_path: str
    current_model_path: str
    previous_model_path: str
    state_file: str


@dataclass
class MLflowConfig:
    """MLflow tracking configuration."""
    tracking_uri: str = "file:./mlruns"
    experiment_name: str = "twcs_topic_modeling"


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list = field(default_factory=lambda: ["http://localhost:8501"])


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    host: str = "0.0.0.0"
    port: int = 8501
    api_base_url: str = "http://localhost:8000"


@dataclass
class SchedulerConfig:
    """Scheduler configuration."""
    batch_size: int = 5000
    window_days: int = 1
    schedule_cron: str = "0 2 * * *"


@dataclass
class Config:
    """Main configuration object."""
    data: DataConfig
    model: ModelConfig
    storage: StorageConfig
    mlflow: MLflowConfig
    api: APIConfig
    dashboard: DashboardConfig
    scheduler: SchedulerConfig


def load_config(config_path: str = "config/config.yaml") -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config object
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return Config(
        data=DataConfig(**config_dict['data']),
        model=ModelConfig(**config_dict['model']),
        storage=StorageConfig(**config_dict['storage']),
        mlflow=MLflowConfig(**config_dict['mlflow']),
        api=APIConfig(**config_dict['api']),
        dashboard=DashboardConfig(**config_dict['dashboard']),
        scheduler=SchedulerConfig(**config_dict['scheduler'])
    )


def load_drift_thresholds(config_path: str = "config/drift_thresholds.yaml") -> Dict[str, Any]:
    """Load drift detection thresholds."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

