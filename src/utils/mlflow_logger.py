"""
Comprehensive MLflow logging utility for Prefect-MLflow integration.

This module provides utilities to:
- Link Prefect runs with MLflow runs
- Log batch statistics
- Log model metrics and parameters
- Log drift detection results
- Log alerts and anomalies
"""
import mlflow
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import json

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


class MLflowLogger:
    """Comprehensive MLflow logger with Prefect integration."""
    
    def __init__(self, tracking_uri: str, experiment_name: str):
        """
        Initialize MLflow logger.
        
        Args:
            tracking_uri: MLflow tracking URI
            experiment_name: MLflow experiment name
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
    
    def start_run_with_prefect_context(
        self,
        batch_id: str,
        prefect_flow_run_id: Optional[str] = None,
        prefect_flow_run_name: Optional[str] = None,
        prefect_flow_run_url: Optional[str] = None
    ) -> mlflow.ActiveRun:
        """
        Start an MLflow run with Prefect context.
        
        Args:
            batch_id: Batch identifier
            prefect_flow_run_id: Prefect flow run ID
            prefect_flow_run_name: Prefect flow run name
            prefect_flow_run_url: Prefect flow run URL for debugging
            
        Returns:
            MLflow active run
        """
        # Use Prefect flow run name as MLflow run name if available
        run_name = prefect_flow_run_name or f"batch_{batch_id}"
        
        run = mlflow.start_run(run_name=run_name)
        
        # Log Prefect context as tags and params
        if prefect_flow_run_id:
            mlflow.set_tag("prefect.flow_run_id", prefect_flow_run_id)
            mlflow.log_param("prefect_flow_run_id", prefect_flow_run_id)
        
        if prefect_flow_run_name:
            mlflow.set_tag("prefect.flow_run_name", prefect_flow_run_name)
        
        if prefect_flow_run_url:
            mlflow.set_tag("prefect.flow_run_url", prefect_flow_run_url)
            mlflow.log_param("prefect_flow_run_url", prefect_flow_run_url)
            logger.info(f"🔗 Prefect Run URL: {prefect_flow_run_url}")
        
        mlflow.set_tag("batch_id", batch_id)
        mlflow.set_tag("pipeline_stage", "complete_pipeline")
        
        logger.info(f"Started MLflow run: {run.info.run_name} (ID: {run.info.run_id})")
        
        return run
    
    def log_batch_statistics(
        self,
        documents: List[str],
        batch_id: str,
        window_start: str,
        window_end: str,
        df: Optional[pd.DataFrame] = None
    ):
        """
        Log comprehensive batch statistics.
        
        Args:
            documents: List of document texts
            batch_id: Batch identifier
            window_start: Window start date
            window_end: Window end date
            df: Optional DataFrame with additional metadata
        """
        logger.info("Logging batch statistics to MLflow")
        
        # Basic batch info
        mlflow.log_param("batch_id", batch_id)
        mlflow.log_param("window_start", window_start)
        mlflow.log_param("window_end", window_end)
        mlflow.log_param("timestamp", datetime.now().isoformat())
        
        # Document count
        num_docs = len(documents)
        mlflow.log_metric("batch_size", num_docs)
        mlflow.log_metric("num_documents", num_docs)
        
        # Text statistics
        if documents:
            doc_lengths = [len(doc) for doc in documents]
            word_counts = [len(doc.split()) for doc in documents]
            
            # Length statistics
            mlflow.log_metric("avg_doc_length_chars", np.mean(doc_lengths))
            mlflow.log_metric("median_doc_length_chars", np.median(doc_lengths))
            mlflow.log_metric("min_doc_length_chars", np.min(doc_lengths))
            mlflow.log_metric("max_doc_length_chars", np.max(doc_lengths))
            mlflow.log_metric("std_doc_length_chars", np.std(doc_lengths))
            
            # Word count statistics
            mlflow.log_metric("avg_word_count", np.mean(word_counts))
            mlflow.log_metric("median_word_count", np.median(word_counts))
            mlflow.log_metric("min_word_count", np.min(word_counts))
            mlflow.log_metric("max_word_count", np.max(word_counts))
            mlflow.log_metric("std_word_count", np.std(word_counts))
            
            # Total statistics
            mlflow.log_metric("total_chars", sum(doc_lengths))
            mlflow.log_metric("total_words", sum(word_counts))
        
        # DataFrame statistics (if available)
        if df is not None:
            mlflow.log_metric("df_rows", len(df))
            mlflow.log_metric("df_columns", len(df.columns))
            
            # Log column names
            mlflow.log_param("df_columns", ",".join(df.columns.tolist()))
            
            # Memory usage
            memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            mlflow.log_metric("df_memory_mb", memory_mb)
        
        logger.info(f"✓ Logged batch statistics: {num_docs} documents")
    
    def log_model_details(
        self,
        model,
        topics: np.ndarray,
        probs: np.ndarray,
        model_config: Dict[str, Any],
        is_initial: bool = False
    ):
        """
        Log comprehensive model details and metrics.
        
        Args:
            model: BERTopic model instance
            topics: Topic assignments
            probs: Topic probabilities
            model_config: Model configuration dictionary
            is_initial: Whether this is initial training
        """
        logger.info("Logging model details to MLflow")
        
        # Model type
        mlflow.set_tag("model_type", "BERTopic")
        mlflow.set_tag("is_initial_training", str(is_initial))
        mlflow.log_param("is_initial_training", is_initial)
        
        # Model configuration parameters
        mlflow.log_param("embedding_model", model_config.get("embedding_model", "unknown"))
        mlflow.log_param("min_cluster_size", model_config.get("min_cluster_size", 0))
        mlflow.log_param("min_samples", model_config.get("min_samples", 0))
        mlflow.log_param("n_neighbors", model_config.get("n_neighbors", 0))
        mlflow.log_param("n_components", model_config.get("n_components", 0))
        mlflow.log_param("top_n_words", model_config.get("top_n_words", 0))
        
        # Topic statistics
        unique_topics = set(topics)
        num_topics = len(unique_topics)
        num_outliers = sum(1 for t in topics if t == -1)
        
        mlflow.log_metric("num_topics", num_topics)
        mlflow.log_metric("num_unique_topics", len([t for t in unique_topics if t != -1]))
        mlflow.log_metric("num_outliers", num_outliers)
        mlflow.log_metric("outlier_ratio", num_outliers / len(topics) if len(topics) > 0 else 0)
        
        # Topic distribution statistics
        topic_counts = pd.Series(topics).value_counts()
        mlflow.log_metric("avg_topic_size", topic_counts.mean())
        mlflow.log_metric("median_topic_size", topic_counts.median())
        mlflow.log_metric("min_topic_size", topic_counts.min())
        mlflow.log_metric("max_topic_size", topic_counts.max())
        mlflow.log_metric("std_topic_size", topic_counts.std())
        
        # Probability statistics
        if len(probs) > 0:
            max_probs = [p.max() if len(p) > 0 else 0.0 for p in probs]
            mlflow.log_metric("avg_confidence", np.mean(max_probs))
            mlflow.log_metric("median_confidence", np.median(max_probs))
            mlflow.log_metric("min_confidence", np.min(max_probs))
            mlflow.log_metric("max_confidence", np.max(max_probs))
            mlflow.log_metric("std_confidence", np.std(max_probs))
            
            # Confidence distribution
            high_confidence = sum(1 for p in max_probs if p > 0.7)
            medium_confidence = sum(1 for p in max_probs if 0.3 < p <= 0.7)
            low_confidence = sum(1 for p in max_probs if p <= 0.3)
            
            mlflow.log_metric("high_confidence_count", high_confidence)
            mlflow.log_metric("medium_confidence_count", medium_confidence)
            mlflow.log_metric("low_confidence_count", low_confidence)
            mlflow.log_metric("high_confidence_ratio", high_confidence / len(max_probs))
        
        # Topic info from model
        try:
            topic_info = model.get_topic_info()
            
            # Log top topics
            top_5_topics = topic_info.head(6)  # 5 + outlier topic
            top_topics_dict = {}
            for idx, row in top_5_topics.iterrows():
                if row['Topic'] != -1:  # Skip outlier
                    topic_id = int(row['Topic'])
                    count = int(row['Count'])
                    top_topics_dict[f"topic_{topic_id}_count"] = count
                    mlflow.log_metric(f"topic_{topic_id}_size", count)
            
            # Save topic info as artifact
            topic_info_path = "topic_info.csv"
            topic_info.to_csv(topic_info_path, index=False)
            mlflow.log_artifact(topic_info_path)
            Path(topic_info_path).unlink()  # Clean up
            
        except Exception as e:
            logger.warning(f"Could not log topic info: {e}")
        
        logger.info(f"✓ Logged model details: {num_topics} topics, {num_outliers} outliers")
    
    def log_drift_metrics(
        self,
        drift_metrics: Dict[str, Any],
        window_start: str
    ):
        """
        Log drift detection metrics.
        
        Args:
            drift_metrics: Dictionary containing drift metrics
            window_start: Window start date
        """
        if not drift_metrics:
            logger.info("No drift metrics to log")
            return
        
        logger.info("Logging drift metrics to MLflow")
        
        mlflow.set_tag("drift_detection_enabled", "true")
        mlflow.log_param("drift_window_start", window_start)
        
        # Log drift scores
        if 'topic_drift' in drift_metrics:
            mlflow.log_metric("drift_topic_score", drift_metrics['topic_drift'])
        
        if 'vocabulary_drift' in drift_metrics:
            mlflow.log_metric("drift_vocabulary_score", drift_metrics['vocabulary_drift'])
        
        if 'distribution_drift' in drift_metrics:
            mlflow.log_metric("drift_distribution_score", drift_metrics['distribution_drift'])
        
        if 'overall_drift' in drift_metrics:
            mlflow.log_metric("drift_overall_score", drift_metrics['overall_drift'])
        
        # Log drift status
        if 'drift_detected' in drift_metrics:
            mlflow.set_tag("drift_detected", str(drift_metrics['drift_detected']))
            mlflow.log_metric("drift_detected", 1 if drift_metrics['drift_detected'] else 0)
        
        # Log thresholds
        if 'thresholds' in drift_metrics:
            for key, value in drift_metrics['thresholds'].items():
                mlflow.log_param(f"drift_threshold_{key}", value)
        
        # Save full drift metrics as artifact
        drift_metrics_path = "drift_metrics.json"
        with open(drift_metrics_path, 'w') as f:
            json.dump(drift_metrics, f, indent=2, default=str)
        mlflow.log_artifact(drift_metrics_path)
        Path(drift_metrics_path).unlink()  # Clean up
        
        logger.info("✓ Logged drift metrics")
    
    def log_alerts(
        self,
        alerts: List[Dict[str, Any]]
    ):
        """
        Log alerts and anomalies.
        
        Args:
            alerts: List of alert dictionaries
        """
        if not alerts:
            logger.info("No alerts to log")
            mlflow.log_metric("num_alerts", 0)
            mlflow.set_tag("alerts_generated", "false")
            return
        
        logger.info(f"Logging {len(alerts)} alerts to MLflow")
        
        mlflow.log_metric("num_alerts", len(alerts))
        mlflow.set_tag("alerts_generated", "true")
        
        # Count alerts by severity
        severity_counts = {}
        alert_types = {}
        
        for alert in alerts:
            severity = alert.get('severity', 'unknown')
            alert_type = alert.get('alert_type', 'unknown')
            
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
        
        # Log severity distribution
        for severity, count in severity_counts.items():
            mlflow.log_metric(f"alerts_{severity}", count)
        
        # Log alert types
        for alert_type, count in alert_types.items():
            mlflow.log_metric(f"alerts_type_{alert_type}", count)
        
        # Save alerts as artifact
        alerts_path = "alerts.json"
        with open(alerts_path, 'w') as f:
            json.dump(alerts, f, indent=2, default=str)
        mlflow.log_artifact(alerts_path)
        Path(alerts_path).unlink()  # Clean up
        
        logger.info(f"✓ Logged {len(alerts)} alerts")
    
    def log_model_artifact(
        self,
        model_path: str,
        artifact_name: str = "bertopic_model"
    ):
        """
        Log model artifact to MLflow.
        
        Args:
            model_path: Path to model file
            artifact_name: Name for the artifact
        """
        if Path(model_path).exists():
            mlflow.log_artifact(model_path, artifact_name)
            logger.info(f"✓ Logged model artifact: {model_path}")
        else:
            logger.warning(f"Model artifact not found: {model_path}")
    
    def log_processing_time(
        self,
        stage: str,
        duration_seconds: float
    ):
        """
        Log processing time for a pipeline stage.
        
        Args:
            stage: Pipeline stage name
            duration_seconds: Duration in seconds
        """
        mlflow.log_metric(f"time_{stage}_seconds", duration_seconds)
        mlflow.log_metric(f"time_{stage}_minutes", duration_seconds / 60)
        logger.info(f"✓ Logged {stage} processing time: {duration_seconds:.2f}s")
    
    def log_system_info(self):
        """Log system and environment information."""
        import platform
        import sys
        
        mlflow.log_param("python_version", sys.version.split()[0])
        mlflow.log_param("platform", platform.platform())
        mlflow.log_param("processor", platform.processor())
        
        logger.info("✓ Logged system info")
    
    def log_pipeline_summary(
        self,
        status: str,
        documents_processed: int,
        num_topics: int,
        drift_detected: bool,
        num_alerts: int,
        total_duration_seconds: float
    ):
        """
        Log overall pipeline summary.
        
        Args:
            status: Pipeline status (success/error)
            documents_processed: Number of documents processed
            num_topics: Number of topics discovered
            drift_detected: Whether drift was detected
            num_alerts: Number of alerts generated
            total_duration_seconds: Total pipeline duration
        """
        mlflow.set_tag("pipeline_status", status)
        mlflow.log_metric("pipeline_documents_processed", documents_processed)
        mlflow.log_metric("pipeline_num_topics", num_topics)
        mlflow.log_metric("pipeline_drift_detected", 1 if drift_detected else 0)
        mlflow.log_metric("pipeline_num_alerts", num_alerts)
        mlflow.log_metric("pipeline_total_duration_seconds", total_duration_seconds)
        mlflow.log_metric("pipeline_total_duration_minutes", total_duration_seconds / 60)
        
        logger.info(f"✓ Logged pipeline summary: {status}, {documents_processed} docs, {num_topics} topics")


def get_prefect_context() -> Dict[str, Optional[str]]:
    """
    Extract Prefect context from current flow run.
    
    Returns:
        Dictionary with Prefect flow run information
    """
    try:
        from prefect.context import get_run_context
        
        context = get_run_context()
        flow_run = context.flow_run
        
        # Construct Prefect UI URL (assuming default Prefect server)
        prefect_api_url = "http://127.0.0.1:4200"
        flow_run_url = f"{prefect_api_url}/flow-runs/flow-run/{flow_run.id}"
        
        return {
            "flow_run_id": str(flow_run.id),
            "flow_run_name": flow_run.name,
            "flow_run_url": flow_run_url,
            "flow_name": context.flow.name if hasattr(context, 'flow') else None
        }
    except Exception as e:
        logger.warning(f"Could not extract Prefect context: {e}")
        return {
            "flow_run_id": None,
            "flow_run_name": None,
            "flow_run_url": None,
            "flow_name": None
        }
