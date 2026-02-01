"""Prefect tasks package."""
# Import granular tasks from each module
from .data_tasks import (
    load_data_window_task,
    clean_text_column_task,
    add_document_ids_task,
    save_to_parquet_task,
    validate_data_task
)

from .model_tasks import (
    initialize_bertopic_model_task,
    fit_seed_model_task,
    load_bertopic_model_task,
    save_bertopic_model_task,
    transform_documents_task,
    update_topic_representations_task,
    archive_model_file_task,
    extract_topic_metadata_task,
    save_topic_metadata_task,
    train_seed_model_task,
    update_model_online_task,
    archive_model_task
)

from .drift_tasks import (
    load_bertopic_models_task,
    calculate_prevalence_change_task,
    calculate_centroid_shift_task,
    calculate_keyword_divergence_task,
    detect_topic_changes_task,
    generate_drift_alerts_task,
    save_drift_alerts_task
)

__all__ = [
    # Data tasks
    'load_data_window_task',
    'clean_text_column_task',
    'add_document_ids_task',
    'save_to_parquet_task',
    'validate_data_task',
    
    # Model tasks (granular)
    'initialize_bertopic_model_task',
    'fit_seed_model_task',
    'load_bertopic_model_task',
    'save_bertopic_model_task',
    'transform_documents_task',
    'update_topic_representations_task',
    'archive_model_file_task',
    'extract_topic_metadata_task',
    'save_topic_metadata_task',
    
    # Model tasks (orchestrators)
    'train_seed_model_task',
    'update_model_online_task',
    'archive_model_task',
    
    # Drift tasks
    'load_bertopic_models_task',
    'calculate_prevalence_change_task',
    'calculate_centroid_shift_task',
    'calculate_keyword_divergence_task',
    'detect_topic_changes_task',
    'generate_drift_alerts_task',
    'save_drift_alerts_task'
]
