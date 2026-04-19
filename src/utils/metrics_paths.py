"""Dataset-scoped paths for metrics JSON files (aligned with ``config.storage.metrics_dir``)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.utils.config import load_config

LEGACY_METRICS_DIR = Path("outputs/metrics")


def metrics_dir() -> Path:
    return Path(load_config().storage.metrics_dir)


def bertopic_metrics_path() -> Path:
    return metrics_dir() / "bertopic_metrics.json"


def lda_metrics_path() -> Path:
    return metrics_dir() / "lda_metrics.json"


def nmf_metrics_path() -> Path:
    return metrics_dir() / "nmf_metrics.json"


def _candidates(primary: Path, legacy_name: str) -> List[Path]:
    """Primary dataset path first, then repo-root legacy ``outputs/metrics/``."""
    return [primary, LEGACY_METRICS_DIR / legacy_name]


def read_json_first_existing(primary: Path, legacy_name: str) -> Dict[str, Any]:
    """
    Load JSON from the first path that exists (dataset-scoped, then legacy).

    Returns empty dict if neither exists.
    """
    for p in _candidates(primary, legacy_name):
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
    return {}


def load_metrics_state_for_save(output_path: str, legacy_filename: str) -> Dict[str, Any]:
    """
    Load existing metrics JSON for append/save.

    If the dataset-scoped file does not exist yet but a legacy file under
    ``outputs/metrics/`` exists, seed from legacy so history is preserved
    after migrating to per-dataset paths.
    """
    out = Path(output_path)
    default: Dict[str, Any] = {"batches": [], "latest": {}}

    if out.exists():
        try:
            with open(out, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "batches" not in data:
                data["batches"] = []
            return data
        except Exception:
            return dict(default)

    legacy = LEGACY_METRICS_DIR / legacy_filename
    if legacy.exists():
        try:
            with open(legacy, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "batches" not in data:
                data["batches"] = []
            return data
        except Exception:
            pass
    return dict(default)
