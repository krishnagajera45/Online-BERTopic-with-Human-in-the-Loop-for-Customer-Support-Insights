"""Ollama client utilities for topic labeling."""
import json
from typing import List, Dict, Any, Optional

import httpx

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


def generate_topic_label(
    base_url: str,
    model: str,
    top_words: List[str],
    examples: Optional[List[str]] = None,
    prompt_template: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 128,
    timeout_seconds: int = 20,
    examples_limit: int = 3
) -> Dict[str, Optional[str]]:
    """
    Generate a concise topic label + summary using Ollama local API.
    
    Returns:
        Dict with keys: label, summary
    """
    examples = examples or []
    keywords_text = ", ".join(top_words)
    examples_text = "\n".join([f"- {ex}" for ex in examples[:examples_limit]])
    if not examples_text:
        examples_text = "- (no examples available)"

    if prompt_template:
        prompt = prompt_template.format(
            keywords=keywords_text,
            examples=examples_text
        )
    else:
        prompt = (
            "You are labeling support-ticket topics.\n"
            "Given keywords and example texts, return JSON with keys: label, summary.\n"
            "Label: 3-6 words. Summary: one sentence.\n\n"
            f"Keywords: {keywords_text}\n"
            f"Examples:\n{examples_text}\n"
        )

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }

    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.post(f"{base_url}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()

        # Ollama returns JSON in "response" when format=json
        raw = data.get("response", "").strip()
        parsed = json.loads(raw) if raw else {}

        label = parsed.get("label")
        summary = parsed.get("summary")

        return {"label": label, "summary": summary}
    except Exception as e:
        logger.warning(f"Ollama labeling failed: {e}")
        return {"label": None, "summary": None}
