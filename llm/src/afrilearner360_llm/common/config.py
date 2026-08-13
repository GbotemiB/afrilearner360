"""Environment/config loading, shared across item_generation, profiling, clustering, and
recommendations modules.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root if present (safe no-op if the file doesn't exist).
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Default model for item/recommendation generation. Chosen for its 1M token context window (fits
# the whole locale knowledge base as grounding) and low cost -- see project notes. Swappable via
# env var without code changes, since we go through OpenRouter's OpenAI-compatible API.
DEFAULT_MODEL = os.environ.get("AFRILEARNER_LLM_MODEL", "minimax/minimax-m3")

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def require_api_key() -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill in your key."
        )
    return OPENROUTER_API_KEY
