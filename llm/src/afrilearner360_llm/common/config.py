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

# Default model for item/recommendation generation. A free-tier model, since the project has no
# API credits. Chosen because it is one of the few free models on OpenRouter that supports
# `structured_outputs` (server-side JSON schema enforcement, which generator.py depends on) and
# because Google's Gemma line has broader multilingual coverage than the other free options --
# relevant for Kinyarwanda item text. See DESIGN.md §7. Swappable via env var without code
# changes, since we go through OpenRouter's OpenAI-compatible API.
DEFAULT_MODEL = os.environ.get("AFRILEARNER_LLM_MODEL", "google/gemma-4-26b-a4b-it:free")

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def require_api_key() -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill in your key."
        )
    return OPENROUTER_API_KEY
