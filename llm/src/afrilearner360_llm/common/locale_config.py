"""Loads a locale's YAML config + cultural knowledge base text (see data/locales/*.yaml and
data/cultural_knowledge_base/*.md).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import PROJECT_ROOT


@dataclass
class LocaleConfig:
    locale: str
    display_name: str
    trait_definitions: dict
    point_budget: int
    language_by_grade_band: dict
    recommendation_context: dict
    knowledge_base_text: str


def load_locale(locale: str) -> LocaleConfig:
    config_path = PROJECT_ROOT / "data" / "locales" / f"{locale}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No locale config found at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    kb_path = PROJECT_ROOT / raw["assessment"]["knowledge_base_path"]
    knowledge_base_text = kb_path.read_text(encoding="utf-8")

    return LocaleConfig(
        locale=raw["locale"],
        display_name=raw["display_name"],
        trait_definitions=raw["trait_definitions"],
        point_budget=raw["assessment"]["point_budget"],
        language_by_grade_band=raw["assessment"]["language_by_grade_band"],
        recommendation_context=raw["recommendation_context"],
        knowledge_base_text=knowledge_base_text,
    )


def language_for_grade_band(locale_config: LocaleConfig, grade_band: str) -> str:
    """Maps a grade band like 'P2' to the configured language (e.g. lower_primary -> kinyarwanda).

    Rwanda-specific convention for now: P1-P3 = lower_primary, P4-P6 = upper_primary. Revisit if a
    future locale's grade structure differs.
    """
    try:
        grade_num = int("".join(ch for ch in grade_band if ch.isdigit()))
    except ValueError:
        grade_num = 0
    band = "lower_primary" if grade_num <= 3 else "upper_primary"
    return locale_config.language_by_grade_band.get(band, "english")
