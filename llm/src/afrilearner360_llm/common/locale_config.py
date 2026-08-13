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
    items_per_assessment: int
    generation_language: str
    delivery_language_by_grade_band: dict
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
        items_per_assessment=raw["assessment"]["items_per_assessment"],
        generation_language=raw["assessment"]["generation_language"],
        delivery_language_by_grade_band=raw["assessment"]["delivery_language_by_grade_band"],
        recommendation_context=raw["recommendation_context"],
        knowledge_base_text=knowledge_base_text,
    )


def delivery_language_for_grade_band(locale_config: LocaleConfig, grade_band: str) -> str:
    """Maps a grade band like 'P2' to the language its items must be DELIVERED in.

    This is not the language items are generated in -- the LLM always drafts in
    `locale_config.generation_language` (see DESIGN.md §7). Where the two differ, the item needs a
    human translation pass before going live.

    Rwanda-specific convention for now: P1-P3 = lower_primary, P4-P6 = upper_primary. Revisit if a
    future locale's grade structure differs.

    Raises ValueError for a grade band with no digits, rather than guessing -- guessing wrong here
    would silently mislabel whether an item needs translating.
    """
    digits = "".join(ch for ch in grade_band if ch.isdigit())
    if not digits:
        raise ValueError(
            f"cannot determine a grade number from grade_band {grade_band!r}; expected something like 'P3'"
        )

    band = "lower_primary" if int(digits) <= 3 else "upper_primary"
    languages = locale_config.delivery_language_by_grade_band
    if band not in languages:
        raise KeyError(
            f"locale {locale_config.locale!r} has no delivery language configured for {band!r}"
        )
    return languages[band]
