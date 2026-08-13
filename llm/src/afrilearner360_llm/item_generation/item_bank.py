"""Reading and writing the on-disk item bank.

Layout (see DESIGN.md §5):

    data/item_banks/<locale>/
      drafts/    P3-fractions-20260813T104500.json   raw LLM output, not usable with students
      approved/  P3.json                             the fixed assessment, reviewed + translated

**The drafts/approved split is the review gate made structural.** Generation only ever writes to
`drafts/`. Moving an item into `approved/` is a deliberate human act -- reviewing it for
relatability and stereotyping, and supplying the translation where `needs_translation` is set.
Nothing in this codebase promotes a draft automatically, and `load_approved` refuses to serve an
item that still claims it needs translating.

Plain JSON files rather than a database: the review gate is a human editing this content, JSON
diffs readably in a pull request, and the operational database belongs to the team's Flask app
(explicitly out of scope for this repo). That app reads `approved/`; this repo produces it.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from ..common.config import PROJECT_ROOT
from ..common.locale_config import LocaleConfig
from .schema import AssessmentItem, ItemGenerationResponse

DRAFTS_DIRNAME = "drafts"
APPROVED_DIRNAME = "approved"


class ItemBankError(Exception):
    """Raised when an item bank file is missing, malformed, or not fit to serve to students."""


def item_bank_root(locale: str) -> Path:
    return PROJECT_ROOT / "data" / "item_banks" / locale


def drafts_dir(locale: str) -> Path:
    return item_bank_root(locale) / DRAFTS_DIRNAME


def approved_dir(locale: str) -> Path:
    return item_bank_root(locale) / APPROVED_DIRNAME


def approved_path(locale: str, grade_band: str) -> Path:
    return approved_dir(locale) / f"{grade_band}.json"


def _slug(text: str) -> str:
    """Filesystem-safe fragment for a filename, e.g. 'the water cycle' -> 'the-water-cycle'."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "untitled"


def save_drafts(
    response: ItemGenerationResponse,
    *,
    locale: str,
    grade_band: str,
    topic: str,
    generated_at: Optional[datetime] = None,
) -> Path:
    """Write generated items to the drafts directory and return the file path.

    Filenames are timestamped rather than overwritten, so repeated generation runs on the same
    topic accumulate for comparison instead of silently destroying the previous batch -- which
    matters when you are still evaluating model output quality.
    """
    generated_at = generated_at or datetime.now()
    target_dir = drafts_dir(locale)
    target_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{grade_band}-{_slug(topic)}-{generated_at:%Y%m%dT%H%M%S}.json"
    path = target_dir / filename
    path.write_text(
        json.dumps(response.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def load_drafts(path: Path) -> ItemGenerationResponse:
    """Load a single drafts file. Raises ItemBankError if it is missing or malformed."""
    if not path.exists():
        raise ItemBankError(f"no draft file at {path}")
    try:
        return ItemGenerationResponse.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pydantic ValidationError or json decode error
        raise ItemBankError(f"draft file {path} is not a valid item set: {exc}") from exc


def list_drafts(locale: str) -> List[Path]:
    """All draft files for a locale, oldest filename first (timestamps sort lexically)."""
    target_dir = drafts_dir(locale)
    if not target_dir.exists():
        return []
    return sorted(target_dir.glob("*.json"))


def load_approved(
    locale: str,
    grade_band: str,
    *,
    locale_config: Optional[LocaleConfig] = None,
) -> List[AssessmentItem]:
    """Load the approved, ready-to-serve assessment for one locale and grade band.

    Every student in a class answers this same fixed set (see the `items_per_assessment` note in
    the locale YAML). Validation is deliberately strict and fails loudly, because everything it
    checks is a way for a broken assessment to reach children silently:

    - duplicate item ids would let one item be scored twice, skewing a student's profile
    - an item still flagged `needs_translation` has not been through the translation step
    - a mismatched locale or grade band means the wrong pack was picked up
    - the wrong item count breaks the profiling resolution the thresholds assume

    Pass `locale_config` to also enforce the configured item count and point budget.
    """
    path = approved_path(locale, grade_band)
    if not path.exists():
        raise ItemBankError(
            f"no approved assessment for locale {locale!r} grade band {grade_band!r} (expected {path}). "
            f"Generate drafts, review and translate them, then save the approved set there."
        )

    try:
        parsed = ItemGenerationResponse.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ItemBankError(f"approved file {path} is not a valid item set: {exc}") from exc

    items = parsed.items
    if not items:
        raise ItemBankError(f"approved file {path} contains no items")

    _check_unique_ids(items, path)
    _check_translated(items, path)
    _check_locale_and_band(items, path, locale=locale, grade_band=grade_band)

    if locale_config is not None:
        _check_against_config(items, path, locale_config=locale_config)

    return items


def _check_unique_ids(items: List[AssessmentItem], path: Path) -> None:
    seen = set()
    duplicates = set()
    for item in items:
        if item.id in seen:
            duplicates.add(item.id)
        seen.add(item.id)
    if duplicates:
        raise ItemBankError(f"approved file {path} has duplicate item ids: {sorted(duplicates)}")


def _check_translated(items: List[AssessmentItem], path: Path) -> None:
    untranslated = [item.id for item in items if item.needs_translation]
    if untranslated:
        raise ItemBankError(
            f"approved file {path} contains items still flagged needs_translation: "
            f"{untranslated}. Translate them and clear the flag before approving."
        )

    mismatched = [
        item.id
        for item in items
        if item.delivery_language is not None
        and item.language.strip().lower() != item.delivery_language.strip().lower()
    ]
    if mismatched:
        raise ItemBankError(
            f"approved file {path} contains items whose `language` does not match their "
            f"`delivery_language`: {mismatched}"
        )


def _check_locale_and_band(
    items: List[AssessmentItem], path: Path, *, locale: str, grade_band: str
) -> None:
    wrong_locale = [item.id for item in items if item.locale != locale]
    if wrong_locale:
        raise ItemBankError(f"approved file {path} contains items from another locale: {wrong_locale}")

    wrong_band = [item.id for item in items if item.grade_band != grade_band]
    if wrong_band:
        raise ItemBankError(
            f"approved file {path} contains items for another grade band: {wrong_band}"
        )


def _check_against_config(
    items: List[AssessmentItem], path: Path, *, locale_config: LocaleConfig
) -> None:
    expected_count = locale_config.items_per_assessment
    if len(items) != expected_count:
        raise ItemBankError(
            f"approved file {path} has {len(items)} items, but locale {locale_config.locale!r} "
            f"configures items_per_assessment={expected_count}. Profiling thresholds assume that count."
        )

    wrong_budget = [item.id for item in items if item.point_budget != locale_config.point_budget]
    if wrong_budget:
        raise ItemBankError(
            f"approved file {path} contains items whose point_budget is not "
            f"{locale_config.point_budget}: {wrong_budget}"
        )
