"""Persistence and retrieval for generated recommendations.

Layout, keyed by class section so P5A and P5B stay separate:

    data/recommendations/<locale>/<class_id>/
      P5-fractions-20260813T104500.json

Unlike the item bank there is no drafts/approved split. Assessment items are an instrument
administered to children, so they get a blocking human gate; recommendations are advisory
guidance handed to a teacher, who is a professional applying their own judgement and can simply
disregard a bad suggestion. `reviewer_notes` still carries the model's own flags.

Files are timestamped rather than overwritten so a class keeps a history: what was suggested,
for which topic, from which clustering, by which model.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional

from ..common.config import PROJECT_ROOT
from .schema import ClassroomRecommendations


class RecommendationStoreError(Exception):
    """Raised when a recommendations file is missing or malformed."""


def recommendations_dir(locale: str, class_id: str) -> Path:
    return PROJECT_ROOT / "data" / "recommendations" / locale / class_id


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "untitled"


def save_recommendations(recommendations: ClassroomRecommendations) -> Path:
    """Write recommendations to the class's directory and return the path."""
    target_dir = recommendations_dir(recommendations.locale, recommendations.class_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        f"{recommendations.grade_band}-{_slug(recommendations.topic)}-"
        f"{recommendations.generated_at:%Y%m%dT%H%M%S}.json"
    )
    path = target_dir / filename
    path.write_text(
        json.dumps(recommendations.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def load_recommendations(path: Path) -> ClassroomRecommendations:
    if not path.exists():
        raise RecommendationStoreError(f"no recommendations file at {path}")
    try:
        return ClassroomRecommendations.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RecommendationStoreError(f"recommendations file {path} is malformed: {exc}") from exc


def list_recommendations(locale: str, class_id: str) -> List[Path]:
    """All recommendation files for a class, oldest first (timestamps sort lexically)."""
    target_dir = recommendations_dir(locale, class_id)
    if not target_dir.exists():
        return []
    return sorted(target_dir.glob("*.json"))


def latest_recommendations(
    locale: str,
    class_id: str,
    *,
    topic: Optional[str] = None,
    grade_band: Optional[str] = None,
) -> Optional[ClassroomRecommendations]:
    """Most recent recommendations for a class, optionally narrowed to a topic or grade band.

    Returns None rather than raising when a class has nothing stored yet -- a teacher opening a
    class for the first time is an ordinary state, not an error. A malformed file still raises,
    because that is a real problem worth surfacing.
    """
    candidates = list_recommendations(locale, class_id)
    for path in reversed(candidates):
        recommendations = load_recommendations(path)
        if topic is not None and recommendations.topic != topic:
            continue
        if grade_band is not None and recommendations.grade_band != grade_band:
            continue
        return recommendations
    return None
