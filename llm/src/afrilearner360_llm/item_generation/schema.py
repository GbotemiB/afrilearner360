"""Pydantic schema for LLM-generated assessment items.

Kept deliberately strict: the LLM is required to return exactly four options per item, one per
trait, so downstream profiling code (see afrilearner360_llm.profiling) can rely on the shape
without defensive parsing.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class Trait(str, Enum):
    VISUAL = "visual"
    GAME = "game"
    STRUCTURED = "structured"
    STORY = "story"


class AssessmentOption(BaseModel):
    trait: Trait = Field(..., description="Which of the four engagement traits this option measures")
    text: str = Field(..., description="The option text shown to the student, in the target language")


class AssessmentItem(BaseModel):
    id: str = Field(..., description="Short unique id, e.g. 'rw-p3-fractions-01'")
    topic: str = Field(..., description="Curriculum topic this item is themed around")
    grade_band: str = Field(..., description="Target grade band, e.g. 'P3'")
    locale: str = Field(..., description="Locale key, e.g. 'rwanda'")
    language: str = Field(..., description="Language the scenario/options are written in")
    point_budget: int = Field(..., description="Total points the student splits across the 4 options")
    scenario_text: str = Field(..., description="The short scenario/prompt shown before the options")
    options: List[AssessmentOption] = Field(..., description="Exactly 4 options, one per trait")
    cultural_anchors_used: List[str] = Field(
        default_factory=list,
        description="Specific cultural references drawn from the knowledge base, for reviewer traceability",
    )
    reviewer_notes: Optional[str] = Field(
        None,
        description="Model's own flags for a human reviewer: uncertainty, possible stereotyping, or content to double-check",
    )

    @field_validator("options")
    @classmethod
    def exactly_one_option_per_trait(cls, options: List[AssessmentOption]) -> List[AssessmentOption]:
        if len(options) != 4:
            raise ValueError(f"expected exactly 4 options, got {len(options)}")
        traits = {opt.trait for opt in options}
        if traits != set(Trait):
            raise ValueError(f"expected one option per trait {set(Trait)}, got traits {traits}")
        return options


class ItemGenerationResponse(BaseModel):
    items: List[AssessmentItem]
