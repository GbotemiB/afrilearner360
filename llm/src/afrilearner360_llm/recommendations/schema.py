"""Schemas for per-cluster teaching recommendations.

Two layers, deliberately separate:
- `RecommendationDraft` / `ClusterRecommendation` -- what the LLM is asked to produce.
- `ClassroomRecommendations` -- the stored artifact, which additionally carries metadata the
  pipeline stamps (locale, class, topic, model, mode, timestamps, cluster sizes). The model is
  never asked for those: it would have to guess, and they are facts the caller already knows.

Same split, and same reasoning, as the assessment item schema.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class RecommendationMode(str, Enum):
    """How many clusters one LLM call covers.

    WHOLE_CLASS sends every cluster in a single call, so the model can differentiate the groups
    against each other and plan how they coexist in one room. It also costs one request instead
    of k, which matters against a 50-request/day free tier.

    PER_CLUSTER sends one call per group. Simpler and failure-isolated, and the natural choice
    when clusters are being (re)generated piecemeal -- but each group is planned blind to the
    others, so the class-level advice is assembled in code rather than authored by the model.
    """

    WHOLE_CLASS = "whole_class"
    PER_CLUSTER = "per_cluster"


class ClusterRecommendation(BaseModel):
    """How to teach one group. Methods, not lesson content -- see DESIGN.md §2 scope line."""

    cluster_id: int = Field(..., description="Which cluster this advice is for")
    group_label: str = Field(
        ..., description="Short teacher-facing name for the group, e.g. 'Group 2 - visual-leaning'"
    )
    teaching_strategies: List[str] = Field(
        ..., description="Three to five concrete teaching approaches suited to this group"
    )
    classroom_activity: str = Field(
        ..., description="One concrete activity the teacher could run with this group for this topic"
    )
    materials_needed: List[str] = Field(
        ...,
        description="Materials the activity requires; must be realistic for the locale's resource constraints",
    )
    watch_out_for: Optional[str] = Field(
        None, description="A pitfall for this group, phrased about the teaching, never about the children"
    )


class RecommendationDraft(BaseModel):
    """Raw LLM output in whole-class mode."""

    class_overview: str = Field(
        ..., description="Plain-language summary of the class's spread across the four modes, for the teacher"
    )
    running_the_class: List[str] = Field(
        ...,
        description="How to run these groups together in one room given the locale's class size and resources",
    )
    cluster_recommendations: List[ClusterRecommendation] = Field(
        ..., description="One entry per cluster"
    )
    constraints_addressed: List[str] = Field(
        default_factory=list,
        description="Which locale resource/pedagogical constraints this advice accounts for, for reviewer traceability",
    )
    cultural_anchors_used: List[str] = Field(
        default_factory=list,
        description="Specific cultural references drawn from the knowledge base, for reviewer traceability",
    )
    reviewer_notes: Optional[str] = Field(
        None, description="Model's own flags: uncertainty, assumptions made, anything to double-check"
    )


class SingleClusterDraft(BaseModel):
    """Raw LLM output in per-cluster mode: one group, no class-level advice."""

    recommendation: ClusterRecommendation
    constraints_addressed: List[str] = Field(default_factory=list)
    cultural_anchors_used: List[str] = Field(default_factory=list)
    reviewer_notes: Optional[str] = Field(None)


class ClassroomRecommendations(BaseModel):
    """The stored artifact: advice for one class, one topic, one point in time."""

    locale: str
    class_id: str = Field(..., description="Identifies the class section, e.g. 'P5A'")
    grade_band: str
    topic: str
    mode: RecommendationMode
    model: str = Field(..., description="Model that generated this, for traceability across runs")
    generated_at: datetime
    cluster_sizes: Dict[int, int] = Field(
        ..., description="cluster_id -> number of students, the diversity picture this advice assumed"
    )

    class_overview: str
    running_the_class: List[str]
    cluster_recommendations: List[ClusterRecommendation]
    constraints_addressed: List[str] = Field(default_factory=list)
    cultural_anchors_used: List[str] = Field(default_factory=list)
    reviewer_notes: Optional[str] = None

    @model_validator(mode="after")
    def every_cluster_covered_exactly_once(self) -> "ClassroomRecommendations":
        """A teacher handed advice for 4 groups must get 4 groups.

        A missing cluster means one group of real children has no guidance and nothing in the UI
        would necessarily show it; a duplicated one means contradictory advice for the same group.
        """
        expected = set(self.cluster_sizes)
        seen = [rec.cluster_id for rec in self.cluster_recommendations]

        duplicates = sorted({cid for cid in seen if seen.count(cid) > 1})
        if duplicates:
            raise ValueError(f"more than one recommendation for cluster(s) {duplicates}")

        missing = sorted(expected - set(seen))
        if missing:
            raise ValueError(f"no recommendation for cluster(s) {missing}")

        unexpected = sorted(set(seen) - expected)
        if unexpected:
            raise ValueError(f"recommendation for unknown cluster(s) {unexpected}")

        return self
