"""Deterministic student profiling from assessment responses (NOT an LLM call -- see
project design notes: profiling is pure arithmetic, so it stays in plain code for
auditability, speed, and reproducibility).
"""
from .scorer import (
    DEFAULT_BALANCED_SPREAD,
    DEFAULT_SECONDARY_GAP,
    TRAIT_ORDER,
    ItemResponse,
    StudentProfile,
    TraitDominance,
    classify_proportions,
    score_student,
)

__all__ = [
    "DEFAULT_BALANCED_SPREAD",
    "DEFAULT_SECONDARY_GAP",
    "TRAIT_ORDER",
    "ItemResponse",
    "StudentProfile",
    "TraitDominance",
    "classify_proportions",
    "score_student",
]
