"""Per-cluster, topic-aware, locale-aware teaching-method recommendation generation.

LLM-based, same prompt-design pattern as item_generation: system prompt + locale-conditioned user
prompt + structured schema. Output is advisory guidance for a teacher, not lesson content -- see
DESIGN.md §4.4.
"""
from .generator import generate_recommendations
from .schema import (
    ClassroomRecommendations,
    ClusterRecommendation,
    RecommendationDraft,
    RecommendationMode,
    SingleClusterDraft,
)
from .store import (
    RecommendationStoreError,
    latest_recommendations,
    list_recommendations,
    load_recommendations,
    save_recommendations,
)

__all__ = [
    "ClassroomRecommendations",
    "ClusterRecommendation",
    "RecommendationDraft",
    "RecommendationMode",
    "RecommendationStoreError",
    "SingleClusterDraft",
    "generate_recommendations",
    "latest_recommendations",
    "list_recommendations",
    "load_recommendations",
    "save_recommendations",
]
