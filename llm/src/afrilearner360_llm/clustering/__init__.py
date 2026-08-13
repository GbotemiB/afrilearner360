"""Classroom clustering (k-means, fixed k) over student trait-proportion vectors, with
nearest-centroid assignment for new mid-term joiners and a scheduled full recompute.
"""
from .cluster import (
    DEFAULT_K,
    DEFAULT_RANDOM_STATE,
    ClassroomClustering,
    ClusterAssignment,
    ClusteringError,
    ClusterModel,
    ClusterSummary,
    assign_student,
    fit_classroom,
    is_stale,
    summarize_clusters,
)

__all__ = [
    "DEFAULT_K",
    "DEFAULT_RANDOM_STATE",
    "ClassroomClustering",
    "ClusterAssignment",
    "ClusterModel",
    "ClusterSummary",
    "ClusteringError",
    "assign_student",
    "fit_classroom",
    "is_stale",
    "summarize_clusters",
]
