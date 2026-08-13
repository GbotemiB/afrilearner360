"""Classroom clustering over student trait-proportion vectors.

Design notes (see DESIGN.md §4.3 for full rationale):
- **Fixed small k (3-4), not auto-selected.** Picking k per class via silhouette score is more
  statistically defensible but yields an unpredictable number of groups per classroom, which is
  hard to build a stable teacher-facing template around. A teacher planning three station
  activities wants three groups every term, not three this term and six the next.
- **k-means, not rule-based binning by dominant trait.** Binning is simpler but discards the
  blended-profile nuance that the whole non-binary framing rests on: two students who are both
  "primary visual" but one at 80% and one at 35% do not need the same teaching.
- **Deterministic.** A fixed `random_state` means the same roster always produces the same
  grouping. Non-reproducible groupings would be indefensible when a teacher asks why a child is
  in a particular group.
- This is plain computation, so no LLM is involved -- k-means is pure distance in feature space
  and has no cultural or linguistic content to interpret. The LLM enters at the next stage, when
  these clusters are turned into teaching recommendations.

Lifecycle, in three parts:
1. `fit_classroom` -- cold start over the whole initial roster.
2. `assign_student` -- a student joining mid-term goes to the nearest existing centroid. Cheap,
   instant, and it does not reshuffle anyone else or invalidate recommendations the teacher is
   already working from.
3. `is_stale` -- reports whether a scheduled full recompute is due. Actually *triggering* the
   recompute belongs to whatever schedules it (the team's Flask app), not to this module; the
   design calls for predictable, scheduled reclustering rather than threshold-based churn.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.cluster import KMeans

from ..item_generation.schema import Trait
from ..profiling.scorer import (
    DEFAULT_BALANCED_SPREAD,
    DEFAULT_SECONDARY_GAP,
    TRAIT_ORDER,
    StudentProfile,
    classify_proportions,
)

# Default number of groups. 4 fits the four traits without implying a one-cluster-per-trait
# mapping (clusters are blends, not trait buckets), and is a workable number of stations or
# activity groups for one teacher to run.
DEFAULT_K = 4

# Fixed so the same roster always yields the same grouping. Overridable, but changing it changes
# every teacher's groups, so treat it as a constant rather than a knob.
DEFAULT_RANDOM_STATE = 0

# k-means is sensitive to initialisation; 10 restarts keeps the result stable without meaningful
# cost at classroom scale (tens of students, not thousands).
N_INIT = 10


class ClusteringError(Exception):
    """Raised when a roster or a saved model cannot support clustering."""


@dataclass
class ClusterModel:
    """Fitted centroids plus the metadata needed to reuse and age them.

    Serializable because the mid-term joiner path (`assign_student`) must work without the
    original roster: the class is clustered once at term start, the model is stored, and each
    later joiner is placed against it.
    """

    centroids: List[List[float]]
    k: int
    trait_order: List[str]
    fitted_at: datetime
    n_students_fitted: int
    random_state: int = DEFAULT_RANDOM_STATE

    def centroid_proportions(self, cluster_id: int) -> Dict[Trait, float]:
        """One centroid as a trait -> proportion mapping."""
        if not 0 <= cluster_id < self.k:
            raise ClusteringError(f"cluster_id {cluster_id} out of range for k={self.k}")
        return {
            Trait(trait_name): value
            for trait_name, value in zip(self.trait_order, self.centroids[cluster_id])
        }

    def to_dict(self) -> dict:
        return {
            "centroids": self.centroids,
            "k": self.k,
            "trait_order": self.trait_order,
            "fitted_at": self.fitted_at.isoformat(),
            "n_students_fitted": self.n_students_fitted,
            "random_state": self.random_state,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "ClusterModel":
        model = cls(
            centroids=raw["centroids"],
            k=raw["k"],
            trait_order=raw["trait_order"],
            fitted_at=datetime.fromisoformat(raw["fitted_at"]),
            n_students_fitted=raw["n_students_fitted"],
            random_state=raw["random_state"],
        )
        model._validate_trait_order()
        return model

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "ClusterModel":
        if not path.exists():
            raise ClusteringError(f"no cluster model at {path}")
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except ClusteringError:
            raise
        except Exception as exc:
            raise ClusteringError(f"cluster model at {path} is malformed: {exc}") from exc

    def _validate_trait_order(self) -> None:
        """A model saved under a different trait ordering would silently mismatch feature
        positions, putting students in the wrong groups with no visible error."""
        expected = [trait.value for trait in TRAIT_ORDER]
        if self.trait_order != expected:
            raise ClusteringError(
                f"cluster model trait_order {self.trait_order} does not match the current "
                f"TRAIT_ORDER {expected}; refit the class rather than reusing it"
            )
        if len(self.centroids) != self.k:
            raise ClusteringError(
                f"cluster model claims k={self.k} but holds {len(self.centroids)} centroids"
            )


@dataclass
class ClusterAssignment:
    student_id: str
    cluster_id: int
    distance_to_centroid: float


@dataclass
class ClusterSummary:
    """One cluster, described for the teacher-facing Diversity Report."""

    cluster_id: int
    size: int
    student_ids: List[str]
    mean_proportions: Dict[Trait, float]
    primary_trait: Trait
    secondary_trait: Optional[Trait]
    is_balanced: bool

    def descriptor(self) -> str:
        """Short human-readable label, e.g. 'visual-leaning, game-secondary'.

        Deterministic string formatting, not an LLM call -- the LLM's job at the next stage is
        turning this into teaching guidance, not naming the group.
        """
        if self.is_balanced:
            return "balanced across modes"
        if self.secondary_trait is not None:
            return f"{self.primary_trait.value}-leaning, {self.secondary_trait.value}-secondary"
        return f"strongly {self.primary_trait.value}"


@dataclass
class ClassroomClustering:
    model: ClusterModel
    assignments: List[ClusterAssignment]
    summaries: List[ClusterSummary]


def fit_classroom(
    profiles: Sequence[StudentProfile],
    *,
    k: int = DEFAULT_K,
    random_state: int = DEFAULT_RANDOM_STATE,
    fitted_at: Optional[datetime] = None,
    secondary_gap_threshold: float = DEFAULT_SECONDARY_GAP,
    balanced_spread_threshold: float = DEFAULT_BALANCED_SPREAD,
) -> ClassroomClustering:
    """Cold start: cluster a whole class roster into `k` groups.

    Raises ClusteringError when the roster cannot support k groups (fewer students than k, or
    duplicate student ids). Both are caller bugs that would otherwise produce a silently
    meaningless grouping -- an empty cluster, or one student counted twice.
    """
    if not profiles:
        raise ClusteringError("cannot cluster an empty roster")

    if k < 2:
        raise ClusteringError(f"k must be at least 2, got {k}")

    student_ids = [profile.student_id for profile in profiles]
    duplicates = sorted({sid for sid in student_ids if student_ids.count(sid) > 1})
    if duplicates:
        raise ClusteringError(f"roster has duplicate student ids: {duplicates}")

    if len(profiles) < k:
        raise ClusteringError(
            f"cannot fit {k} clusters over {len(profiles)} student(s); "
            f"reduce k or wait until more of the class has been assessed"
        )

    features = np.array([profile.vector() for profile in profiles], dtype=float)
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=N_INIT)
    labels = kmeans.fit_predict(features)

    model = ClusterModel(
        centroids=kmeans.cluster_centers_.tolist(),
        k=k,
        trait_order=[trait.value for trait in TRAIT_ORDER],
        fitted_at=fitted_at or datetime.now(),
        n_students_fitted=len(profiles),
        random_state=random_state,
    )

    assignments = [
        ClusterAssignment(
            student_id=profile.student_id,
            cluster_id=int(label),
            distance_to_centroid=float(
                np.linalg.norm(features[index] - kmeans.cluster_centers_[label])
            ),
        )
        for index, (profile, label) in enumerate(zip(profiles, labels))
    ]

    summaries = summarize_clusters(
        model,
        assignments,
        profiles,
        secondary_gap_threshold=secondary_gap_threshold,
        balanced_spread_threshold=balanced_spread_threshold,
    )

    return ClassroomClustering(model=model, assignments=assignments, summaries=summaries)


def assign_student(model: ClusterModel, profile: StudentProfile) -> ClusterAssignment:
    """Place a mid-term joiner into an existing cluster by nearest centroid.

    Deliberately does not refit: refitting on every arrival would reshuffle students the teacher
    has already grouped and invalidate recommendations already in use. The cost is that centroids
    drift out of date as joiners accumulate, which the scheduled recompute (`is_stale`) exists to
    correct at a predictable moment.
    """
    model._validate_trait_order()

    vector = np.array(profile.vector(), dtype=float)
    centroids = np.array(model.centroids, dtype=float)
    distances = np.linalg.norm(centroids - vector, axis=1)
    cluster_id = int(np.argmin(distances))

    return ClusterAssignment(
        student_id=profile.student_id,
        cluster_id=cluster_id,
        distance_to_centroid=float(distances[cluster_id]),
    )


def summarize_clusters(
    model: ClusterModel,
    assignments: Sequence[ClusterAssignment],
    profiles: Sequence[StudentProfile],
    *,
    secondary_gap_threshold: float = DEFAULT_SECONDARY_GAP,
    balanced_spread_threshold: float = DEFAULT_BALANCED_SPREAD,
) -> List[ClusterSummary]:
    """Describe each cluster for the Diversity Report and the recommendation prompt.

    Empty clusters are still reported (with size 0 and the centroid's own proportions) rather than
    dropped: a teacher expecting `k` groups should be told one came out empty, not silently handed
    fewer groups than the system promised.
    """
    profiles_by_id = {profile.student_id: profile for profile in profiles}
    members: Dict[int, List[str]] = {cluster_id: [] for cluster_id in range(model.k)}
    for assignment in assignments:
        if assignment.cluster_id not in members:
            raise ClusteringError(
                f"assignment for {assignment.student_id!r} references cluster "
                f"{assignment.cluster_id}, outside k={model.k}"
            )
        members[assignment.cluster_id].append(assignment.student_id)

    summaries = []
    for cluster_id in range(model.k):
        student_ids = members[cluster_id]

        if student_ids:
            mean_proportions = {
                trait: float(
                    np.mean([profiles_by_id[sid].proportions[trait] for sid in student_ids])
                )
                for trait in TRAIT_ORDER
            }
        else:
            mean_proportions = model.centroid_proportions(cluster_id)

        dominance = classify_proportions(
            mean_proportions,
            secondary_gap_threshold=secondary_gap_threshold,
            balanced_spread_threshold=balanced_spread_threshold,
        )

        summaries.append(
            ClusterSummary(
                cluster_id=cluster_id,
                size=len(student_ids),
                student_ids=student_ids,
                mean_proportions=mean_proportions,
                primary_trait=dominance.primary_trait,
                secondary_trait=dominance.secondary_trait,
                is_balanced=dominance.is_balanced,
            )
        )

    return summaries


def is_stale(
    model: ClusterModel, *, now: Optional[datetime] = None, max_age_days: int = 90
) -> bool:
    """Whether a scheduled full recompute is due (default ~one school term).

    A predicate, not a trigger: the design calls for reclustering at predictable moments (term
    start) rather than whenever some threshold trips, so the scheduling itself belongs to the
    application that owns the calendar. This just answers the question when it asks.
    """
    now = now or datetime.now()
    return (now - model.fitted_at) >= timedelta(days=max_age_days)
