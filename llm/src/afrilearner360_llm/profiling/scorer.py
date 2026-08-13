"""Deterministic scoring of a student's forced-choice responses into a trait profile.

Design notes (see DESIGN.md §4.2 for full rationale):
- This is plain arithmetic, deliberately NOT an LLM call. Responses are numeric point
  allocations, so there is no language to interpret; using a model here would add cost,
  latency, and non-determinism while destroying auditability. We want to be able to tell a
  teacher "42% because of this math", not "the model said so".
- Forced-choice allocation already sums to a constant per item, so normalization is a single
  division -- no extra rescaling logic.
- Output proportions are IPSATIVE (relative to the student's own total, not an absolute scale).
  See DESIGN.md §6: fine for within-student dominance and for clustering shape, but a documented
  caveat when comparing raw numbers across students.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from pydantic import BaseModel, Field, field_validator

from ..item_generation.schema import Trait

# Fixed trait ordering. Everything downstream that needs a numeric vector (clustering, centroid
# distance) must use this order so feature positions are stable across runs and processes.
TRAIT_ORDER: tuple[Trait, ...] = (Trait.VISUAL, Trait.GAME, Trait.STRUCTURED, Trait.STORY)

# A second trait only counts as "secondary" if it is within this many percentage points of the
# primary. Beyond that gap the student has one clear dominant mode and reporting a runner-up
# would overstate it.
DEFAULT_SECONDARY_GAP = 15.0

# If the spread between the student's highest and lowest trait is under this many percentage
# points, no mode really dominates -- report the student as balanced rather than forcing an
# arbitrary primary out of noise.
DEFAULT_BALANCED_SPREAD = 12.0


class TraitDominance(BaseModel):
    """Which traits stand out in a proportion vector, and whether any really does.

    Applies to any set of trait proportions, not just one student's -- clustering reuses it to
    describe a cluster centroid, so a cluster and a student are characterised by identical rules.
    """

    primary_trait: Trait
    secondary_trait: Optional[Trait] = None
    is_balanced: bool


def classify_proportions(
    proportions: Dict[Trait, float],
    *,
    secondary_gap_threshold: float = DEFAULT_SECONDARY_GAP,
    balanced_spread_threshold: float = DEFAULT_BALANCED_SPREAD,
) -> TraitDominance:
    """Apply the §4.2 dominance rules to a trait-proportion vector.

    Ties resolve by TRAIT_ORDER position, so identical input always classifies identically.
    """
    ranked = sorted(TRAIT_ORDER, key=lambda trait: (-proportions[trait], TRAIT_ORDER.index(trait)))
    primary_trait = ranked[0]
    runner_up = ranked[1]

    spread = proportions[ranked[0]] - proportions[ranked[-1]]
    is_balanced = spread < balanced_spread_threshold

    secondary_trait: Optional[Trait] = None
    if not is_balanced and (proportions[primary_trait] - proportions[runner_up]) <= secondary_gap_threshold:
        secondary_trait = runner_up

    return TraitDominance(
        primary_trait=primary_trait,
        secondary_trait=secondary_trait,
        is_balanced=is_balanced,
    )


class ItemResponse(BaseModel):
    """One student's point allocation for one assessment item.

    `allocations` must contain all four traits. A trait the student gave nothing to is an
    explicit 0, not a missing key -- an absent key is far more likely to be a serialization bug
    than a deliberate answer, and silently treating it as 0 would skew the profile.
    """

    item_id: str = Field(..., description="Id of the assessment item this response answers")
    allocations: Dict[Trait, int] = Field(
        ..., description="Points the student gave to each of the four traits"
    )

    @field_validator("allocations")
    @classmethod
    def all_traits_present_and_non_negative(cls, allocations: Dict[Trait, int]) -> Dict[Trait, int]:
        if set(allocations) != set(Trait):
            raise ValueError(f"expected an allocation for every trait {set(Trait)}, got {set(allocations)}")
        negative = [trait for trait, points in allocations.items() if points < 0]
        if negative:
            raise ValueError(f"point allocations must be non-negative, got negative for {negative}")
        return allocations

    def total_points(self) -> int:
        return sum(self.allocations.values())


class StudentProfile(BaseModel):
    """A student's engagement-trait profile: proportions across all four modes, never one label."""

    student_id: str
    raw_points: Dict[Trait, int] = Field(..., description="Total points summed per trait")
    proportions: Dict[Trait, float] = Field(
        ..., description="Percentage of the student's total points per trait; sums to 100.0"
    )
    total_points: int
    items_answered: int
    primary_trait: Trait = Field(..., description="Highest-proportion trait")
    secondary_trait: Optional[Trait] = Field(
        None,
        description="Runner-up trait, only set when within the secondary-gap threshold of the primary",
    )
    is_balanced: bool = Field(
        ...,
        description="True when no mode meaningfully dominates; present the profile as balanced rather than leading with primary_trait",
    )

    def vector(self) -> List[float]:
        """Proportions in TRAIT_ORDER -- the feature vector clustering consumes."""
        return [self.proportions[trait] for trait in TRAIT_ORDER]


def score_student(
    *,
    student_id: str,
    responses: List[ItemResponse],
    expected_point_budget: Optional[int] = None,
    known_item_ids: Optional[Iterable[str]] = None,
    secondary_gap_threshold: float = DEFAULT_SECONDARY_GAP,
    balanced_spread_threshold: float = DEFAULT_BALANCED_SPREAD,
) -> StudentProfile:
    """Sum a student's allocations per trait and normalize into a proportional profile.

    `expected_point_budget` is the locale's per-item budget (see LocaleConfig.point_budget). When
    supplied, every response must allocate exactly that many points; a mismatch raises rather
    than quietly producing a distorted profile, since an item answered with the wrong budget
    would be over- or under-weighted relative to the others.

    `known_item_ids` is the approved assessment's item ids (see item_generation.item_bank
    .load_approved). When supplied, a response referencing an unknown item raises -- that means
    the caller is scoring against a different item set than the one the student actually sat.

    A student answering the same item twice always raises, supplied or not: it would double-weight
    that item's allocation and silently skew the profile.

    Raises ValueError for empty response lists, all-zero allocations, duplicate items, unknown
    items, or budget mismatches -- callers should surface these as "this student's assessment is
    incomplete or mismatched", not swallow them.
    """
    if not responses:
        raise ValueError(f"no responses supplied for student {student_id!r}")

    seen_item_ids = set()
    duplicates = set()
    for response in responses:
        if response.item_id in seen_item_ids:
            duplicates.add(response.item_id)
        seen_item_ids.add(response.item_id)
    if duplicates:
        raise ValueError(
            f"student {student_id!r} has more than one response for item(s) {sorted(duplicates)}"
        )

    if known_item_ids is not None:
        unknown = sorted(seen_item_ids - set(known_item_ids))
        if unknown:
            raise ValueError(
                f"student {student_id!r} has responses for item(s) not in the approved "
                f"assessment: {unknown}"
            )

    if expected_point_budget is not None:
        for response in responses:
            actual = response.total_points()
            if actual != expected_point_budget:
                raise ValueError(
                    f"item {response.item_id!r} for student {student_id!r} allocates {actual} "
                    f"points, expected {expected_point_budget}"
                )

    raw_points = {trait: 0 for trait in TRAIT_ORDER}
    for response in responses:
        for trait in TRAIT_ORDER:
            raw_points[trait] += response.allocations[trait]

    total_points = sum(raw_points.values())
    if total_points == 0:
        raise ValueError(f"student {student_id!r} allocated 0 points across all responses")

    proportions = {trait: 100.0 * raw_points[trait] / total_points for trait in TRAIT_ORDER}
    dominance = classify_proportions(
        proportions,
        secondary_gap_threshold=secondary_gap_threshold,
        balanced_spread_threshold=balanced_spread_threshold,
    )

    return StudentProfile(
        student_id=student_id,
        raw_points=raw_points,
        proportions=proportions,
        total_points=total_points,
        items_answered=len(responses),
        primary_trait=dominance.primary_trait,
        secondary_trait=dominance.secondary_trait,
        is_balanced=dominance.is_balanced,
    )
