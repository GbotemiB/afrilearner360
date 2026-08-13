"""Tests for the deterministic profiling scorer -- no API calls, pure arithmetic."""
import pytest
from pydantic import ValidationError

from afrilearner360_llm.item_generation.schema import Trait
from afrilearner360_llm.profiling.scorer import (
    TRAIT_ORDER,
    ItemResponse,
    score_student,
)


def _response(item_id, visual, game, structured, story):
    return ItemResponse(
        item_id=item_id,
        allocations={
            Trait.VISUAL: visual,
            Trait.GAME: game,
            Trait.STRUCTURED: structured,
            Trait.STORY: story,
        },
    )


def test_proportions_sum_to_100_and_match_raw_points():
    profile = score_student(
        student_id="s1",
        responses=[
            _response("i1", 6, 2, 1, 1),
            _response("i2", 4, 3, 2, 1),
        ],
    )
    assert profile.total_points == 20
    assert profile.items_answered == 2
    assert profile.raw_points == {
        Trait.VISUAL: 10,
        Trait.GAME: 5,
        Trait.STRUCTURED: 3,
        Trait.STORY: 2,
    }
    assert profile.proportions[Trait.VISUAL] == pytest.approx(50.0)
    assert sum(profile.proportions.values()) == pytest.approx(100.0)


def test_clear_dominant_trait_has_no_secondary():
    # 70/10/10/10 -- runner-up is 60 points behind, well outside the secondary threshold.
    profile = score_student(student_id="s1", responses=[_response("i1", 7, 1, 1, 1)])
    assert profile.primary_trait == Trait.VISUAL
    assert profile.secondary_trait is None
    assert profile.is_balanced is False


def test_close_runner_up_becomes_secondary():
    # 40/30/20/10 -- gap of 10 points is inside the 15-point secondary threshold.
    profile = score_student(student_id="s1", responses=[_response("i1", 4, 3, 2, 1)])
    assert profile.primary_trait == Trait.VISUAL
    assert profile.secondary_trait == Trait.GAME
    assert profile.is_balanced is False


def test_runner_up_outside_threshold_is_dropped():
    # 50/30/10/10 -- gap of 20 points exceeds the default 15-point threshold.
    profile = score_student(student_id="s1", responses=[_response("i1", 5, 3, 1, 1)])
    assert profile.primary_trait == Trait.VISUAL
    assert profile.secondary_trait is None


def test_balanced_profile_flags_and_suppresses_secondary():
    # 30/30/20/20 -- spread of 10 points is under the 12-point balanced threshold.
    profile = score_student(student_id="s1", responses=[_response("i1", 3, 3, 2, 2)])
    assert profile.is_balanced is True
    assert profile.secondary_trait is None


def test_perfectly_even_split_is_balanced():
    profile = score_student(student_id="s1", responses=[_response("i1", 3, 3, 2, 2), _response("i2", 2, 2, 3, 3)])
    assert profile.proportions[Trait.VISUAL] == pytest.approx(25.0)
    assert profile.is_balanced is True


def test_thresholds_are_overridable():
    # Same 30/30/20/20 input, but a stricter balanced threshold makes it non-balanced, which
    # then lets the tied runner-up surface as secondary.
    profile = score_student(
        student_id="s1",
        responses=[_response("i1", 3, 3, 2, 2)],
        balanced_spread_threshold=5.0,
    )
    assert profile.is_balanced is False
    assert profile.primary_trait == Trait.VISUAL
    assert profile.secondary_trait == Trait.GAME


def test_ties_resolve_deterministically_by_trait_order():
    # Story and visual tie at 40 each; TRAIT_ORDER puts visual first, so it wins consistently.
    first = score_student(student_id="s1", responses=[_response("i1", 4, 1, 1, 4)])
    second = score_student(student_id="s1", responses=[_response("i1", 4, 1, 1, 4)])
    assert first.primary_trait == Trait.VISUAL
    assert first.model_dump() == second.model_dump()


def test_vector_follows_trait_order():
    profile = score_student(student_id="s1", responses=[_response("i1", 4, 3, 2, 1)])
    assert profile.vector() == [
        profile.proportions[trait] for trait in TRAIT_ORDER
    ]
    assert profile.vector() == pytest.approx([40.0, 30.0, 20.0, 10.0])


def test_point_budget_mismatch_raises():
    with pytest.raises(ValueError, match="expected 10"):
        score_student(
            student_id="s1",
            responses=[_response("i1", 4, 3, 2, 1), _response("i2", 5, 5, 5, 5)],
            expected_point_budget=10,
        )


def test_matching_point_budget_passes():
    profile = score_student(
        student_id="s1",
        responses=[_response("i1", 4, 3, 2, 1), _response("i2", 1, 2, 3, 4)],
        expected_point_budget=10,
    )
    assert profile.total_points == 20


def test_duplicate_item_responses_raise():
    # Scoring the same item twice would double-weight it and silently skew the profile.
    with pytest.raises(ValueError, match="more than one response"):
        score_student(
            student_id="s1",
            responses=[_response("i1", 4, 3, 2, 1), _response("i1", 1, 2, 3, 4)],
        )


def test_unknown_item_id_raises_when_approved_set_supplied():
    with pytest.raises(ValueError, match="not in the approved"):
        score_student(
            student_id="s1",
            responses=[_response("i1", 4, 3, 2, 1), _response("i9", 1, 2, 3, 4)],
            known_item_ids={"i1", "i2"},
        )


def test_known_item_ids_accepts_a_subset():
    # A partially completed assessment is still scoreable; only unknown items are an error.
    profile = score_student(
        student_id="s1",
        responses=[_response("i1", 4, 3, 2, 1)],
        known_item_ids={"i1", "i2", "i3"},
    )
    assert profile.items_answered == 1


def test_empty_responses_raise():
    with pytest.raises(ValueError, match="no responses"):
        score_student(student_id="s1", responses=[])


def test_all_zero_allocations_raise():
    with pytest.raises(ValueError, match="0 points"):
        score_student(student_id="s1", responses=[_response("i1", 0, 0, 0, 0)])


def test_missing_trait_in_allocation_rejected():
    with pytest.raises(ValidationError):
        ItemResponse(item_id="i1", allocations={Trait.VISUAL: 5, Trait.GAME: 5})


def test_negative_points_rejected():
    with pytest.raises(ValidationError):
        _response("i1", 12, -2, 0, 0)
