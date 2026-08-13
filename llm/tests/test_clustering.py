"""Tests for classroom clustering: cold start, mid-term joiners, summaries, and staleness.

No API calls -- k-means is pure computation.
"""
from datetime import datetime, timedelta

import pytest

from afrilearner360_llm.clustering.cluster import (
    ClusteringError,
    ClusterModel,
    assign_student,
    fit_classroom,
    is_stale,
    summarize_clusters,
)
from afrilearner360_llm.item_generation.schema import Trait
from afrilearner360_llm.profiling.scorer import ItemResponse, TRAIT_ORDER, score_student

FITTED_AT = datetime(2026, 8, 13, 12, 0, 0)


def _profile(student_id, visual, game, structured, story):
    """Build a profile from a single response, so proportions are the given points x10."""
    response = ItemResponse(
        item_id="i1",
        allocations={
            Trait.VISUAL: visual,
            Trait.GAME: game,
            Trait.STRUCTURED: structured,
            Trait.STORY: story,
        },
    )
    return score_student(student_id=student_id, responses=[response])


def _roster():
    """Twelve students in three obvious groups: visual-heavy, game-heavy, structured-heavy."""
    visual = [_profile(f"v{n}", 7, 1, 1, 1) for n in range(4)]
    game = [_profile(f"g{n}", 1, 7, 1, 1) for n in range(4)]
    structured = [_profile(f"s{n}", 1, 1, 7, 1) for n in range(4)]
    return visual + game + structured


# --- cold start -------------------------------------------------------------------------------


def test_fit_assigns_every_student_exactly_once():
    roster = _roster()
    result = fit_classroom(roster, k=3, fitted_at=FITTED_AT)

    assert len(result.assignments) == len(roster)
    assert {a.student_id for a in result.assignments} == {p.student_id for p in roster}


def test_fit_separates_obvious_groups():
    result = fit_classroom(_roster(), k=3, fitted_at=FITTED_AT)
    by_student = {a.student_id: a.cluster_id for a in result.assignments}

    # All four visual-heavy students land together, and apart from the game-heavy ones.
    visual_clusters = {by_student[f"v{n}"] for n in range(4)}
    game_clusters = {by_student[f"g{n}"] for n in range(4)}
    assert len(visual_clusters) == 1
    assert len(game_clusters) == 1
    assert visual_clusters != game_clusters


def test_fit_is_deterministic():
    # A teacher asking "why is my child in this group?" needs the same answer every time.
    first = fit_classroom(_roster(), k=3, fitted_at=FITTED_AT)
    second = fit_classroom(_roster(), k=3, fitted_at=FITTED_AT)

    assert [(a.student_id, a.cluster_id) for a in first.assignments] == [
        (a.student_id, a.cluster_id) for a in second.assignments
    ]


def test_model_records_roster_size_and_trait_order():
    result = fit_classroom(_roster(), k=3, fitted_at=FITTED_AT)
    assert result.model.n_students_fitted == 12
    assert result.model.trait_order == [t.value for t in TRAIT_ORDER]
    assert result.model.k == 3
    assert len(result.model.centroids) == 3


def test_empty_roster_raises():
    with pytest.raises(ClusteringError, match="empty roster"):
        fit_classroom([])


def test_roster_smaller_than_k_raises():
    with pytest.raises(ClusteringError, match="cannot fit"):
        fit_classroom([_profile("a", 7, 1, 1, 1), _profile("b", 1, 7, 1, 1)], k=4)


def test_duplicate_student_ids_raise():
    with pytest.raises(ClusteringError, match="duplicate student ids"):
        fit_classroom([_profile("a", 7, 1, 1, 1), _profile("a", 1, 7, 1, 1), _profile("b", 1, 1, 7, 1)], k=2)


def test_k_below_two_raises():
    with pytest.raises(ClusteringError, match="at least 2"):
        fit_classroom(_roster(), k=1)


# --- mid-term joiners -------------------------------------------------------------------------


def test_joiner_goes_to_nearest_centroid():
    result = fit_classroom(_roster(), k=3, fitted_at=FITTED_AT)
    by_student = {a.student_id: a.cluster_id for a in result.assignments}

    joiner = _profile("new", 8, 1, 1, 0)  # clearly visual
    assignment = assign_student(result.model, joiner)

    assert assignment.cluster_id == by_student["v0"]
    assert assignment.distance_to_centroid >= 0


def test_joiner_does_not_change_the_model():
    result = fit_classroom(_roster(), k=3, fitted_at=FITTED_AT)
    centroids_before = [list(c) for c in result.model.centroids]

    assign_student(result.model, _profile("new", 8, 1, 1, 0))

    assert result.model.centroids == centroids_before
    assert result.model.n_students_fitted == 12


def test_joiner_assignment_is_deterministic():
    result = fit_classroom(_roster(), k=3, fitted_at=FITTED_AT)
    joiner = _profile("new", 4, 3, 2, 1)

    first = assign_student(result.model, joiner)
    second = assign_student(result.model, joiner)
    assert first.cluster_id == second.cluster_id
    assert first.distance_to_centroid == pytest.approx(second.distance_to_centroid)


# --- persistence ------------------------------------------------------------------------------


def test_model_round_trips_through_disk(tmp_path):
    result = fit_classroom(_roster(), k=3, fitted_at=FITTED_AT)
    path = result.model.save(tmp_path / "class-a.json")

    loaded = ClusterModel.load(path)
    assert loaded.k == 3
    assert loaded.fitted_at == FITTED_AT
    assert loaded.n_students_fitted == 12

    joiner = _profile("new", 8, 1, 1, 0)
    assert assign_student(loaded, joiner).cluster_id == assign_student(result.model, joiner).cluster_id


def test_loading_missing_model_raises(tmp_path):
    with pytest.raises(ClusteringError, match="no cluster model"):
        ClusterModel.load(tmp_path / "nope.json")


def test_model_saved_under_a_different_trait_order_is_refused(tmp_path):
    # Feature positions would silently shift, putting students in the wrong groups.
    result = fit_classroom(_roster(), k=3, fitted_at=FITTED_AT)
    raw = result.model.to_dict()
    raw["trait_order"] = ["story", "game", "structured", "visual"]
    path = tmp_path / "stale.json"
    path.write_text(__import__("json").dumps(raw), encoding="utf-8")

    with pytest.raises(ClusteringError, match="trait_order"):
        ClusterModel.load(path)


def test_centroid_count_mismatch_is_refused():
    with pytest.raises(ClusteringError, match="holds"):
        ClusterModel.from_dict(
            {
                "centroids": [[25.0, 25.0, 25.0, 25.0]],
                "k": 3,
                "trait_order": [t.value for t in TRAIT_ORDER],
                "fitted_at": FITTED_AT.isoformat(),
                "n_students_fitted": 12,
                "random_state": 0,
            }
        )


# --- summaries --------------------------------------------------------------------------------


def test_summaries_cover_every_cluster_and_all_students():
    result = fit_classroom(_roster(), k=3, fitted_at=FITTED_AT)

    assert len(result.summaries) == 3
    assert sum(s.size for s in result.summaries) == 12
    assert sorted(sid for s in result.summaries for sid in s.student_ids) == sorted(
        p.student_id for p in _roster()
    )


def test_summary_describes_the_dominant_trait():
    result = fit_classroom(_roster(), k=3, fitted_at=FITTED_AT)
    by_student = {a.student_id: a.cluster_id for a in result.assignments}
    visual_cluster = next(s for s in result.summaries if s.cluster_id == by_student["v0"])

    assert visual_cluster.primary_trait == Trait.VISUAL
    assert visual_cluster.mean_proportions[Trait.VISUAL] == pytest.approx(70.0)
    assert "visual" in visual_cluster.descriptor()


def test_descriptor_wording_by_shape():
    strong = _profile("a", 7, 1, 1, 1)
    blended = _profile("b", 4, 3, 2, 1)
    even = _profile("c", 3, 3, 2, 2)

    result = fit_classroom([strong, blended, even], k=3, fitted_at=FITTED_AT)
    descriptors = {s.student_ids[0]: s.descriptor() for s in result.summaries if s.student_ids}

    assert descriptors["a"] == "strongly visual"
    assert descriptors["b"] == "visual-leaning, game-secondary"
    assert descriptors["c"] == "balanced across modes"


def test_empty_cluster_is_reported_not_dropped():
    # A teacher promised k groups must be told one came out empty.
    result = fit_classroom(_roster(), k=3, fitted_at=FITTED_AT)
    summaries = summarize_clusters(result.model, [], _roster())

    assert len(summaries) == 3
    assert all(s.size == 0 for s in summaries)
    assert all(s.student_ids == [] for s in summaries)


def test_assignment_outside_k_raises():
    result = fit_classroom(_roster(), k=3, fitted_at=FITTED_AT)
    bad = result.assignments[0]
    bad.cluster_id = 99

    with pytest.raises(ClusteringError, match="outside k"):
        summarize_clusters(result.model, [bad], _roster())


# --- scheduled recompute ----------------------------------------------------------------------


def test_fresh_model_is_not_stale():
    result = fit_classroom(_roster(), k=3, fitted_at=FITTED_AT)
    assert is_stale(result.model, now=FITTED_AT + timedelta(days=30)) is False


def test_model_past_the_term_boundary_is_stale():
    result = fit_classroom(_roster(), k=3, fitted_at=FITTED_AT)
    assert is_stale(result.model, now=FITTED_AT + timedelta(days=91)) is True


def test_staleness_window_is_configurable():
    result = fit_classroom(_roster(), k=3, fitted_at=FITTED_AT)
    assert is_stale(result.model, now=FITTED_AT + timedelta(days=31), max_age_days=30) is True
