"""Tests for recommendation schema, assembly, and persistence.

The LLM call itself is stubbed -- these cover the logic around it: cluster coverage validation,
mode differences, deterministic class-level text, and the store.
"""
import json
from datetime import datetime

import pytest
from pydantic import ValidationError

from afrilearner360_llm.clustering.cluster import ClusterSummary
from afrilearner360_llm.common.locale_config import LocaleConfig
from afrilearner360_llm.item_generation.schema import Trait
from afrilearner360_llm.recommendations import generator as generator_module
from afrilearner360_llm.recommendations import store as store_module
from afrilearner360_llm.recommendations.generator import (
    default_running_the_class,
    describe_class,
    generate_recommendations,
)
from afrilearner360_llm.recommendations.schema import (
    ClassroomRecommendations,
    ClusterRecommendation,
    RecommendationDraft,
    RecommendationMode,
    SingleClusterDraft,
)
from afrilearner360_llm.recommendations.store import (
    RecommendationStoreError,
    latest_recommendations,
    list_recommendations,
    load_recommendations,
    save_recommendations,
)

GENERATED_AT = datetime(2026, 8, 13, 10, 45, 0)


@pytest.fixture
def locale_config():
    return LocaleConfig(
        locale="rwanda",
        display_name="Rwanda",
        trait_definitions={},
        point_budget=10,
        items_per_assessment=10,
        generation_language="english",
        delivery_language_by_grade_band={"upper_primary": "english"},
        recommendation_context={"pupil_teacher_ratio": "44:1", "ict_assumption": "no devices"},
        knowledge_base_text="Igisoro is a board game played with seeds.",
    )


def _summary(cluster_id, size, primary=Trait.VISUAL, balanced=False):
    proportions = {Trait.VISUAL: 25.0, Trait.GAME: 25.0, Trait.STRUCTURED: 25.0, Trait.STORY: 25.0}
    if not balanced:
        proportions = {**proportions, primary: 70.0}
    return ClusterSummary(
        cluster_id=cluster_id,
        size=size,
        student_ids=[f"s{cluster_id}-{n}" for n in range(size)],
        mean_proportions=proportions,
        primary_trait=primary,
        secondary_trait=None,
        is_balanced=balanced,
    )


def _summaries():
    return [
        _summary(0, 10, Trait.VISUAL),
        _summary(1, 12, Trait.GAME),
        _summary(2, 8, Trait.STRUCTURED),
    ]


def _recommendation(cluster_id):
    return ClusterRecommendation(
        cluster_id=cluster_id,
        group_label=f"Group {cluster_id}",
        teaching_strategies=["Draw the idea on the chalkboard first"],
        classroom_activity="Pairs copy an Imigongo pattern and shade one part",
        materials_needed=["chalkboard", "exercise books"],
        watch_out_for="The sequence can lose pace if it runs too long",
    )


def _stub_whole_class(monkeypatch, cluster_ids=(0, 1, 2)):
    draft = RecommendationDraft(
        class_overview="A mixed class.",
        running_the_class=["Introduce to everyone first."],
        cluster_recommendations=[_recommendation(cid) for cid in cluster_ids],
        constraints_addressed=["44:1"],
        cultural_anchors_used=["Igisoro"],
        reviewer_notes="no concerns",
    )
    monkeypatch.setattr(generator_module, "_call_model", lambda *a, **k: draft)
    return draft


def _stub_per_cluster(monkeypatch, echoed_cluster_id=None):
    calls = []

    def fake_call(prompt, *, schema_model, model, label):
        calls.append(label)
        cluster_id = echoed_cluster_id if echoed_cluster_id is not None else len(calls) - 1
        return SingleClusterDraft(
            recommendation=_recommendation(cluster_id),
            constraints_addressed=["44:1"],
            cultural_anchors_used=["Igisoro"],
            reviewer_notes="check timing",
        )

    monkeypatch.setattr(generator_module, "_call_model", fake_call)
    return calls


# --- schema -----------------------------------------------------------------------------------


def _classroom(**overrides):
    base = dict(
        locale="rwanda",
        class_id="P5A",
        grade_band="P5",
        topic="fractions",
        mode=RecommendationMode.WHOLE_CLASS,
        model="google/gemma-4-26b-a4b-it:free",
        generated_at=GENERATED_AT,
        cluster_sizes={0: 10, 1: 12, 2: 8},
        class_overview="A mixed class.",
        running_the_class=["Introduce to everyone first."],
        cluster_recommendations=[_recommendation(n) for n in range(3)],
    )
    base.update(overrides)
    return ClassroomRecommendations(**base)


def test_valid_recommendations_parse():
    recommendations = _classroom()
    assert len(recommendations.cluster_recommendations) == 3


def test_missing_cluster_is_rejected():
    # A group of real children with no guidance, which nothing downstream would necessarily show.
    with pytest.raises(ValidationError, match="no recommendation for cluster"):
        _classroom(cluster_recommendations=[_recommendation(0), _recommendation(1)])


def test_duplicate_cluster_is_rejected():
    with pytest.raises(ValidationError, match="more than one recommendation"):
        _classroom(
            cluster_recommendations=[_recommendation(0), _recommendation(1), _recommendation(1)]
        )


def test_unknown_cluster_is_rejected():
    with pytest.raises(ValidationError, match="unknown cluster"):
        _classroom(
            cluster_recommendations=[_recommendation(n) for n in range(3)] + [_recommendation(9)]
        )


# --- generation modes -------------------------------------------------------------------------


def test_whole_class_mode_makes_one_call(locale_config, monkeypatch):
    calls = []
    draft = _stub_whole_class(monkeypatch)
    monkeypatch.setattr(
        generator_module, "_call_model", lambda *a, **k: (calls.append(1), draft)[1]
    )

    result = generate_recommendations(
        locale_config=locale_config,
        summaries=_summaries(),
        class_id="P5A",
        grade_band="P5",
        topic="fractions",
        mode=RecommendationMode.WHOLE_CLASS,
        generated_at=GENERATED_AT,
    )

    assert len(calls) == 1
    assert result.mode is RecommendationMode.WHOLE_CLASS
    assert result.class_overview == "A mixed class."  # model-authored in this mode
    assert result.cluster_sizes == {0: 10, 1: 12, 2: 8}


def test_per_cluster_mode_makes_one_call_per_cluster(locale_config, monkeypatch):
    calls = _stub_per_cluster(monkeypatch)

    result = generate_recommendations(
        locale_config=locale_config,
        summaries=_summaries(),
        class_id="P5A",
        grade_band="P5",
        topic="fractions",
        mode=RecommendationMode.PER_CLUSTER,
        generated_at=GENERATED_AT,
    )

    assert len(calls) == 3
    assert len(result.cluster_recommendations) == 3
    # No call saw the whole class, so class-level advice is assembled in code.
    assert "3 groups" in result.class_overview
    assert result.running_the_class == default_running_the_class(_summaries())


def test_per_cluster_mode_corrects_a_wrong_echoed_cluster_id(locale_config, monkeypatch):
    # The model is told which cluster it is advising on but may echo the wrong id; the caller's
    # cluster is authoritative, otherwise advice silently attaches to the wrong group.
    _stub_per_cluster(monkeypatch, echoed_cluster_id=0)

    result = generate_recommendations(
        locale_config=locale_config,
        summaries=_summaries(),
        class_id="P5A",
        grade_band="P5",
        topic="fractions",
        mode=RecommendationMode.PER_CLUSTER,
        generated_at=GENERATED_AT,
    )

    assert sorted(r.cluster_id for r in result.cluster_recommendations) == [0, 1, 2]


def test_per_cluster_mode_merges_traceability_without_duplicates(locale_config, monkeypatch):
    _stub_per_cluster(monkeypatch)

    result = generate_recommendations(
        locale_config=locale_config,
        summaries=_summaries(),
        class_id="P5A",
        grade_band="P5",
        topic="fractions",
        mode=RecommendationMode.PER_CLUSTER,
        generated_at=GENERATED_AT,
    )

    assert result.cultural_anchors_used == ["Igisoro"]
    assert result.constraints_addressed == ["44:1"]
    assert "cluster 0:" in result.reviewer_notes


def test_empty_summaries_raise(locale_config):
    with pytest.raises(ValueError, match="no cluster summaries"):
        generate_recommendations(
            locale_config=locale_config,
            summaries=[],
            class_id="P5A",
            grade_band="P5",
            topic="fractions",
        )


def test_model_output_missing_a_cluster_fails_loudly(locale_config, monkeypatch):
    _stub_whole_class(monkeypatch, cluster_ids=(0, 1))

    with pytest.raises(ValidationError, match="no recommendation for cluster"):
        generate_recommendations(
            locale_config=locale_config,
            summaries=_summaries(),
            class_id="P5A",
            grade_band="P5",
            topic="fractions",
            generated_at=GENERATED_AT,
        )


# --- deterministic class-level text -------------------------------------------------------------


def test_describe_class_avoids_ability_language():
    text = describe_class(_summaries())
    assert "30 students" in text
    for banned in ("struggl", "weak", "slow", "less able", "behind"):
        assert banned not in text.lower()
    assert "ability" in text.lower()  # explicitly says these are not ability groups


def test_describe_class_handles_an_unassessed_class():
    assert "No students" in describe_class([_summary(0, 0)])


# --- store ------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def store_root(tmp_path, monkeypatch):
    monkeypatch.setattr(store_module, "PROJECT_ROOT", tmp_path)
    return tmp_path


def test_save_and_load_round_trip():
    path = save_recommendations(_classroom())
    loaded = load_recommendations(path)

    assert loaded.class_id == "P5A"
    assert loaded.topic == "fractions"
    assert len(loaded.cluster_recommendations) == 3


def test_filename_includes_band_topic_and_timestamp():
    path = save_recommendations(_classroom())
    assert path.name == "P5-fractions-20260813T104500.json"


def test_class_sections_are_stored_separately():
    save_recommendations(_classroom(class_id="P5A"))
    save_recommendations(_classroom(class_id="P5B"))

    assert len(list_recommendations("rwanda", "P5A")) == 1
    assert len(list_recommendations("rwanda", "P5B")) == 1


def test_latest_returns_most_recent():
    save_recommendations(_classroom(generated_at=datetime(2026, 8, 13, 9, 0, 0)))
    save_recommendations(_classroom(generated_at=datetime(2026, 8, 13, 15, 0, 0)))

    latest = latest_recommendations("rwanda", "P5A")
    assert latest.generated_at == datetime(2026, 8, 13, 15, 0, 0)


def test_latest_can_filter_by_topic():
    save_recommendations(_classroom(topic="fractions", generated_at=datetime(2026, 8, 13, 9, 0, 0)))
    save_recommendations(
        _classroom(topic="the water cycle", generated_at=datetime(2026, 8, 13, 15, 0, 0))
    )

    assert latest_recommendations("rwanda", "P5A", topic="fractions").topic == "fractions"


def test_latest_returns_none_for_a_class_with_nothing_stored():
    # A teacher opening a class for the first time is an ordinary state, not an error.
    assert latest_recommendations("rwanda", "P9Z") is None


def test_list_is_empty_for_unknown_class():
    assert list_recommendations("rwanda", "nope") == []


def test_malformed_file_raises(store_root):
    path = store_root / "data" / "recommendations" / "rwanda" / "P5A" / "broken.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(RecommendationStoreError, match="malformed"):
        load_recommendations(path)


def test_missing_file_raises(store_root):
    with pytest.raises(RecommendationStoreError, match="no recommendations file"):
        load_recommendations(store_root / "nope.json")


def test_stored_json_is_human_readable():
    path = save_recommendations(_classroom())
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["mode"] == "whole_class"
    assert raw["cluster_recommendations"][0]["group_label"] == "Group 0"
