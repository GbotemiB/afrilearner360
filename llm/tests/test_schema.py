"""Sanity tests for the assessment item schema -- no API calls, just validation logic."""
import pytest
from pydantic import ValidationError

from afrilearner360_llm.item_generation.schema import (
    AssessmentItem,
    AssessmentOption,
    ItemGenerationResponse,
    Trait,
)


def _valid_item(**overrides):
    base = dict(
        id="rwanda-p3-fractions-01",
        topic="fractions",
        grade_band="P3",
        locale="rwanda",
        language="english",
        point_budget=10,
        scenario_text="Your class is learning how a whole is divided into equal parts...",
        options=[
            AssessmentOption(trait=Trait.VISUAL, text="Look at the patterns on an Imigongo painting"),
            AssessmentOption(trait=Trait.GAME, text="Play a game splitting Igisoro seeds into groups"),
            AssessmentOption(trait=Trait.STRUCTURED, text="Follow the steps to divide a shape evenly"),
            AssessmentOption(trait=Trait.STORY, text="Listen to a story about a farmer sharing a harvest"),
        ],
        cultural_anchors_used=["Imigongo", "Igisoro"],
        reviewer_notes="no concerns",
    )
    base.update(overrides)
    return AssessmentItem(**base)


def test_valid_item_parses():
    item = _valid_item()
    assert len(item.options) == 4
    assert {opt.trait for opt in item.options} == set(Trait)


def test_rejects_missing_trait():
    options = [
        AssessmentOption(trait=Trait.VISUAL, text="a"),
        AssessmentOption(trait=Trait.VISUAL, text="b"),  # duplicate trait, missing another
        AssessmentOption(trait=Trait.STRUCTURED, text="c"),
        AssessmentOption(trait=Trait.STORY, text="d"),
    ]
    with pytest.raises(ValidationError):
        _valid_item(options=options)


def test_rejects_wrong_option_count():
    options = [
        AssessmentOption(trait=Trait.VISUAL, text="a"),
        AssessmentOption(trait=Trait.GAME, text="b"),
        AssessmentOption(trait=Trait.STRUCTURED, text="c"),
    ]
    with pytest.raises(ValidationError):
        _valid_item(options=options)


def test_translation_fields_default_to_unstamped():
    # The model never fills these in -- generator.py stamps them after parsing, so a freshly
    # parsed item must look "not yet stamped" rather than "no translation needed".
    item = _valid_item()
    assert item.delivery_language is None
    assert item.needs_translation is False


def test_generation_response_wraps_items():
    response = ItemGenerationResponse(items=[_valid_item()])
    assert len(response.items) == 1
    # round-trips through dict/JSON cleanly (used by generator.py + the CLI script)
    dumped = response.model_dump()
    assert dumped["items"][0]["locale"] == "rwanda"
