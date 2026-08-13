"""Sanity tests for the assessment item schema -- no API calls, just validation logic."""
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from afrilearner360_llm.item_generation.schema import (  # noqa: E402
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
        language="kinyarwanda",
        point_budget=10,
        scenario_text="Umwarimu arigisha ku bice bya fraction...",
        options=[
            AssessmentOption(trait=Trait.VISUAL, text="Reba amashusho y'ibice by'igikoni"),
            AssessmentOption(trait=Trait.GAME, text="Kina umukino wo gutandukanya ibice"),
            AssessmentOption(trait=Trait.STRUCTURED, text="Kurikira intambwe zigaragaza ibice"),
            AssessmentOption(trait=Trait.STORY, text="Umva inkuru y'umuhinzi ugabanya umusaruro"),
        ],
        cultural_anchors_used=["local farming reference"],
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


def test_generation_response_wraps_items():
    response = ItemGenerationResponse(items=[_valid_item()])
    assert len(response.items) == 1
    # round-trips through dict/JSON cleanly (used by generator.py + the CLI script)
    dumped = response.model_dump()
    assert dumped["items"][0]["locale"] == "rwanda"
