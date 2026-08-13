"""Tests for generator helpers that need no API call: response parsing and translation stamping."""
import json

import pytest
from pydantic import ValidationError

from afrilearner360_llm.item_generation.generator import (
    _parse_response,
    _stamp_translation_status,
)
from afrilearner360_llm.item_generation.schema import (
    AssessmentItem,
    AssessmentOption,
    ItemGenerationResponse,
    Trait,
)


def _item(item_id="rwanda-p3-fractions-01", language="english"):
    return AssessmentItem(
        id=item_id,
        topic="fractions",
        grade_band="P3",
        locale="rwanda",
        language=language,
        point_budget=10,
        scenario_text="Your class is learning how a whole is divided into equal parts...",
        options=[
            AssessmentOption(trait=Trait.VISUAL, text="Look at an Imigongo pattern"),
            AssessmentOption(trait=Trait.GAME, text="Play a game with Igisoro seeds"),
            AssessmentOption(trait=Trait.STRUCTURED, text="Follow the steps to divide a shape"),
            AssessmentOption(trait=Trait.STORY, text="Listen to a story about sharing a harvest"),
        ],
    )


def test_stamps_needs_translation_when_languages_differ():
    response = ItemGenerationResponse(items=[_item()])
    stamped = _stamp_translation_status(response, delivery_language="kinyarwanda")

    item = stamped.items[0]
    assert item.delivery_language == "kinyarwanda"
    assert item.needs_translation is True
    # The drafted text itself is untouched -- translation is a human step, not a rewrite here.
    assert item.language == "english"


def test_no_translation_needed_when_languages_match():
    response = ItemGenerationResponse(items=[_item()])
    stamped = _stamp_translation_status(response, delivery_language="english")

    assert stamped.items[0].delivery_language == "english"
    assert stamped.items[0].needs_translation is False


def test_language_comparison_ignores_case_and_whitespace():
    response = ItemGenerationResponse(items=[_item(language="  English ")])
    stamped = _stamp_translation_status(response, delivery_language="english")
    assert stamped.items[0].needs_translation is False


def test_stamps_every_item():
    response = ItemGenerationResponse(items=[_item("a"), _item("b"), _item("c")])
    stamped = _stamp_translation_status(response, delivery_language="kinyarwanda")
    assert all(item.needs_translation for item in stamped.items)
    assert all(item.delivery_language == "kinyarwanda" for item in stamped.items)


def test_parses_plain_json():
    payload = json.dumps(ItemGenerationResponse(items=[_item()]).model_dump())
    parsed = _parse_response(payload)
    assert parsed.items[0].id == "rwanda-p3-fractions-01"


def test_parses_json_wrapped_in_markdown_fence():
    # Not every model routed through OpenRouter honors strict json_schema mode identically.
    payload = json.dumps(ItemGenerationResponse(items=[_item()]).model_dump())
    parsed = _parse_response(f"```json\n{payload}\n```")
    assert parsed.items[0].topic == "fractions"


def test_malformed_item_raises_rather_than_being_silently_dropped():
    bad = {"items": [{"id": "x", "topic": "fractions"}]}
    with pytest.raises(ValidationError):
        _parse_response(json.dumps(bad))
