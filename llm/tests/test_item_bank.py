"""Tests for item bank persistence and the approved-set validation gate.

No API calls. Everything writes into pytest's tmp_path via a patched PROJECT_ROOT.
"""
import json
from datetime import datetime

import pytest

from afrilearner360_llm.common.locale_config import LocaleConfig
from afrilearner360_llm.item_generation import item_bank
from afrilearner360_llm.item_generation.item_bank import (
    ItemBankError,
    list_drafts,
    load_approved,
    load_drafts,
    save_drafts,
)
from afrilearner360_llm.item_generation.schema import (
    AssessmentItem,
    AssessmentOption,
    ItemGenerationResponse,
    Trait,
)

ITEMS_PER_ASSESSMENT = 3  # small, so fixtures stay readable


@pytest.fixture(autouse=True)
def bank_root(tmp_path, monkeypatch):
    """Point the item bank at a temp directory so tests never touch the real data/ tree."""
    monkeypatch.setattr(item_bank, "PROJECT_ROOT", tmp_path)
    return tmp_path


@pytest.fixture
def locale_config():
    return LocaleConfig(
        locale="rwanda",
        display_name="Rwanda",
        trait_definitions={},
        point_budget=10,
        items_per_assessment=ITEMS_PER_ASSESSMENT,
        generation_language="english",
        delivery_language_by_grade_band={"lower_primary": "kinyarwanda"},
        recommendation_context={},
        knowledge_base_text="kb",
    )


def _item(item_id="rw-p3-01", **overrides):
    base = dict(
        id=item_id,
        topic="fractions",
        grade_band="P3",
        locale="rwanda",
        language="english",
        point_budget=10,
        scenario_text="A whole is divided into equal parts...",
        options=[
            AssessmentOption(trait=Trait.VISUAL, text="Look at an Imigongo pattern"),
            AssessmentOption(trait=Trait.GAME, text="Play a game with Igisoro seeds"),
            AssessmentOption(trait=Trait.STRUCTURED, text="Follow the steps to divide a shape"),
            AssessmentOption(trait=Trait.STORY, text="Listen to a story about sharing a harvest"),
        ],
    )
    base.update(overrides)
    return AssessmentItem(**base)


def _approved_items(count=ITEMS_PER_ASSESSMENT, **overrides):
    return [
        _item(f"rw-p3-{n:02d}", language="kinyarwanda", delivery_language="kinyarwanda", **overrides)
        for n in range(count)
    ]


def _write_approved(items, locale="rwanda", grade_band="P3"):
    path = item_bank.approved_path(locale, grade_band)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = ItemGenerationResponse(items=items).model_dump(mode="json")
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# --- drafts -----------------------------------------------------------------------------------


def test_save_drafts_writes_readable_json():
    response = ItemGenerationResponse(items=[_item()])
    path = save_drafts(response, locale="rwanda", grade_band="P3", topic="fractions")

    assert path.exists()
    assert load_drafts(path).items[0].id == "rw-p3-01"


def test_draft_filename_includes_band_topic_and_timestamp():
    response = ItemGenerationResponse(items=[_item()])
    path = save_drafts(
        response,
        locale="rwanda",
        grade_band="P3",
        topic="the water cycle",
        generated_at=datetime(2026, 8, 13, 10, 45, 0),
    )
    assert path.name == "P3-the-water-cycle-20260813T104500.json"


def test_repeated_runs_accumulate_rather_than_overwrite():
    response = ItemGenerationResponse(items=[_item()])
    first = save_drafts(
        response, locale="rwanda", grade_band="P3", topic="fractions",
        generated_at=datetime(2026, 8, 13, 10, 0, 0),
    )
    second = save_drafts(
        response, locale="rwanda", grade_band="P3", topic="fractions",
        generated_at=datetime(2026, 8, 13, 11, 0, 0),
    )
    assert first != second
    assert len(list_drafts("rwanda")) == 2


def test_list_drafts_is_empty_for_unknown_locale():
    assert list_drafts("atlantis") == []


def test_load_drafts_rejects_malformed_file(bank_root):
    path = bank_root / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ItemBankError, match="not a valid item set"):
        load_drafts(path)


# --- approved ---------------------------------------------------------------------------------


def test_load_approved_returns_items(locale_config):
    _write_approved(_approved_items())
    items = load_approved("rwanda", "P3", locale_config=locale_config)
    assert len(items) == ITEMS_PER_ASSESSMENT


def test_missing_approved_file_raises_with_guidance():
    with pytest.raises(ItemBankError, match="no approved assessment"):
        load_approved("rwanda", "P3")


def test_untranslated_item_is_refused():
    # The whole point of the gate: a draft flagged needs_translation must never be servable.
    items = _approved_items()
    items[1].needs_translation = True
    _write_approved(items)

    with pytest.raises(ItemBankError, match="needs_translation"):
        load_approved("rwanda", "P3")


def test_language_not_matching_delivery_language_is_refused():
    items = _approved_items()
    items[0].language = "english"  # never translated, flag cleared by mistake
    _write_approved(items)

    with pytest.raises(ItemBankError, match="does not match"):
        load_approved("rwanda", "P3")


def test_duplicate_item_ids_are_refused():
    items = _approved_items()
    items[2].id = items[0].id
    _write_approved(items)

    with pytest.raises(ItemBankError, match="duplicate item ids"):
        load_approved("rwanda", "P3")


def test_wrong_grade_band_is_refused():
    items = _approved_items()
    items[1].grade_band = "P5"
    _write_approved(items)

    with pytest.raises(ItemBankError, match="another grade band"):
        load_approved("rwanda", "P3")


def test_wrong_locale_is_refused():
    items = _approved_items()
    items[0].locale = "kenya"
    _write_approved(items)

    with pytest.raises(ItemBankError, match="another locale"):
        load_approved("rwanda", "P3")


def test_wrong_item_count_is_refused_when_config_supplied(locale_config):
    _write_approved(_approved_items(count=ITEMS_PER_ASSESSMENT - 1))

    with pytest.raises(ItemBankError, match="items_per_assessment"):
        load_approved("rwanda", "P3", locale_config=locale_config)


def test_item_count_not_checked_without_config():
    # Without a locale config there is nothing to check the count against; the other guards
    # still apply.
    _write_approved(_approved_items(count=ITEMS_PER_ASSESSMENT - 1))
    assert len(load_approved("rwanda", "P3")) == ITEMS_PER_ASSESSMENT - 1


def test_wrong_point_budget_is_refused(locale_config):
    items = _approved_items()
    items[0].point_budget = 20
    _write_approved(items)

    with pytest.raises(ItemBankError, match="point_budget"):
        load_approved("rwanda", "P3", locale_config=locale_config)


def test_empty_approved_file_is_refused():
    _write_approved([])
    with pytest.raises(ItemBankError, match="no items"):
        load_approved("rwanda", "P3")
