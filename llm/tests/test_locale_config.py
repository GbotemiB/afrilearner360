"""Tests for locale loading and the generation-vs-delivery language split.

No API calls -- these read the real data/locales/rwanda.yaml, so they double as a check that the
shipped locale pack stays loadable.
"""
import pytest

from afrilearner360_llm.common.locale_config import (
    LocaleConfig,
    delivery_language_for_grade_band,
    load_locale,
)


@pytest.fixture
def rwanda():
    return load_locale("rwanda")


def test_rwanda_locale_loads(rwanda):
    assert rwanda.locale == "rwanda"
    assert rwanda.point_budget == 10
    assert rwanda.items_per_assessment == 10
    assert set(rwanda.trait_definitions) == {"visual", "game", "structured", "story"}
    assert rwanda.knowledge_base_text.strip()


def test_generation_language_is_english(rwanda):
    # Items are always drafted in English; Kinyarwanda comes from a human translator inside the
    # review gate. See DESIGN.md §7.
    assert rwanda.generation_language == "english"


def test_both_bands_deliver_in_english(rwanda):
    # MVP decision: lower primary switched from kinyarwanda to english, which removes the
    # translation step. Only sound under oral administration -- see DESIGN.md §4.1.
    for grade_band in ("P1", "P3", "P4", "P6"):
        assert delivery_language_for_grade_band(rwanda, grade_band) == "english"


def test_no_rwanda_band_currently_needs_translation(rwanda):
    for grade_band in ("P1", "P3", "P4", "P6"):
        assert delivery_language_for_grade_band(rwanda, grade_band) == rwanda.generation_language


def test_translation_still_triggers_for_a_locale_that_needs_it():
    # The translation machinery is locale-configurable, not deleted -- Rwanda just no longer
    # exercises it. A locale delivering in a non-generation language must still flag it.
    config = LocaleConfig(
        locale="testland",
        display_name="Testland",
        trait_definitions={},
        point_budget=10,
        items_per_assessment=10,
        generation_language="english",
        delivery_language_by_grade_band={"lower_primary": "swahili", "upper_primary": "english"},
        recommendation_context={},
        knowledge_base_text="",
    )
    assert delivery_language_for_grade_band(config, "P2") != config.generation_language
    assert delivery_language_for_grade_band(config, "P5") == config.generation_language


def test_grade_band_without_digits_raises(rwanda):
    # Guessing here would silently mislabel whether an item still owes a translation pass.
    with pytest.raises(ValueError, match="grade number"):
        delivery_language_for_grade_band(rwanda, "primary")


def test_missing_band_config_raises():
    config = LocaleConfig(
        locale="testland",
        display_name="Testland",
        trait_definitions={},
        point_budget=10,
        items_per_assessment=10,
        generation_language="english",
        delivery_language_by_grade_band={"upper_primary": "english"},
        recommendation_context={},
        knowledge_base_text="",
    )
    with pytest.raises(KeyError, match="lower_primary"):
        delivery_language_for_grade_band(config, "P2")


def test_unknown_locale_raises():
    with pytest.raises(FileNotFoundError):
        load_locale("atlantis")
