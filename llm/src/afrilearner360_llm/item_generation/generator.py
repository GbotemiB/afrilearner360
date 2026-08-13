"""Calls the LLM (via OpenRouter's OpenAI-compatible API) to draft candidate assessment items.

Generated items are DRAFTS. Nothing here writes to a "live" item bank -- see README for the
human (ideally local teacher) review step that must happen before an item is used with real
students.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from openai import OpenAI

from ..common.config import DEFAULT_MODEL, OPENROUTER_BASE_URL, require_api_key
from ..common.locale_config import LocaleConfig, delivery_language_for_grade_band
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .schema import ItemGenerationResponse

logger = logging.getLogger(__name__)


def _client() -> OpenAI:
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=require_api_key())


def generate_items(
    *,
    locale_config: LocaleConfig,
    topic: str,
    grade_band: str,
    num_items: int = 3,
    model: str = DEFAULT_MODEL,
    delivery_language: Optional[str] = None,
) -> ItemGenerationResponse:
    """Generate `num_items` draft assessment items for one topic/grade band.

    Items are always drafted in `locale_config.generation_language` (English). Where the grade
    band's delivery language differs, the returned items are flagged `needs_translation` -- a
    human translator handles that inside the review gate, because free-tier models produce
    unusable Kinyarwanda (see DESIGN.md §7).

    Raises pydantic.ValidationError if the model's output doesn't match the schema (e.g. wrong
    number of options) -- caller should treat that as "retry or escalate to a human", not silently
    swallow it, since a malformed item must never reach students.
    """
    generation_language = locale_config.generation_language
    delivery_language = delivery_language or delivery_language_for_grade_band(
        locale_config, grade_band
    )
    point_budget = locale_config.point_budget

    user_prompt = build_user_prompt(
        topic=topic,
        grade_band=grade_band,
        locale=locale_config.locale,
        generation_language=generation_language,
        delivery_language=delivery_language,
        point_budget=point_budget,
        num_items=num_items,
        knowledge_base_excerpt=locale_config.knowledge_base_text,
    )

    schema = ItemGenerationResponse.model_json_schema()

    response = _client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "assessment_item_generation_response",
                "schema": schema,
                "strict": True,
            },
        },
    )

    raw_content = response.choices[0].message.content
    parsed = _parse_response(raw_content)
    return _stamp_translation_status(parsed, delivery_language=delivery_language)


def _stamp_translation_status(
    parsed: ItemGenerationResponse, *, delivery_language: str
) -> ItemGenerationResponse:
    """Record each item's delivery language and whether a human translation pass is still owed.

    Done here rather than asked of the model: the model only ever writes English and knows nothing
    about the locale's delivery-language policy, so letting it self-report would just invite it to
    guess.
    """
    for item in parsed.items:
        item.delivery_language = delivery_language
        item.needs_translation = item.language.strip().lower() != delivery_language.strip().lower()
        if item.needs_translation:
            logger.info(
                "item %s drafted in %s, needs translation to %s before going live",
                item.id,
                item.language,
                delivery_language,
            )
    return parsed


def _parse_response(raw_content: str) -> ItemGenerationResponse:
    """Parses the model's JSON output into validated items.

    Not every model routed through OpenRouter honors `strict` json_schema mode identically, so we
    fall back to plain json.loads + pydantic validation if the content isn't already clean JSON
    (e.g. wrapped in a markdown code fence).
    """
    text = raw_content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    data = json.loads(text)
    return ItemGenerationResponse.model_validate(data)
