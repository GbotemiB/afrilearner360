"""Calls the LLM to produce per-cluster teaching recommendations.

Two modes (see RecommendationMode): one call for the whole class, or one call per cluster. The
mode changes only how many calls are made and how the class-level advice is produced -- both
return the same `ClassroomRecommendations` artifact, so callers downstream do not branch.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import List, Optional, Sequence, Type

from openai import OpenAI
from pydantic import BaseModel

from ..clustering.cluster import ClusterSummary
from ..common.config import DEFAULT_MODEL, OPENROUTER_BASE_URL, require_api_key
from ..common.locale_config import LocaleConfig
from .prompts import SYSTEM_PROMPT, build_single_cluster_prompt, build_whole_class_prompt
from .schema import (
    ClassroomRecommendations,
    ClusterRecommendation,
    RecommendationDraft,
    RecommendationMode,
    SingleClusterDraft,
)

logger = logging.getLogger(__name__)


def _client() -> OpenAI:
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=require_api_key())


def generate_recommendations(
    *,
    locale_config: LocaleConfig,
    summaries: Sequence[ClusterSummary],
    class_id: str,
    grade_band: str,
    topic: str,
    mode: RecommendationMode = RecommendationMode.WHOLE_CLASS,
    model: str = DEFAULT_MODEL,
    generated_at: Optional[datetime] = None,
) -> ClassroomRecommendations:
    """Generate teaching recommendations for one class and topic.

    `mode` trades cost against isolation. WHOLE_CLASS is one request regardless of k, and lets the
    model differentiate groups against each other -- but one malformed response loses the lot.
    PER_CLUSTER is k requests and survives a single bad response, at the cost of each group being
    planned without sight of the others.

    Raises ValueError if `summaries` is empty, and pydantic.ValidationError if the model's output
    does not match the schema or fails to cover every cluster exactly once.
    """
    if not summaries:
        raise ValueError(f"no cluster summaries supplied for class {class_id!r}")

    generated_at = generated_at or datetime.now()
    cluster_sizes = {summary.cluster_id: summary.size for summary in summaries}

    if mode is RecommendationMode.WHOLE_CLASS:
        draft = _generate_whole_class(
            locale_config=locale_config,
            summaries=summaries,
            topic=topic,
            grade_band=grade_band,
            model=model,
        )
        class_overview = draft.class_overview
        running_the_class = draft.running_the_class
        recommendations = draft.cluster_recommendations
        constraints = draft.constraints_addressed
        anchors = draft.cultural_anchors_used
        notes = draft.reviewer_notes
    else:
        recommendations, constraints, anchors, notes = _generate_per_cluster(
            locale_config=locale_config,
            summaries=summaries,
            topic=topic,
            grade_band=grade_band,
            model=model,
        )
        # In per-cluster mode no single call ever sees the whole class, so class-level advice is
        # assembled deterministically rather than attributed to a model that never had the
        # information to write it.
        class_overview = describe_class(summaries)
        running_the_class = default_running_the_class(summaries)

    return ClassroomRecommendations(
        locale=locale_config.locale,
        class_id=class_id,
        grade_band=grade_band,
        topic=topic,
        mode=mode,
        model=model,
        generated_at=generated_at,
        cluster_sizes=cluster_sizes,
        class_overview=class_overview,
        running_the_class=running_the_class,
        cluster_recommendations=recommendations,
        constraints_addressed=constraints,
        cultural_anchors_used=anchors,
        reviewer_notes=notes,
    )


def _generate_whole_class(
    *,
    locale_config: LocaleConfig,
    summaries: Sequence[ClusterSummary],
    topic: str,
    grade_band: str,
    model: str,
) -> RecommendationDraft:
    prompt = build_whole_class_prompt(
        summaries=summaries,
        topic=topic,
        grade_band=grade_band,
        locale=locale_config.locale,
        recommendation_context=locale_config.recommendation_context,
        knowledge_base_excerpt=locale_config.knowledge_base_text,
    )
    return _call_model(prompt, schema_model=RecommendationDraft, model=model, label="whole class")


def _generate_per_cluster(
    *,
    locale_config: LocaleConfig,
    summaries: Sequence[ClusterSummary],
    topic: str,
    grade_band: str,
    model: str,
) -> tuple[List[ClusterRecommendation], List[str], List[str], Optional[str]]:
    recommendations: List[ClusterRecommendation] = []
    constraints: List[str] = []
    anchors: List[str] = []
    notes: List[str] = []

    for summary in summaries:
        prompt = build_single_cluster_prompt(
            summary=summary,
            all_summaries=summaries,
            topic=topic,
            grade_band=grade_band,
            locale=locale_config.locale,
            recommendation_context=locale_config.recommendation_context,
            knowledge_base_excerpt=locale_config.knowledge_base_text,
        )
        draft = _call_model(
            prompt,
            schema_model=SingleClusterDraft,
            model=model,
            label=f"cluster {summary.cluster_id}",
        )

        # The model is told which cluster it is advising on, but nothing stops it echoing the
        # wrong id; the caller's cluster is authoritative.
        recommendation = draft.recommendation
        if recommendation.cluster_id != summary.cluster_id:
            logger.warning(
                "model returned cluster_id %s for cluster %s; correcting",
                recommendation.cluster_id,
                summary.cluster_id,
            )
            recommendation = recommendation.model_copy(update={"cluster_id": summary.cluster_id})

        recommendations.append(recommendation)
        constraints.extend(draft.constraints_addressed)
        anchors.extend(draft.cultural_anchors_used)
        if draft.reviewer_notes:
            notes.append(f"cluster {summary.cluster_id}: {draft.reviewer_notes}")

    return (
        recommendations,
        _dedupe(constraints),
        _dedupe(anchors),
        " | ".join(notes) if notes else None,
    )


def _dedupe(values: Sequence[str]) -> List[str]:
    """Order-preserving dedupe, so merged traceability lists stay readable."""
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _call_model(prompt: str, *, schema_model: Type[BaseModel], model: str, label: str) -> BaseModel:
    response = _client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema_model.__name__.lower(),
                "schema": schema_model.model_json_schema(),
                "strict": True,
            },
        },
    )
    logger.info("generated recommendations for %s using %s", label, model)
    return _parse_response(response.choices[0].message.content, schema_model=schema_model)


def _parse_response(raw_content: str, *, schema_model: Type[BaseModel]) -> BaseModel:
    """Parse the model's JSON output, tolerating a markdown code fence.

    Same fallback as item generation: not every model routed through OpenRouter honors strict
    json_schema mode identically.
    """
    text = raw_content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    return schema_model.model_validate(json.loads(text))


def describe_class(summaries: Sequence[ClusterSummary]) -> str:
    """Plain-language diversity summary, generated in code rather than by the model.

    Used in per-cluster mode, where no single call ever sees the whole class.
    """
    total = sum(summary.size for summary in summaries)
    if total == 0:
        return "No students have been assessed in this class yet."

    parts = [
        f"{summary.size} {summary.descriptor()}"
        for summary in summaries
        if summary.size > 0
    ]
    return (
        f"This class of {total} students falls into {len([s for s in summaries if s.size > 0])} "
        f"groups: {'; '.join(parts)}. Every group contains children of every ability level -- "
        f"these describe how children currently prefer to engage, not what they can do."
    )


def default_running_the_class(summaries: Sequence[ClusterSummary]) -> List[str]:
    """Baseline advice for running k groups in one room, used in per-cluster mode."""
    active = [summary for summary in summaries if summary.size > 0]
    return [
        f"Introduce the topic to the whole class first, then split into {len(active)} groups for "
        f"practice.",
        "Rotate: work directly with one group while the others do their activity independently or "
        "in pairs.",
        "Give each group its activity instructions before you start rotating, so no group is idle "
        "waiting for you.",
        "Groups are a tool for specific activities, not fixed seating -- remix them for other "
        "lessons.",
    ]
