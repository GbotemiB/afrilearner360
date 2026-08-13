#!/usr/bin/env python3
"""CLI: run a class through the whole pipeline and save teaching recommendations.

Reads student responses, profiles each student, clusters the class, then asks the LLM for
per-group teaching recommendations and saves them.

Usage:
    uv run python scripts/generate_recommendations.py \\
        --responses data/responses/P5A.json --topic "fractions"

The responses file is the contract between this repo and the team's Flask app:

    {
      "class_id": "P5A",
      "grade_band": "P5",
      "students": [
        {
          "student_id": "s-001",
          "responses": [
            {"item_id": "rw-p5-everyday-learning-01",
             "allocations": {"visual": 5, "game": 3, "structured": 1, "story": 1}}
          ]
        }
      ]
    }

Requires OPENROUTER_API_KEY (see .env.example). Whole-class mode costs one request; per-cluster
mode costs one per group, against a 50-request/day free tier.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from afrilearner360_llm.clustering import fit_classroom
from afrilearner360_llm.common.locale_config import load_locale
from afrilearner360_llm.item_generation.item_bank import load_approved
from afrilearner360_llm.item_generation.schema import Trait
from afrilearner360_llm.profiling import ItemResponse, score_student
from afrilearner360_llm.recommendations import (
    RecommendationMode,
    generate_recommendations,
    save_recommendations,
)


def load_profiles(payload: dict, *, locale_config, known_item_ids):
    """Score every student in the responses file."""
    profiles = []
    for student in payload["students"]:
        responses = [
            ItemResponse(
                item_id=entry["item_id"],
                allocations={Trait(trait): points for trait, points in entry["allocations"].items()},
            )
            for entry in student["responses"]
        ]
        profiles.append(
            score_student(
                student_id=student["student_id"],
                responses=responses,
                expected_point_budget=locale_config.point_budget,
                known_item_ids=known_item_ids,
            )
        )
    return profiles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locale", default="rwanda")
    parser.add_argument("--responses", type=Path, required=True, help="Path to the responses JSON")
    parser.add_argument("--topic", required=True, help='e.g. "fractions"')
    parser.add_argument("--k", type=int, default=4, help="Number of groups (default 4)")
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in RecommendationMode],
        default=RecommendationMode.WHOLE_CLASS.value,
        help="whole_class = 1 request, groups differentiated against each other; "
        "per_cluster = 1 request per group, failure-isolated",
    )
    parser.add_argument("--model", default=None, help="Override the default model")
    parser.add_argument(
        "--skip-item-check",
        action="store_true",
        help="Do not validate response item ids against the approved assessment",
    )
    args = parser.parse_args()

    locale_config = load_locale(args.locale)
    payload = json.loads(args.responses.read_text(encoding="utf-8"))
    class_id = payload["class_id"]
    grade_band = payload["grade_band"]

    known_item_ids = None
    if not args.skip_item_check:
        known_item_ids = [item.id for item in load_approved(args.locale, grade_band, locale_config=locale_config)]

    profiles = load_profiles(
        payload, locale_config=locale_config, known_item_ids=known_item_ids
    )
    print(f"Profiled {len(profiles)} students in class {class_id} ({grade_band})")

    clustering = fit_classroom(profiles, k=args.k)
    for summary in clustering.summaries:
        print(f"  cluster {summary.cluster_id}: {summary.size} students, {summary.descriptor()}")

    kwargs = dict(
        locale_config=locale_config,
        summaries=clustering.summaries,
        class_id=class_id,
        grade_band=grade_band,
        topic=args.topic,
        mode=RecommendationMode(args.mode),
    )
    if args.model:
        kwargs["model"] = args.model

    recommendations = generate_recommendations(**kwargs)
    path = save_recommendations(recommendations)

    print(f"\nSaved recommendations to {path}")
    print(f"\n{recommendations.class_overview}\n")
    for rec in recommendations.cluster_recommendations:
        print(f"[{rec.group_label}]")
        for strategy in rec.teaching_strategies:
            print(f"  - {strategy}")
        print(f"  Activity: {rec.classroom_activity}")
        print(f"  Materials: {', '.join(rec.materials_needed)}")
        if rec.watch_out_for:
            print(f"  Watch: {rec.watch_out_for}")
        print()


if __name__ == "__main__":
    main()
