#!/usr/bin/env python3
"""CLI demo: generate draft assessment items for one topic/grade band and print them as JSON.

Usage:
    python scripts/generate_items.py --topic "fractions" --grade-band P3 --num-items 3

Requires OPENROUTER_API_KEY to be set (see .env.example). Output is DRAFT content -- see README
for the required human review step before any item is used with real students.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from afrilearner360_llm.common.locale_config import load_locale  # noqa: E402
from afrilearner360_llm.item_generation.generator import generate_items  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locale", default="rwanda")
    parser.add_argument("--topic", required=True, help='e.g. "fractions", "the water cycle"')
    parser.add_argument("--grade-band", required=True, help='e.g. "P3"')
    parser.add_argument("--num-items", type=int, default=3)
    parser.add_argument("--model", default=None, help="Override the default model")
    args = parser.parse_args()

    locale_config = load_locale(args.locale)

    kwargs = dict(
        locale_config=locale_config,
        topic=args.topic,
        grade_band=args.grade_band,
        num_items=args.num_items,
    )
    if args.model:
        kwargs["model"] = args.model

    result = generate_items(**kwargs)
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
