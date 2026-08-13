#!/usr/bin/env python3
"""CLI: generate draft assessment items for one topic/grade band and save them to the item bank.

Usage:
    uv run python scripts/generate_items.py --topic "fractions" --grade-band P3 --num-items 10

Requires OPENROUTER_API_KEY to be set (see .env.example). Output is written to
data/item_banks/<locale>/drafts/ as DRAFT content. Drafts are never served to students: a human
must review them and supply translations where needed, then save the result to approved/. See
README for that gate.

Free-tier models are capped at 20 requests/minute and 50 requests/day, and the cap is on requests
rather than tokens -- so prefer one call with a large --num-items over many small calls.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from afrilearner360_llm.common.locale_config import load_locale
from afrilearner360_llm.item_generation.generator import generate_items
from afrilearner360_llm.item_generation.item_bank import save_drafts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locale", default="rwanda")
    parser.add_argument("--topic", required=True, help='e.g. "fractions", "the water cycle"')
    parser.add_argument("--grade-band", required=True, help='e.g. "P3"')
    parser.add_argument(
        "--num-items",
        type=int,
        default=None,
        help="Items to generate (default: the locale's items_per_assessment)",
    )
    parser.add_argument("--model", default=None, help="Override the default model")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write to this path instead of the locale's drafts directory",
    )
    parser.add_argument(
        "--print", action="store_true", dest="print_json", help="Also print the JSON to stdout"
    )
    args = parser.parse_args()

    locale_config = load_locale(args.locale)
    num_items = args.num_items or locale_config.items_per_assessment

    kwargs = dict(
        locale_config=locale_config,
        topic=args.topic,
        grade_band=args.grade_band,
        num_items=num_items,
    )
    if args.model:
        kwargs["model"] = args.model

    result = generate_items(**kwargs)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        path = args.out
    else:
        path = save_drafts(
            result,
            locale=args.locale,
            grade_band=args.grade_band,
            topic=args.topic,
        )

    if args.print_json:
        print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))

    needing_translation = [item.id for item in result.items if item.needs_translation]
    print(f"Wrote {len(result.items)} draft item(s) to {path}")
    if needing_translation:
        print(
            f"{len(needing_translation)} item(s) need translation before approval: "
            f"{', '.join(needing_translation)}"
        )
    print(
        "These are DRAFTS. Review them, supply any translations, then save the approved set to "
        f"{Path('data/item_banks') / args.locale / 'approved' / (args.grade_band + '.json')}"
    )


if __name__ == "__main__":
    main()
