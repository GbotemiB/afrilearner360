#!/usr/bin/env python3
"""Project wall-clock and cost for generating a full term of content, from measured figures.

Reviewers asked what the rate-limit premise actually costs in practice. This answers it from
the exp1 measurements rather than from estimates: per-call latency and token counts are read
from the stored nemotron run, and the only inputs supplied here are the curriculum shape
(grades, topics per term, representations) and the free-tier request cap.

The paid-tier figure is a counterfactual, not an expenditure. The project spent nothing; the
number exists to show how small the bill would be if the free tier vanished, which is the
deployment risk the availability section describes.

Run:
    .venv/bin/python scripts/eval/cost_projection.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
MEASURED_MODEL_FILE = "exp1_nvidia__nemotron-3-super-120b-a12b_free.json"

# OpenRouter free tier on an account that has never purchased credits.
FREE_REQUESTS_PER_DAY = 50

# Paid pricing for minimax/minimax-m3, whose :free variant was withdrawn between the evaluation
# (2026-09-06) and 2026-09-27. USD per million tokens. Used as the counterfactual because it is
# the concrete case of a model this project measured for free and would now have to pay for.
PAID_PROMPT_USD_PER_M = 0.30
PAID_COMPLETION_USD_PER_M = 1.20


def load_measured() -> Dict[str, Any]:
    path = RESULTS / MEASURED_MODEL_FILE
    report = json.loads(path.read_text(encoding="utf-8"))
    summary = report["summary"]
    return {
        "model": report["model"],
        "items_per_call": report["num_items_per_call"],
        "latency_p50_s": summary["latency_p50_s"],
        "latency_p95_s": summary["latency_p95_s"],
        "prompt_tokens": summary["mean_prompt_tokens"],
        "completion_tokens": summary["mean_completion_tokens"],
        "schema_adherence_rate": summary["schema_adherence_rate"],
    }


def project(
    measured: Dict[str, Any], *, grades: int, topics_per_term: int, representations: int
) -> Dict[str, Any]:
    """One generation call per (grade, topic, representation)."""
    calls = grades * topics_per_term * representations

    wall_clock_s = calls * measured["latency_p50_s"]
    days_on_free_tier = calls / FREE_REQUESTS_PER_DAY

    prompt_cost = calls * measured["prompt_tokens"] / 1e6 * PAID_PROMPT_USD_PER_M
    completion_cost = calls * measured["completion_tokens"] / 1e6 * PAID_COMPLETION_USD_PER_M

    return {
        "grades": grades,
        "topics_per_term": topics_per_term,
        "representations": representations,
        "calls": calls,
        "wall_clock_hours": round(wall_clock_s / 3600, 2),
        "days_at_free_tier_cap": round(days_on_free_tier, 1),
        "free_requests_per_day": FREE_REQUESTS_PER_DAY,
        "paid_cost_usd": round(prompt_cost + completion_cost, 2),
        "paid_cost_per_call_usd": round((prompt_cost + completion_cost) / calls, 5),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grades", type=int, default=5, help="P1-P5")
    parser.add_argument("--topics-per-term", type=int, default=10)
    parser.add_argument("--representations", type=int, default=4)
    args = parser.parse_args()

    measured = load_measured()
    result = project(
        measured,
        grades=args.grades,
        topics_per_term=args.topics_per_term,
        representations=args.representations,
    )

    print(f"measured from {measured['model']}")
    print(f"  median latency        {measured['latency_p50_s']} s/call")
    print(f"  mean prompt tokens    {measured['prompt_tokens']}")
    print(f"  mean completion tok   {measured['completion_tokens']}\n")
    print(f"a full term, {args.grades} grades x {args.topics_per_term} topics "
          f"x {args.representations} representations")
    print(f"  generation calls      {result['calls']}")
    print(f"  wall clock            {result['wall_clock_hours']} h of model time")
    print(f"  at {FREE_REQUESTS_PER_DAY} requests/day       {result['days_at_free_tier_cap']} days")
    print(f"  paid counterfactual   ${result['paid_cost_usd']} "
          f"(${result['paid_cost_per_call_usd']}/call)")

    out = RESULTS / "cost_projection.json"
    out.write_text(json.dumps({"measured": measured, "projection": result}, indent=2) + "\n",
                   encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
