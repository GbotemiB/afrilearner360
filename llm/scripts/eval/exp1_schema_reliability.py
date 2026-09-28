#!/usr/bin/env python3
"""Experiment 1: how often does the free-tier model return schema-valid assessment items?

The pipeline asks OpenRouter for server-side JSON-schema enforcement and then validates the reply
against the same pydantic model anyway (generator.py:_parse_response). This measures the gap
between those two things over repeated calls: how often the reply parses as JSON at all, how
often it satisfies the schema, and when it fails, how.

Nothing about the request is re-invented here. SYSTEM_PROMPT, build_user_prompt,
ItemGenerationResponse and the engine's own _parse_response are imported from the package, so a
failure counted below is a failure the shipping pipeline would also have hit. The one thing this
script adds that the engine does not do is RETRY -- generate_items raises on the first bad reply
and leaves the decision to its caller, so attempts-to-success is measured here rather than read
off the engine.

Quota discipline (free tier: 20 requests/minute, 50 requests/day on an account with no credits):
  - a fixed pause between every request
  - 429 handled with exponential backoff, and NOT charged against the per-task content attempts
  - a hard global request budget; when it is reached the run stops and reports the N it reached
  - every attempt is checkpointed to disk as it completes, so an interrupted run still has data

Run:
    .venv/bin/python scripts/eval/exp1_schema_reliability.py
    .venv/bin/python scripts/eval/exp1_schema_reliability.py --model z-ai/glm-5.2:free --n 12
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
LLM_ROOT = HERE.parents[1]
SRC = LLM_ROOT / "src"
RESULTS = HERE / "results"
RAW_DIR = RESULTS / "raw"

sys.path.insert(0, str(SRC))

from openai import OpenAI  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from afrilearner360_llm.common.config import (  # noqa: E402
    DEFAULT_MODEL,
    OPENROUTER_BASE_URL,
    require_api_key,
)
from afrilearner360_llm.common.locale_config import (  # noqa: E402
    load_locale,
    delivery_language_for_grade_band,
)
from afrilearner360_llm.item_generation.generator import _parse_response  # noqa: E402
from afrilearner360_llm.item_generation.prompts import SYSTEM_PROMPT, build_user_prompt  # noqa: E402
from afrilearner360_llm.item_generation.schema import ItemGenerationResponse  # noqa: E402

GRADE_BANDS = ["P1", "P2", "P3", "P4", "P5"]

# The engine takes topic as free text and the locale config carries no curriculum list, so this
# grid is defined here rather than imported. Six topics across maths, science and general school
# life, crossed with the five grade bands, gives exactly 30 distinct calls -- the point being that
# N=30 is 30 different requests, not one request repeated 30 times.
TOPICS = [
    "counting and numbers",
    "fractions",
    "shapes and patterns",
    "the water cycle",
    "plants and animals",
    "everyday learning at school",
]

REQUEST_PAUSE_S = 4.0        # 20 requests/minute cap -> 3.0s floor; 4.0 leaves headroom
MAX_CONTENT_ATTEMPTS = 2     # retries on a bad reply; kept at 2 so retries cannot eat the quota
RATE_LIMIT_BACKOFF_S = [10, 30, 60]
# Free endpoints sit behind a provider-shared pool that 429s independently of this account's
# quota, so a task can be unservable for reasons that have nothing to do with the model's
# schema behaviour. One such task is skipped and recorded; only a sustained run of them ends
# the experiment, because a partial N is a usable result and a crashed run is not.
MAX_CONSECUTIVE_BLOCKED_TASKS = 3
DEFAULT_MAX_REQUESTS = 45    # hard stop below the 50/day cap
DEFAULT_NUM_ITEMS = 3        # per call; matches paper/scripts/run_model_comparison.py


class QuotaExhausted(Exception):
    """Raised when the global request budget or the daily cap is reached; ends the run cleanly."""


def build_tasks(n: int, seed: int) -> List[Dict[str, str]]:
    """Grade x topic combinations, shuffled deterministically then truncated to n.

    Shuffled rather than taken in order so that a run cut short by the quota still covers a spread
    of grades instead of stopping after P1 and P2.
    """
    grid = [
        {"grade_band": grade, "topic": topic} for grade in GRADE_BANDS for topic in TOPICS
    ]
    random.Random(seed).shuffle(grid)
    if n <= len(grid):
        return grid[:n]
    repeats = (n // len(grid)) + 1
    return (grid * repeats)[:n]


def classify_error_list(errors: List[Dict[str, Any]]) -> str:
    """Bucket a pydantic failure into the paper's taxonomy.

    Takes plain dicts rather than a ValidationError so the same rules can be re-applied to
    records already written to disk -- the taxonomy can be revised without spending API calls to
    regenerate the data.

    One reply can violate several rules at once, so buckets are applied in priority order and the
    reply is counted once, under its most structural failure. The full error list stays in the
    per-attempt record, so any other grouping can be recomputed from raw.json.
    """
    types = {err["type"] for err in errors}
    messages = " | ".join(str(err.get("msg", "")) for err in errors)

    # A failure at loc [] means the top-level payload was not the wrapper object at all (the
    # model emitted a bare array of items). That is a different defect from a field inside an
    # otherwise well-shaped response, so it gets its own bucket rather than being filed as a
    # generic type error.
    if any(err["type"] == "model_type" and not err["loc"] for err in errors):
        return "wrong_root_container"
    if "expected exactly 4 options" in messages:
        return "wrong_option_count"
    if "one option per trait" in messages:
        return "trait_set_mismatch"
    if "missing" in types:
        return "missing_required_field"
    if any(t.startswith("enum") for t in types):
        return "invalid_enum_value"
    if any("_type" in t or "_parsing" in t for t in types):
        return "wrong_type"
    return "other_validation_error"


def classify_validation_error(exc: ValidationError) -> str:
    return classify_error_list(
        [{"type": e["type"], "loc": [str(p) for p in e["loc"]], "msg": str(e["msg"])}
         for e in exc.errors()]
    )


def classify_json_error(text: str, finish_reason: Optional[str]) -> str:
    """Separate a reply that ran out of tokens from one that was simply not JSON."""
    if finish_reason == "length":
        return "truncated_output"
    stripped = text.strip()
    if stripped and not stripped.endswith(("}", "]", "`")):
        return "truncated_output"
    return "not_json_other"


class Runner:
    def __init__(
        self,
        *,
        model: str,
        num_items: int,
        max_requests: int,
        locale: str,
        time_budget_s: Optional[float] = None,
    ) -> None:
        self.model = model
        self.time_budget_s = time_budget_s
        self.started_at = time.monotonic()
        self.num_items = num_items
        self.max_requests = max_requests
        self.locale_config = load_locale(locale)
        self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=require_api_key())
        self.schema = ItemGenerationResponse.model_json_schema()
        self.requests_made = 0
        self.records: List[Dict[str, Any]] = []
        self.rate_limit_headers: Dict[str, str] = {}
        self._last_request_at: Optional[float] = None

    def _pause(self) -> None:
        if self._last_request_at is None:
            return
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < REQUEST_PAUSE_S:
            time.sleep(REQUEST_PAUSE_S - elapsed)

    def call(self, *, topic: str, grade_band: str) -> Dict[str, Any]:
        """One HTTP request. Returns a record; never raises except QuotaExhausted."""
        if self.requests_made >= self.max_requests:
            raise QuotaExhausted(f"global request budget of {self.max_requests} reached")
        # Checked before spending a request rather than after, so the budget is never overrun by
        # one call -- on a 90s-latency model that would be a meaningful overshoot.
        if self.time_budget_s is not None:
            elapsed = time.monotonic() - self.started_at
            if elapsed > self.time_budget_s:
                raise QuotaExhausted(
                    f"wall-clock budget of {self.time_budget_s:.0f}s reached after {elapsed:.0f}s"
                )

        delivery_language = delivery_language_for_grade_band(self.locale_config, grade_band)
        user_prompt = build_user_prompt(
            topic=topic,
            grade_band=grade_band,
            locale=self.locale_config.locale,
            generation_language=self.locale_config.generation_language,
            delivery_language=delivery_language,
            point_budget=self.locale_config.point_budget,
            num_items=self.num_items,
            knowledge_base_excerpt=self.locale_config.knowledge_base_text,
        )

        record: Dict[str, Any] = {
            "model": self.model,
            "topic": topic,
            "grade_band": grade_band,
            "num_items_requested": self.num_items,
            "requested_at": datetime.now(timezone.utc).isoformat(),
        }

        self._pause()
        started = time.monotonic()
        self.requests_made += 1
        try:
            raw = self.client.chat.completions.with_raw_response.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "assessment_item_generation_response",
                        "schema": self.schema,
                        "strict": True,
                    },
                },
                timeout=120.0,
            )
        except Exception as exc:  # noqa: BLE001 -- an API failure is a recorded outcome
            self._last_request_at = time.monotonic()
            status = getattr(exc, "status_code", None)
            record.update(
                {
                    "latency_s": round(time.monotonic() - started, 3),
                    "http_status": status,
                    "api_ok": False,
                    "json_ok": False,
                    "schema_ok": False,
                    "raw_text": None,
                    "finish_reason": None,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "failure_category": "rate_limited" if status == 429 else "api_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:2000],
                }
            )
            return record

        self._last_request_at = time.monotonic()
        record["latency_s"] = round(time.monotonic() - started, 3)
        record["http_status"] = raw.status_code
        for header in ("x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset"):
            if header in raw.headers:
                self.rate_limit_headers[header] = raw.headers[header]

        completion = raw.parse()
        usage = getattr(completion, "usage", None)
        record["prompt_tokens"] = getattr(usage, "prompt_tokens", None)
        record["completion_tokens"] = getattr(usage, "completion_tokens", None)

        choice = completion.choices[0] if completion.choices else None
        record["finish_reason"] = getattr(choice, "finish_reason", None) if choice else None
        text = (getattr(choice.message, "content", None) or "") if choice else ""
        record["raw_text"] = text
        record["api_ok"] = True

        if not text.strip():
            record.update(
                {
                    "json_ok": False,
                    "schema_ok": False,
                    "failure_category": "empty_content",
                    "error_type": None,
                    "error": "model returned empty content",
                }
            )
            return record

        # The engine's own parse path: fence-stripping, json.loads, then pydantic validation.
        # Which exception comes out separates "not JSON" from "JSON but wrong shape".
        try:
            parsed = _parse_response(text)
        except json.JSONDecodeError as exc:
            record.update(
                {
                    "json_ok": False,
                    "schema_ok": False,
                    "failure_category": classify_json_error(text, record["finish_reason"]),
                    "error_type": "JSONDecodeError",
                    "error": str(exc)[:2000],
                }
            )
        except ValidationError as exc:
            record.update(
                {
                    "json_ok": True,
                    "schema_ok": False,
                    "failure_category": classify_validation_error(exc),
                    "error_type": "ValidationError",
                    "error": str(exc)[:2000],
                    "validation_errors": [
                        {"type": e["type"], "loc": [str(p) for p in e["loc"]], "msg": str(e["msg"])}
                        for e in exc.errors()
                    ],
                }
            )
        except Exception as exc:  # noqa: BLE001
            record.update(
                {
                    "json_ok": False,
                    "schema_ok": False,
                    "failure_category": "other",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:2000],
                }
            )
        else:
            record.update(
                {
                    "json_ok": True,
                    "schema_ok": True,
                    "failure_category": None,
                    "error_type": None,
                    "error": None,
                    "n_items_returned": len(parsed.items),
                }
            )
        return record

    def run_task(self, task: Dict[str, str], task_index: int) -> List[Dict[str, Any]]:
        """Attempt one grade/topic until it validates, the attempts run out, or the quota does."""
        attempts: List[Dict[str, Any]] = []
        content_attempt = 0
        backoff_index = 0

        while content_attempt < MAX_CONTENT_ATTEMPTS:
            record = self.call(topic=task["topic"], grade_band=task["grade_band"])
            record["task_index"] = task_index
            record["request_index"] = self.requests_made

            if record["failure_category"] == "rate_limited":
                # Not the model's fault and not a content attempt -- back off and try again.
                record["content_attempt"] = None
                attempts.append(record)
                self.records.append(record)
                if backoff_index >= len(RATE_LIMIT_BACKOFF_S):
                    record["blocked"] = True
                    return attempts
                wait = RATE_LIMIT_BACKOFF_S[backoff_index]
                backoff_index += 1
                print(f"    429 -- backing off {wait}s", flush=True)
                time.sleep(wait)
                continue

            content_attempt += 1
            record["content_attempt"] = content_attempt
            attempts.append(record)
            self.records.append(record)
            if record["schema_ok"]:
                break

        return attempts


def summarise(records: List[Dict[str, Any]], *, model: str) -> Dict[str, Any]:
    """Compute the reported metrics from the per-attempt records."""
    content = [r for r in records if r["content_attempt"] is not None]
    rate_limited = [r for r in records if r["failure_category"] == "rate_limited"]
    first_attempts = [r for r in content if r["content_attempt"] == 1]

    n_content = len(content)
    json_ok = sum(1 for r in content if r["json_ok"])
    schema_ok = sum(1 for r in content if r["schema_ok"])

    # Attempts-to-success is defined only over tasks that eventually succeeded; a task that never
    # validated has no attempt count, and folding it in as MAX_CONTENT_ATTEMPTS would understate
    # the cost of failure rather than report it.
    by_task: Dict[int, List[Dict[str, Any]]] = {}
    for record in content:
        by_task.setdefault(record["task_index"], []).append(record)
    attempts_to_success = [
        min(r["content_attempt"] for r in group if r["schema_ok"])
        for group in by_task.values()
        if any(r["schema_ok"] for r in group)
    ]

    latencies = sorted(r["latency_s"] for r in content if r["latency_s"] is not None)
    prompt_tokens = [r["prompt_tokens"] for r in content if r["prompt_tokens"]]
    completion_tokens = [r["completion_tokens"] for r in content if r["completion_tokens"]]

    def pct(values: List[float], q: float) -> Optional[float]:
        if not values:
            return None
        # Nearest-rank on a sorted list: no interpolation, so the reported p95 is a latency that
        # was actually observed rather than a synthetic value between two samples.
        index = min(len(values) - 1, max(0, int(round(q * len(values) + 0.5)) - 1))
        return round(values[index], 3)

    taxonomy = Counter(
        r["failure_category"] for r in content if r["failure_category"] is not None
    )

    return {
        "model": model,
        "n_tasks_attempted": len(by_task),
        "n_tasks_succeeded": len(attempts_to_success),
        "n_content_requests": n_content,
        "n_rate_limited_requests": len(rate_limited),
        "n_tasks_blocked_upstream": len({r["task_index"] for r in records if r.get("blocked")}),
        "n_total_requests": len(records),
        "json_parse_rate": round(json_ok / n_content, 4) if n_content else None,
        "schema_adherence_rate": round(schema_ok / n_content, 4) if n_content else None,
        "first_attempt_success_rate": (
            round(sum(1 for r in first_attempts if r["schema_ok"]) / len(first_attempts), 4)
            if first_attempts
            else None
        ),
        "mean_attempts_to_success": (
            round(statistics.mean(attempts_to_success), 3) if attempts_to_success else None
        ),
        "latency_p50_s": pct(latencies, 0.50),
        "latency_p95_s": pct(latencies, 0.95),
        "latency_mean_s": round(statistics.mean(latencies), 3) if latencies else None,
        "mean_prompt_tokens": round(statistics.mean(prompt_tokens), 1) if prompt_tokens else None,
        "mean_completion_tokens": (
            round(statistics.mean(completion_tokens), 1) if completion_tokens else None
        ),
        "failure_taxonomy": dict(sorted(taxonomy.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


def slugify(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--n", type=int, default=30, help="number of grade/topic tasks")
    parser.add_argument("--num-items", type=int, default=DEFAULT_NUM_ITEMS)
    parser.add_argument("--max-requests", type=int, default=DEFAULT_MAX_REQUESTS)
    parser.add_argument("--locale", default="rwanda")
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument(
        "--time-budget-s",
        type=float,
        default=None,
        help="stop and report the N reached once this much wall clock has passed",
    )
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    slug = slugify(args.model)
    out_path = RESULTS / f"exp1_{slug}.json"

    tasks = build_tasks(args.n, args.seed)
    runner = Runner(
        model=args.model,
        num_items=args.num_items,
        max_requests=args.max_requests,
        locale=args.locale,
        time_budget_s=args.time_budget_s,
    )

    started = datetime.now(timezone.utc)
    stopped_early: Optional[str] = None
    print(f"model      {args.model}")
    print(f"tasks      {len(tasks)} ({args.num_items} items per call)")
    print(f"budget     {args.max_requests} requests, {REQUEST_PAUSE_S}s pause\n")

    def write_report() -> None:
        report = {
            "experiment": "exp1_structured_output_reliability",
            "generated_at": started.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "locale": args.locale,
            "num_items_per_call": args.num_items,
            "n_tasks_planned": len(tasks),
            "max_content_attempts": MAX_CONTENT_ATTEMPTS,
            "request_pause_s": REQUEST_PAUSE_S,
            "max_requests": args.max_requests,
            "time_budget_s": args.time_budget_s,
            "task_seed": args.seed,
            "stopped_early": stopped_early,
            "rate_limit_headers_last_seen": runner.rate_limit_headers,
            "summary": summarise(runner.records, model=args.model),
            "attempts": runner.records,
        }
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    try:
        consecutive_blocked = 0
        for index, task in enumerate(tasks):
            attempts = runner.run_task(task, index)
            final = attempts[-1] if attempts else None
            blocked = bool(final and final.get("blocked"))

            if blocked:
                consecutive_blocked += 1
                mark = "BLKD"
            else:
                consecutive_blocked = 0
                mark = "ok  " if final and final["schema_ok"] else "FAIL"

            detail = "" if (final and final["schema_ok"]) else f"  {final['failure_category'] if final else 'no attempt'}"
            print(
                f"[{index + 1:>2}/{len(tasks)}] {mark} {task['grade_band']:<3} "
                f"{task['topic']:<30} {final['latency_s'] if final else '-':>7}s{detail}",
                flush=True,
            )
            write_report()  # checkpoint after every task

            if consecutive_blocked >= MAX_CONSECUTIVE_BLOCKED_TASKS:
                raise QuotaExhausted(
                    f"{consecutive_blocked} consecutive tasks blocked upstream after full backoff"
                )
    except QuotaExhausted as exc:
        stopped_early = str(exc)
        print(f"\nSTOPPED EARLY: {exc}")
    except KeyboardInterrupt:
        stopped_early = "interrupted by user"
        print("\nINTERRUPTED -- writing what was collected")

    write_report()

    # Raw bodies go to their own files as well as into the report: they are the primary evidence
    # for the failure taxonomy and are far easier to read one per file than nested in JSON.
    for record in runner.records:
        if record.get("raw_text"):
            name = f"{slug}__task{record['task_index']:02d}__req{record['request_index']:02d}.txt"
            (RAW_DIR / name).write_text(record["raw_text"], encoding="utf-8")

    summary = summarise(runner.records, model=args.model)
    print("\n--- summary ---")
    for key, value in summary.items():
        if key == "failure_taxonomy":
            continue
        print(f"{key:<30} {value}")
    print("failure_taxonomy")
    for category, count in summary["failure_taxonomy"].items():
        print(f"  {category:<28} {count}")
    if runner.rate_limit_headers:
        print(f"rate limit headers: {runner.rate_limit_headers}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
