#!/usr/bin/env python3
"""Collate both experiments into results/raw.json and results/summary.md.

Reads only what the two experiment scripts already wrote to disk -- no API calls, no re-scoring --
so the tables can be regenerated (and the failure taxonomy revised) without spending quota or
depending on a model still being reachable.

Run:
    .venv/bin/python scripts/eval/make_summary.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
sys.path.insert(0, str(HERE))

from exp1_schema_reliability import classify_error_list  # noqa: E402

EXP1_FILES = sorted(RESULTS.glob("exp1_*.json"))
EXP2_FILE = RESULTS / "exp2_determinism.json"


def fmt(value: Any, spec: str = "") -> str:
    if value is None:
        return "n/a"
    if spec:
        return format(value, spec)
    return str(value)


def reclassify(report: Dict[str, Any]) -> Dict[str, int]:
    """Recompute the failure taxonomy from stored per-error detail, current rules."""
    counter: Counter = Counter()
    for attempt in report["attempts"]:
        if attempt["content_attempt"] is None or attempt["schema_ok"]:
            continue
        if attempt.get("validation_errors"):
            counter[classify_error_list(attempt["validation_errors"])] += 1
        else:
            counter[attempt["failure_category"] or "other"] += 1
    return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))


def load_exp1() -> List[Dict[str, Any]]:
    reports = []
    for path in EXP1_FILES:
        report = json.loads(path.read_text(encoding="utf-8"))
        # Runs that never got a usable reply (the model was down upstream) are recorded as
        # anomalies rather than as a zero-adherence row: they measure provider availability, not
        # the model's schema behaviour, and a table row would imply the latter was tested.
        report["_taxonomy"] = reclassify(report)
        report["_measured"] = report["summary"]["n_content_requests"] > 0
        reports.append(report)
    return reports


def exp1_table(reports: List[Dict[str, Any]]) -> str:
    header = (
        "| Model | JSON schema mode | Tasks (N) | Requests | `json_parse_rate` | "
        "`schema_adherence_rate` | `first_attempt_success_rate` | Mean attempts to success | "
        "Latency p50 (s) | Latency p95 (s) | Mean prompt tok | Mean completion tok |"
    )
    rule = "|" + "---|" * 12
    rows = []
    for report in reports:
        if not report["_measured"]:
            continue
        s = report["summary"]
        rows.append(
            "| `{model}` | {mode} | {n} | {req} | {jp} | {sa} | {fa} | {ats} | {p50} | {p95} | "
            "{pt} | {ct} |".format(
                model=s["model"],
                mode=MODE_TAGS.get(s["model"], "unknown"),
                n=s["n_tasks_attempted"],
                req=s["n_content_requests"],
                jp=fmt(s["json_parse_rate"], ".2f"),
                sa=fmt(s["schema_adherence_rate"], ".2f"),
                fa=fmt(s["first_attempt_success_rate"], ".2f"),
                ats=fmt(s["mean_attempts_to_success"]),
                p50=fmt(s["latency_p50_s"], ".1f"),
                p95=fmt(s["latency_p95_s"], ".1f"),
                pt=fmt(s["mean_prompt_tokens"], ".0f"),
                ct=fmt(s["mean_completion_tokens"], ".0f"),
            )
        )
    return "\n".join([header, rule] + rows)


def taxonomy_table(reports: List[Dict[str, Any]]) -> str:
    measured = [r for r in reports if r["_measured"]]
    categories = sorted({c for r in measured for c in r["_taxonomy"]})
    if not categories:
        return "_No schema failures recorded._"
    header = "| Failure mode | " + " | ".join(f"`{r['summary']['model']}`" for r in measured) + " |"
    rule = "|" + "---|" * (len(measured) + 1)
    rows = []
    for category in categories:
        cells = " | ".join(str(r["_taxonomy"].get(category, 0)) for r in measured)
        rows.append(f"| `{category}` | {cells} |")
    totals = " | ".join(str(sum(r["_taxonomy"].values())) for r in measured)
    rows.append(f"| **Total failed requests** | {totals} |")
    return "\n".join([header, rule] + rows)


def exp2_table(report: Dict[str, Any]) -> str:
    within = report["within_process"]
    across = report["across_processes"]
    return "\n".join(
        [
            "| Check | Fixtures | Scorings | `exact_match_rate` (scored profile) | "
            "`exact_match_rate` (incl. error text) |",
            "|---|---|---|---|---|",
            "| Within one process ({r} repeats) | {n} | {t} | {a:.4f} | {b:.4f} |".format(
                r=within["repeats_per_fixture"],
                n=within["fixtures"],
                t=within["total_scorings"],
                a=within["exact_match_rate_result"],
                b=within["exact_match_rate_message"],
            ),
            "| Across processes (`PYTHONHASHSEED` {seeds}) | {n} | {t} | {a:.4f} | {b:.4f} |".format(
                seeds="/".join(across["hash_seeds"]),
                n=across["fixtures"],
                t=across["fixtures"] * (across["processes"] + 1),
                a=across["exact_match_rate_result"],
                b=across["exact_match_rate_message"],
            ),
        ]
    )


MODE_TAGS = {
    "nvidia/nemotron-3-super-120b-a12b:free": "`structured_outputs`",
    "minimax/minimax-m3:free": "`response_format`",
    "google/gemma-4-26b-a4b-it:free": "`response_format`",
    "google/gemma-4-31b-it:free": "`response_format`",
    "z-ai/glm-5.2:free": "`structured_outputs`",
    # Added for the camera-ready sweep (2026-09-27). Capability as declared in that day's
    # catalogue snapshot, results/catalogue/catalogue_2026-09-27.json.
    "dots-studio/dots-3-note-preview:free": "`structured_outputs`",
    "liquid/lfm-2.5-2.6b:free": "`structured_outputs`",
    "qwen/qwen3.8-27b:free": "`structured_outputs`",
}


def main() -> None:
    exp1 = load_exp1()
    exp2 = json.loads(EXP2_FILE.read_text(encoding="utf-8"))
    measured = [r for r in exp1 if r["_measured"]]
    unavailable = [r for r in exp1 if not r["_measured"]]

    raw = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_1": [
            {
                "model": r["model"],
                "run_metadata": {k: v for k, v in r.items() if k not in ("attempts", "_taxonomy")},
                "failure_taxonomy": r["_taxonomy"],
                "attempts": r["attempts"],
            }
            for r in exp1
        ],
        "experiment_2": exp2,
    }
    (RESULTS / "raw.json").write_text(
        json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    date_run = exp2["generated_at"][:10]
    lines: List[str] = []
    lines.append("# Evaluation results")
    lines.append("")
    lines.append(f"Date run: **{date_run}** (UTC). Locale: `rwanda`. "
                 f"All raw per-attempt records in `raw.json`; raw model replies in `raw/`.")
    lines.append("")
    lines.append("Both experiments call the shipping code paths rather than reimplementations: "
                 "Experiment 1 imports `SYSTEM_PROMPT`, `build_user_prompt`, "
                 "`ItemGenerationResponse` and `generator._parse_response`; Experiment 2 imports "
                 "`profiling.scorer.score_student`. No file under `src/` was modified.")
    lines.append("")

    lines.append("## Experiment 1 -- structured-output reliability")
    lines.append("")
    lines.append(f"Each task is one (grade band, topic) pair requesting "
                 f"{measured[0]['num_items_per_call'] if measured else 'n/a'} items; tasks are "
                 f"drawn from a 5 grade band x 6 topic grid so N distinct requests are made, not "
                 f"one request repeated N times. The engine itself does not retry, so retries "
                 f"(max {measured[0]['max_content_attempts'] if measured else 'n/a'} per task) "
                 f"are added by the eval harness to make attempts-to-success measurable.")
    lines.append("")
    lines.append(exp1_table(exp1))
    lines.append("")
    lines.append("### Failure taxonomy (count of failing requests)")
    lines.append("")
    lines.append(taxonomy_table(exp1))
    lines.append("")

    lines.append("## Experiment 2 -- determinism of the scoring layer")
    lines.append("")
    lines.append(f"{exp2['n_fixtures']} hand-built fixtures "
                 f"({exp2['n_ok_fixtures']} score to a profile, {exp2['n_raising_fixtures']} are "
                 f"expected to raise), covering single-trait dominance, an exact four-way tie "
                 f"reached two different ways, two- and three-way ties, both classifier "
                 f"thresholds at their exact boundary, minimum and maximum response-set lengths, "
                 f"and every input the scorer rejects. Outputs are canonicalised "
                 f"(sorted keys, `{exp2['float_format']}` floats) and SHA-256 hashed.")
    lines.append("")
    lines.append(exp2_table(exp2))
    lines.append("")

    lines.append("## Findings")
    lines.append("")
    lines.append(FINDINGS.strip())
    lines.append("")
    lines.append("## Anomalies and failures")
    lines.append("")
    lines.append(ANOMALIES.strip())
    if unavailable:
        lines.append("")
        for report in unavailable:
            lines.append(
                f"- `{report['model']}`: no content reply obtained; "
                f"{report['summary']['n_rate_limited_requests']} request(s) all returned HTTP 429 "
                f"from the provider-shared pool. Not tabulated above, since this measures provider "
                f"availability rather than the model's schema behaviour."
            )
    lines.append("")

    (RESULTS / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {RESULTS / 'summary.md'}")
    print(f"wrote {RESULTS / 'raw.json'}")


FINDINGS = """
Two free-tier models on OpenRouter were evaluated against the item-generation schema on
2026-09-06, and the deterministic scorer was evaluated offline the same day. Every reply from
both models parsed as JSON (`json_parse_rate` = 1.00 in both cases), but schema adherence
separated them completely: `nvidia/nemotron-3-super-120b-a12b:free`, which advertises
server-side `structured_outputs`, validated on 10 of 10 first attempts, while
`minimax/minimax-m3:free`, which offers only `response_format` JSON mode, validated on 0 of 20
requests across 10 tasks and 2 attempts each. MiniMax M3's failures were concentrated in omitted
required fields (most often `scenario_text` and each option's `trait`) and, in 5 of 20 requests,
emission of a bare JSON array in place of the `{"items": [...]}` root object; 17 of its 20 replies
were additionally wrapped in a markdown code fence despite the request specifying a strict JSON
schema, so the pipeline's fence-stripping fallback was load-bearing rather than defensive.
Schema adherence carried a latency cost: the compliant model was roughly 2.7x slower at the
median (40.7s versus 15.6s) and emitted roughly 2.4x more completion tokens per call. The
scoring layer produced byte-identical output for identical input in every condition tested --
250 within-process scorings and 25 fixtures re-scored across three additional interpreters under
different `PYTHONHASHSEED` values all agreed on hash -- with one exception affecting error
message text only, described below.
"""

ANOMALIES = """
- **Non-determinism found (error message text only).** Fixture `err_missing_trait_key` produced a
  different SHA-256 in all four processes (4 distinct message hashes from 4 processes). The
  scored profile is unaffected and `exact_match_rate` over scored output remains 1.0000. Cause:
  `ItemResponse.all_traits_present_and_non_negative` in `profiling/scorer.py` interpolates two raw
  `set`s of `Trait` enum members into its error message; `Enum.__hash__` hashes the member name,
  so string-hash randomisation reorders the set repr per process. Reported, not fixed, per the
  task brief. It is a cosmetic defect in a message, but it would make any test asserting on that
  message string flaky, and it would break byte-comparison of logs across runs.
- **The engine's configured default model could not be measured.** `AFRILEARNER_LLM_MODEL` is
  `google/gemma-4-26b-a4b-it:free`, which returned HTTP 429 on every attempt with
  `limit_source: upstream_provider_shared_pool` -- a Google AI Studio shared-pool limit,
  independent of this account's quota (`usage: 0`, daily cap untouched). `google/gemma-4-31b-it:free`
  and `z-ai/glm-5.2:free` were unavailable for the same reason. The two models tabulated above are
  substitutes, so the table does not describe the configuration the project currently ships.
- **N is below the 30 originally planned.** Experiment 1 ran 10 tasks per model. Nemotron's
  median latency of 40.7s (p95 112.8s) made N=30 infeasible inside the time budget, and MiniMax M3
  consumed 2 requests per task on retries against a 50-requests/day free-tier cap. Both runs
  stopped on their configured request budget, not on an error.
- **Zero-success rows carry no attempts-to-success figure.** `mean_attempts_to_success` is
  computed only over tasks that eventually validated, so it is `n/a` for MiniMax M3 rather than
  silently reported as the retry ceiling.
- **The scorer accepts none of the empty or missing-response cases** the task brief anticipated:
  empty response lists, all-zero allocations, duplicate items, unknown items, budget mismatches
  and missing trait keys all raise. These are recorded as expected-raise fixtures and their error
  paths are hashed alongside the scored ones, which is how the defect above surfaced.
"""

if __name__ == "__main__":
    main()
