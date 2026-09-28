#!/usr/bin/env python3
"""Experiment 2: is the non-AI scorer byte-identical for identical input?

Two levels of check:

  within-process   every fixture scored N times in one interpreter
  across-process   the whole fixture set re-scored in separate subprocesses under different
                   PYTHONHASHSEED values

The second level is the one that matters. String hashing is randomised per process, so anything
that reaches an output through set or dict iteration order is stable within a run and unstable
between runs -- invisible to a normal test suite, and the usual cause of a tie-break that quietly
picks a different winner on a different day. The scorer's tie cases are exactly where that would
bite, which is why the fixture set leans on ties.

Outputs are canonicalised before hashing (sorted keys, fixed-width floats, no whitespace) so a
hash difference means a value difference, not a formatting difference.

Two hashes are recorded per run:
  result_hash    the scored profile, or just the exception TYPE for fixtures that raise
  message_hash   the same, plus the exception MESSAGE text

They are separated on purpose. A scorer that always refuses the same input for the same reason is
deterministic in the sense the pipeline depends on, even if the human-readable message it prints
varies. Reporting them as one number would conflate the two.

Run:
    scripts/eval/../../.venv/bin/python scripts/eval/exp2_determinism.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
LLM_ROOT = HERE.parents[1]           # .../SIP/llm
SRC = LLM_ROOT / "src"
RESULTS = HERE / "results"

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(HERE))

from fixtures import FIXTURES  # noqa: E402

from afrilearner360_llm.profiling.scorer import ItemResponse, score_student  # noqa: E402

WITHIN_PROCESS_REPEATS = 10
HASH_SEEDS = ["0", "1", "42"]

# Floats are rendered at fixed width rather than via repr(). repr() is already round-trip stable
# in CPython, but pinning the format removes the question entirely and makes the hashes readable
# by anything, not just Python.
FLOAT_FORMAT = "{:.10f}"


def _canonicalise(value: Any) -> Any:
    if isinstance(value, float):
        return FLOAT_FORMAT.format(value)
    if isinstance(value, dict):
        return {str(k): _canonicalise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonicalise(v) for v in value]
    return value


def canonical_json(payload: Any) -> str:
    return json.dumps(
        _canonicalise(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def score_one(fixture: Dict[str, Any]) -> Dict[str, Any]:
    """Score a fixture and reduce it to canonical text + hashes.

    Response objects are constructed INSIDE the try block: some fixtures are invalid at the
    ItemResponse level rather than the score_student level, and those pydantic errors are part of
    what we are measuring.
    """
    try:
        responses: List[ItemResponse] = [
            ItemResponse(item_id=r["item_id"], allocations=r["allocations"])
            for r in fixture["responses"]
        ]
        profile = score_student(
            student_id=fixture["student_id"],
            responses=responses,
            **fixture["kwargs"],
        )
        result_payload: Any = {"status": "ok", "profile": profile.model_dump(mode="json")}
        message_payload: Any = result_payload
    except Exception as exc:  # noqa: BLE001 -- a raise IS the result for the err_* fixtures
        result_payload = {"status": "raised", "error_type": type(exc).__name__}
        message_payload = {
            "status": "raised",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }

    result_text = canonical_json(result_payload)
    message_text = canonical_json(message_payload)
    return {
        "fixture": fixture["name"],
        "status": result_payload["status"],
        "result_canonical": result_text,
        "message_canonical": message_text,
        "result_hash": sha256(result_text),
        "message_hash": sha256(message_text),
    }


def score_all() -> List[Dict[str, Any]]:
    return [score_one(fx) for fx in FIXTURES]


def emit_hashes() -> None:
    """Subprocess mode: score everything once and print it, for the parent to compare."""
    payload = {
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", "<unset>"),
        "python": sys.version.split()[0],
        "runs": score_all(),
    }
    sys.stdout.write(json.dumps(payload))


def run_within_process() -> Dict[str, Any]:
    """Score every fixture WITHIN_PROCESS_REPEATS times and check the hashes never move."""
    per_fixture: Dict[str, List[Dict[str, Any]]] = {fx["name"]: [] for fx in FIXTURES}
    for _ in range(WITHIN_PROCESS_REPEATS):
        for record in score_all():
            per_fixture[record["fixture"]].append(record)

    results = []
    for name, records in per_fixture.items():
        result_hashes = {r["result_hash"] for r in records}
        message_hashes = {r["message_hash"] for r in records}
        results.append(
            {
                "fixture": name,
                "status": records[0]["status"],
                "repeats": len(records),
                "result_hash": records[0]["result_hash"],
                "message_hash": records[0]["message_hash"],
                "result_stable": len(result_hashes) == 1,
                "message_stable": len(message_hashes) == 1,
                "distinct_result_hashes": sorted(result_hashes),
                "distinct_message_hashes": sorted(message_hashes),
            }
        )

    matched = sum(1 for r in results if r["result_stable"])
    msg_matched = sum(1 for r in results if r["message_stable"])
    return {
        "repeats_per_fixture": WITHIN_PROCESS_REPEATS,
        "fixtures": len(results),
        "total_scorings": WITHIN_PROCESS_REPEATS * len(results),
        "exact_match_rate_result": matched / len(results),
        "exact_match_rate_message": msg_matched / len(results),
        "per_fixture": results,
    }


def run_across_processes(baseline: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Re-score the fixture set in fresh interpreters under different PYTHONHASHSEED values."""
    env_base = dict(os.environ)
    env_base["PYTHONPATH"] = os.pathsep.join(
        [str(SRC), str(HERE), env_base.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    subprocess_runs = []
    for seed in HASH_SEEDS:
        env = dict(env_base)
        env["PYTHONHASHSEED"] = seed
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--emit-hashes"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(HERE),
            timeout=180,
        )
        if completed.returncode != 0:
            subprocess_runs.append(
                {
                    "pythonhashseed": seed,
                    "ok": False,
                    "returncode": completed.returncode,
                    "stderr": completed.stderr[-4000:],
                }
            )
            continue
        payload = json.loads(completed.stdout)
        payload["ok"] = True
        subprocess_runs.append(payload)

    comparisons = []
    ok_runs = [r for r in subprocess_runs if r["ok"]]
    for fixture in FIXTURES:
        name = fixture["name"]
        base = baseline[name]
        observed_result = {base["result_hash"]}
        observed_message = {base["message_hash"]}
        divergent = []
        for run in ok_runs:
            record = next(r for r in run["runs"] if r["fixture"] == name)
            observed_result.add(record["result_hash"])
            observed_message.add(record["message_hash"])
            if record["message_hash"] != base["message_hash"]:
                divergent.append(
                    {
                        "pythonhashseed": run["pythonhashseed"],
                        "result_hash": record["result_hash"],
                        "message_hash": record["message_hash"],
                        "result_differs": record["result_hash"] != base["result_hash"],
                        "baseline_canonical": base["message_canonical"],
                        "observed_canonical": record["message_canonical"],
                    }
                )
        comparisons.append(
            {
                "fixture": name,
                "status": base["status"],
                "result_stable": len(observed_result) == 1,
                "message_stable": len(observed_message) == 1,
                "distinct_result_hashes": sorted(observed_result),
                "distinct_message_hashes": sorted(observed_message),
                "divergences": divergent,
            }
        )

    matched = sum(1 for c in comparisons if c["result_stable"])
    msg_matched = sum(1 for c in comparisons if c["message_stable"])
    return {
        "hash_seeds": HASH_SEEDS,
        "processes": len(HASH_SEEDS),
        "subprocess_failures": [r for r in subprocess_runs if not r["ok"]],
        "fixtures": len(comparisons),
        "exact_match_rate_result": matched / len(comparisons),
        "exact_match_rate_message": msg_matched / len(comparisons),
        "per_fixture": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emit-hashes",
        action="store_true",
        help="internal: score the fixture set once and print JSON (used for the subprocess runs)",
    )
    args = parser.parse_args()

    if args.emit_hashes:
        emit_hashes()
        return

    started = datetime.now(timezone.utc)
    within = run_within_process()

    # The first within-process scoring is the baseline every subprocess is compared against.
    baseline = {r["fixture"]: r for r in score_all()}
    across = run_across_processes(baseline)

    report = {
        "experiment": "exp2_scorer_determinism",
        "generated_at": started.isoformat(),
        "python": sys.version.split()[0],
        "parent_pythonhashseed": os.environ.get("PYTHONHASHSEED", "<unset>"),
        "scorer_module": "afrilearner360_llm.profiling.scorer",
        "float_format": FLOAT_FORMAT,
        "n_fixtures": len(FIXTURES),
        "n_ok_fixtures": sum(1 for r in baseline.values() if r["status"] == "ok"),
        "n_raising_fixtures": sum(1 for r in baseline.values() if r["status"] == "raised"),
        "within_process": within,
        "across_processes": across,
        "baseline": [
            {
                "fixture": name,
                "status": rec["status"],
                "result_hash": rec["result_hash"],
                "message_hash": rec["message_hash"],
                "canonical": rec["message_canonical"],
            }
            for name, rec in baseline.items()
        ],
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / "exp2_determinism.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"fixtures                     {len(FIXTURES)} "
          f"({report['n_ok_fixtures']} scored, {report['n_raising_fixtures']} expected to raise)")
    print(f"within-process scorings      {within['total_scorings']} "
          f"({WITHIN_PROCESS_REPEATS} x {within['fixtures']})")
    print(f"within  exact_match (result) {within['exact_match_rate_result']:.4f}")
    print(f"within  exact_match (msg)    {within['exact_match_rate_message']:.4f}")
    print(f"across  hash seeds           {', '.join(HASH_SEEDS)}")
    print(f"across  exact_match (result) {across['exact_match_rate_result']:.4f}")
    print(f"across  exact_match (msg)    {across['exact_match_rate_message']:.4f}")

    if across["subprocess_failures"]:
        print("\nSUBPROCESS FAILURES:")
        for failure in across["subprocess_failures"]:
            print(f"  seed={failure['pythonhashseed']} rc={failure['returncode']}")
            print(failure["stderr"])

    unstable = [c for c in across["per_fixture"] if not c["message_stable"]]
    if unstable:
        print(f"\nNONDETERMINISM: {len(unstable)} fixture(s) differ across hash seeds")
        for entry in unstable:
            scope = "SCORED OUTPUT" if not entry["result_stable"] else "error message text only"
            print(f"  - {entry['fixture']}  [{scope}]")
            print(f"      distinct message hashes across {len(entry['divergences']) + 1} processes: "
                  f"{len(entry['distinct_message_hashes'])}")
            parent_seed = os.environ.get("PYTHONHASHSEED", "unset")
            print(f"      parent (PYTHONHASHSEED={parent_seed}): "
                  f"{entry['divergences'][0]['baseline_canonical'][:200]}")
            for divergence in entry["divergences"]:
                print(f"      PYTHONHASHSEED={divergence['pythonhashseed']:<3} "
                      f"{divergence['observed_canonical'][:200]}")
    else:
        print("\nno nondeterminism detected")

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
