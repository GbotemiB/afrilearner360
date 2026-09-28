#!/usr/bin/env python3
"""Snapshot the OpenRouter model catalogue: how many models exist, how many are free, and how
many of those declare server-side JSON-schema enforcement.

This is the measurement behind the paper's availability claim. It is a separate script from
exp1 because it costs no generation quota (one unauthenticated GET) and because the claim it
supports is longitudinal: a single snapshot says little, whereas a series of dated snapshots is
the evidence that free-tier availability is non-stationary.

Snapshots accumulate under results/catalogue/ as catalogue_<UTC date>.json and are never
overwritten, so re-running on a day already captured is refused rather than silently rewriting
history.

A note on the distinction this script is careful about: a model and its free variant are
different catalogue entries. `minimax/minimax-m3` and `minimax/minimax-m3:free` can have
different fates, and on 2026-09-27 they did. A zero-budget pipeline depends on the `:free`
entry, so that is what `free_models` counts.

Run:
    .venv/bin/python scripts/eval/catalogue_snapshot.py
    .venv/bin/python scripts/eval/catalogue_snapshot.py --compare
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
SNAPSHOT_DIR = HERE / "results" / "catalogue"
CATALOGUE_URL = "https://openrouter.ai/api/v1/models"

FREE_SUFFIX = ":free"
SCHEMA_PARAM = "structured_outputs"
JSON_MODE_PARAM = "response_format"


def fetch_catalogue(url: str = CATALOGUE_URL) -> List[Dict[str, Any]]:
    """Fetch the public model list. No API key required, so this costs no quota."""
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["data"]


def supported_parameters(model: Dict[str, Any]) -> List[str]:
    """Declared parameters for one model, as a list.

    The key is absent for some entries rather than present-and-empty, so absence is checked
    explicitly. An absent key means the model declares nothing, which is materially different
    from declaring schema support, and must not be conflated with it.
    """
    if "supported_parameters" not in model:
        return []
    declared = model["supported_parameters"]
    if declared is None:
        return []
    return list(declared)


def is_free_variant(model: Dict[str, Any]) -> bool:
    return model["id"].endswith(FREE_SUFFIX)


def summarise(models: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Counts plus the free-model IDs, split by the JSON capability each declares."""
    free = [m for m in models if is_free_variant(m)]
    schema = sorted(m["id"] for m in free if SCHEMA_PARAM in supported_parameters(m))
    json_mode = sorted(
        m["id"]
        for m in free
        if JSON_MODE_PARAM in supported_parameters(m) and SCHEMA_PARAM not in supported_parameters(m)
    )
    neither = sorted(
        m["id"]
        for m in free
        if SCHEMA_PARAM not in supported_parameters(m)
        and JSON_MODE_PARAM not in supported_parameters(m)
    )
    return {
        "total_models": len(models),
        "free_models": len(free),
        "free_declaring_schema": len(schema),
        "free_declaring_json_mode_only": len(json_mode),
        "free_declaring_neither": len(neither),
        "schema_model_ids": schema,
        "json_mode_only_model_ids": json_mode,
        "neither_model_ids": neither,
    }


def paid_counterpart_present(models: List[Dict[str, Any]], free_id: str) -> bool:
    """Whether the paid entry for a `:free` id still exists.

    Distinguishes 'this model was withdrawn' from 'this model is no longer free', which are
    different claims and only the second is true of minimax-m3 and glm-5.2 as of 2026-09-27.
    """
    base = free_id[: -len(FREE_SUFFIX)]
    return any(m["id"] == base for m in models)


def write_snapshot(models: List[Dict[str, Any]], *, force: bool) -> Path:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    captured = datetime.now(timezone.utc)
    path = SNAPSHOT_DIR / f"catalogue_{captured.date().isoformat()}.json"
    if path.exists() and not force:
        raise SystemExit(f"snapshot already exists for today: {path} (use --force to overwrite)")

    snapshot = {
        "captured_at": captured.isoformat(),
        "source": CATALOGUE_URL,
        **summarise(models),
    }
    path.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    return path


def load_snapshots() -> List[Dict[str, Any]]:
    if not SNAPSHOT_DIR.exists():
        return []
    snapshots = []
    for path in sorted(SNAPSHOT_DIR.glob("catalogue_*.json")):
        snapshots.append(json.loads(path.read_text(encoding="utf-8")))
    return snapshots


def print_series(snapshots: List[Dict[str, Any]]) -> None:
    print(f"{'captured':<12} {'total':>6} {'free':>5} {'schema':>7} {'json-only':>10}")
    for snap in snapshots:
        date = snap["captured_at"][:10]
        print(
            f"{date:<12} {snap['total_models']:>6} {snap['free_models']:>5} "
            f"{snap['free_declaring_schema']:>7} {snap['free_declaring_json_mode_only']:>10}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="overwrite today's snapshot")
    parser.add_argument(
        "--compare", action="store_true", help="print the series of stored snapshots and exit"
    )
    args = parser.parse_args()

    if args.compare:
        stored = load_snapshots()
        if not stored:
            raise SystemExit("no snapshots stored yet")
        print_series(stored)
        return

    models = fetch_catalogue()
    path = write_snapshot(models, force=args.force)
    summary = summarise(models)

    print(f"total models                  {summary['total_models']}")
    print(f"free models                   {summary['free_models']}")
    print(f"  declaring {SCHEMA_PARAM:<18} {summary['free_declaring_schema']}")
    print(f"  declaring {JSON_MODE_PARAM} only {summary['free_declaring_json_mode_only']}")
    print("\nfree models declaring schema enforcement:")
    for model_id in summary["schema_model_ids"]:
        print(f"    {model_id}")
    print(f"\nwrote {path}")

    stored = load_snapshots()
    if len(stored) > 1:
        print("\nseries:")
        print_series(stored)


if __name__ == "__main__":
    main()
