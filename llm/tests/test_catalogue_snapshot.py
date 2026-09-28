"""Tests for the catalogue snapshot used by the paper's model-availability claim.

The claim these support is longitudinal and load-bearing, so the counting rules are pinned:
a `:free` entry and its paid counterpart are distinct models, a model declaring
`structured_outputs` is not counted again under `response_format`, and an absent
`supported_parameters` key is not treated as if it declared something.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parents[1] / "scripts" / "eval"
sys.path.insert(0, str(EVAL_DIR))

from catalogue_snapshot import (  # noqa: E402
    is_free_variant,
    paid_counterpart_present,
    summarise,
    supported_parameters,
    write_snapshot,
)


def model(model_id: str, params=None, omit_params: bool = False) -> dict:
    entry = {"id": model_id}
    if not omit_params:
        entry["supported_parameters"] = params if params is not None else []
    return entry


class TestSupportedParameters:
    def test_absent_key_is_empty_not_error(self):
        assert supported_parameters(model("a/b", omit_params=True)) == []

    def test_explicit_none_is_empty(self):
        assert supported_parameters({"id": "a/b", "supported_parameters": None}) == []

    def test_declared_parameters_returned(self):
        assert supported_parameters(model("a/b", ["structured_outputs"])) == ["structured_outputs"]


class TestFreeVariant:
    def test_free_suffix_detected(self):
        assert is_free_variant(model("minimax/minimax-m3:free"))

    def test_paid_entry_is_not_free(self):
        # The distinction the paper's claim rests on: the paid model is not the free model.
        assert not is_free_variant(model("minimax/minimax-m3"))

    def test_free_substring_elsewhere_does_not_count(self):
        assert not is_free_variant(model("vendor/free-lunch-7b"))


class TestSummarise:
    def test_counts_only_free_variants(self):
        models = [
            model("a/x:free", ["structured_outputs"]),
            model("a/x", ["structured_outputs"]),
            model("b/y", ["structured_outputs"]),
        ]
        result = summarise(models)
        assert result["total_models"] == 3
        assert result["free_models"] == 1
        assert result["free_declaring_schema"] == 1

    def test_schema_model_not_double_counted_as_json_mode(self):
        # OpenRouter lists both parameters for schema-capable models; counting the model in
        # both buckets would inflate the json-mode-only figure the paper contrasts against.
        models = [model("a/x:free", ["response_format", "structured_outputs"])]
        result = summarise(models)
        assert result["free_declaring_schema"] == 1
        assert result["free_declaring_json_mode_only"] == 0

    def test_json_mode_only_counted_separately(self):
        models = [model("a/x:free", ["response_format"])]
        result = summarise(models)
        assert result["free_declaring_schema"] == 0
        assert result["free_declaring_json_mode_only"] == 1

    def test_model_declaring_nothing_counted_in_neither(self):
        models = [model("a/x:free", []), model("b/y:free", omit_params=True)]
        result = summarise(models)
        assert result["free_declaring_neither"] == 2
        assert result["free_declaring_schema"] == 0

    def test_buckets_partition_the_free_models(self):
        models = [
            model("a/x:free", ["structured_outputs"]),
            model("b/y:free", ["response_format"]),
            model("c/z:free", []),
            model("d/w", ["structured_outputs"]),
        ]
        result = summarise(models)
        total = (
            result["free_declaring_schema"]
            + result["free_declaring_json_mode_only"]
            + result["free_declaring_neither"]
        )
        assert total == result["free_models"]

    def test_ids_are_sorted_for_stable_diffs(self):
        models = [model("z/z:free", ["structured_outputs"]), model("a/a:free", ["structured_outputs"])]
        assert summarise(models)["schema_model_ids"] == ["a/a:free", "z/z:free"]

    def test_empty_catalogue(self):
        result = summarise([])
        assert result["total_models"] == 0
        assert result["free_models"] == 0
        assert result["schema_model_ids"] == []


class TestPaidCounterpart:
    def test_detects_surviving_paid_entry(self):
        # The real 2026-09-27 case: the free variant is gone, the paid model is not.
        models = [model("minimax/minimax-m3", ["structured_outputs"])]
        assert paid_counterpart_present(models, "minimax/minimax-m3:free")

    def test_absent_paid_entry_reported(self):
        models = [model("other/model")]
        assert not paid_counterpart_present(models, "minimax/minimax-m3:free")

    def test_does_not_match_on_prefix(self):
        models = [model("minimax/minimax-m30")]
        assert not paid_counterpart_present(models, "minimax/minimax-m3:free")


class TestWriteSnapshot:
    def test_refuses_to_overwrite_without_force(self, tmp_path, monkeypatch):
        import catalogue_snapshot

        monkeypatch.setattr(catalogue_snapshot, "SNAPSHOT_DIR", tmp_path)
        models = [model("a/x:free", ["structured_outputs"])]
        first = catalogue_snapshot.write_snapshot(models, force=False)
        assert first.exists()
        with pytest.raises(SystemExit):
            catalogue_snapshot.write_snapshot(models, force=False)

    def test_force_overwrites(self, tmp_path, monkeypatch):
        import catalogue_snapshot

        monkeypatch.setattr(catalogue_snapshot, "SNAPSHOT_DIR", tmp_path)
        catalogue_snapshot.write_snapshot([model("a/x:free", ["structured_outputs"])], force=False)
        path = catalogue_snapshot.write_snapshot([model("a/x:free", []), model("b/y:free", [])], force=True)
        stored = json.loads(path.read_text())
        assert stored["free_models"] == 2
        assert stored["free_declaring_schema"] == 0

    def test_snapshot_records_capture_time_and_source(self, tmp_path, monkeypatch):
        import catalogue_snapshot

        monkeypatch.setattr(catalogue_snapshot, "SNAPSHOT_DIR", tmp_path)
        path = catalogue_snapshot.write_snapshot([model("a/x:free", [])], force=False)
        stored = json.loads(path.read_text())
        assert stored["source"] == catalogue_snapshot.CATALOGUE_URL
        assert stored["captured_at"].endswith("+00:00")
