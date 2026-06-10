"""Testes unitários para tools/monitor.py."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ---------------------------------------------------------------------------
# helpers — importamos as funções puras diretamente, sem executar o módulo
# ---------------------------------------------------------------------------

def _import_monitor_funcs():
    import importlib
    import types

    fake_config = types.ModuleType("data.configuration")
    fake_config.projects = ["Lang", "Math"]
    fake_config.target_models = ["gemma3_4b"]
    fake_config.formats = ["comment"]
    fake_config.strategies = ["extend"]
    fake_config.ablations = ["full"]
    fake_config.code_base = "/fake"

    sys.modules.setdefault("data", types.ModuleType("data"))
    sys.modules["data.configuration"] = fake_config

    import tools.monitor as m
    importlib.reload(m)
    return m


# ---------------------------------------------------------------------------
# expected
# ---------------------------------------------------------------------------

def test_expected_counts_missing_file(tmp_path):
    m = _import_monitor_funcs()
    with patch.object(m, "CODE_BASE", str(tmp_path)):
        count = m.expected("gemma3_4b", "Lang")
    assert count == 0


def test_expected_counts_filters_project_and_public(tmp_path):
    m = _import_monitor_funcs()
    records = [
        {"project": "Lang", "is_public": "1"},
        {"project": "Lang", "is_public": "0"},
        {"project": "Math", "is_public": "1"},
    ]
    outputs_dir = tmp_path / "data" / "outputs"
    outputs_dir.mkdir(parents=True)
    output_file = outputs_dir / "gemma3_4b_comment_extend_full.jsonl"
    output_file.write_text(
        "\n".join(json.dumps(r) for r in records), encoding="utf-8"
    )
    with patch.object(m, "CODE_BASE", str(tmp_path)):
        count = m.expected("gemma3_4b", "Lang")
    assert count == 1


# ---------------------------------------------------------------------------
# got
# ---------------------------------------------------------------------------

def test_got_counts_lines(tmp_path):
    m = _import_monitor_funcs()
    results_dir = tmp_path / "gemma3_4b"
    results_dir.mkdir(parents=True)
    result_file = results_dir / "Lang_comment_extend_full.jsonl"
    result_file.write_text(
        json.dumps({"id": "Lang_1"}) + "\n" +
        json.dumps({"id": "Lang_2"}) + "\n" +
        "\n",
        encoding="utf-8",
    )
    with patch.object(m, "RESULTS", str(tmp_path)):
        count = m.got("gemma3_4b", "Lang")
    assert count == 2


def test_got_returns_zero_for_missing(tmp_path):
    m = _import_monitor_funcs()
    with patch.object(m, "RESULTS", str(tmp_path)):
        assert m.got("gemma3_4b", "Lang") == 0


# ---------------------------------------------------------------------------
# quick_stats
# ---------------------------------------------------------------------------

def test_quick_stats_csr_calculation(tmp_path):
    m = _import_monitor_funcs()
    results = [
        {"second_compile_res": "success", "num_compilable_uts": 3},
        {"second_compile_res": "failed",  "num_compilable_uts": 0},
        {"second_compile_res": "success", "num_compilable_uts": 2},
    ]
    results_dir = tmp_path / "gemma3_4b"
    results_dir.mkdir(parents=True)
    (results_dir / "Lang_comment_extend_full.jsonl").write_text(
        "\n".join(json.dumps(r) for r in results), encoding="utf-8"
    )
    with patch.object(m, "RESULTS", str(tmp_path)):
        stats = m.quick_stats("gemma3_4b", "Lang")
    assert stats["n"] == 3
    assert stats["csr"] == 2
    assert stats["uts_c"] == 5


def test_quick_stats_returns_none_on_empty(tmp_path):
    m = _import_monitor_funcs()
    results_dir = tmp_path / "gemma3_4b"
    results_dir.mkdir(parents=True)
    (results_dir / "Lang_comment_extend_full.jsonl").write_text("", encoding="utf-8")
    with patch.object(m, "RESULTS", str(tmp_path)):
        assert m.quick_stats("gemma3_4b", "Lang") is None
