"""Testes de integração: verifica compatibilidade da saída de inferência com rq1."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.run_groq_inference import strip_thinking_blocks


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_completion_record(completion: str, **kwargs) -> dict:
    base = {
        "id": "Lang_1",
        "project": "Lang",
        "method_signature": "org.apache.commons.lang3#StringUtils#isEmpty(java.lang.String)",
        "is_public": "1",
        "prompt": "...",
        "focal_method": "public static boolean isEmpty(String s) { return s == null; }",
        "format": "comment",
        "strategy": "extend",
        "ablation": "full",
        "completion": completion,
        "model": "gemma3:4b",
    }
    base.update(kwargs)
    return base


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# test_ollama_output_has_completion_key
# ---------------------------------------------------------------------------

def test_ollama_output_has_completion_key(tmp_path):
    records = [
        _make_completion_record("@Test\npublic void testIsEmpty() { assertTrue(true); }"),
        _make_completion_record("@Test\npublic void testNull() { assertNull(null); }"),
    ]
    out = tmp_path / "completions.jsonl"
    _write_jsonl(out, records)

    loaded = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    for rec in loaded:
        assert "completion" in rec, f"chave 'completion' ausente em {rec.get('id')}"
        assert rec["completion"] != "", "completion não pode ser vazia"


# ---------------------------------------------------------------------------
# test_groq_output_compatible_with_rq1
# ---------------------------------------------------------------------------

def test_groq_output_compatible_with_rq1(tmp_path):
    """Simula o que filter_data_according_to_project faz ao ler o output do Groq."""
    records = [
        _make_completion_record(
            "@Test\npublic void testA() {}",
            id="Lang_1",
            project="Lang",
            is_public="1",
        ),
        _make_completion_record(
            "@Test\npublic void testB() {}",
            id="Math_1",
            project="Math",
            is_public="1",
        ),
    ]
    out = tmp_path / "completions.jsonl"
    _write_jsonl(out, records)

    loaded = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    lang_records = [r for r in loaded if r.get("project") == "Lang" and r.get("is_public") == "1"]

    assert len(lang_records) == 1
    assert lang_records[0]["id"] == "Lang_1"
    assert "completion" in lang_records[0]
    assert "method_signature" in lang_records[0]


# ---------------------------------------------------------------------------
# test_thinking_blocks_absent_in_output
# ---------------------------------------------------------------------------

def test_thinking_blocks_absent_in_output(tmp_path):
    raw_completions = [
        "<think>let me reason step by step</think>@Test\npublic void test() {}",
        "@Test\npublic void cleanTest() { assertTrue(true); }",
        "code<think>truncated by max_tokens",
    ]
    records = [_make_completion_record(c) for c in raw_completions]
    for rec in records:
        rec["completion"] = strip_thinking_blocks(rec["completion"])

    out = tmp_path / "completions.jsonl"
    _write_jsonl(out, records)

    loaded = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    for rec in loaded:
        assert "<think>" not in rec["completion"], (
            f"bloco <think> encontrado na completion de {rec.get('id')}: {rec['completion'][:80]}"
        )
