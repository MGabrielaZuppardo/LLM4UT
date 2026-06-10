"""Testes unitários para tools/run_ollama_inference.py."""
import json
import sys
import urllib.error
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.run_ollama_inference import _key, load_done, ollama_generate


# ---------------------------------------------------------------------------
# _key
# ---------------------------------------------------------------------------

def test_key_uniqueness():
    rec = {"id": "Lang_1", "method_signature": "org.Lang#Foo#bar()"}
    assert _key(rec) == ("Lang_1", "org.Lang#Foo#bar()")


def test_key_missing_fields():
    assert _key({}) == ("", "")


# ---------------------------------------------------------------------------
# load_done
# ---------------------------------------------------------------------------

def test_load_done_empty_file(tmp_path):
    assert load_done(str(tmp_path / "nonexistent.jsonl")) == set()


def test_load_done_reads_completed(tmp_path):
    path = tmp_path / "out.jsonl"
    rec = {"id": "Lang_1", "method_signature": "sig", "completion": "code"}
    path.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    done = load_done(str(path))
    assert ("Lang_1", "sig") in done


def test_load_done_ignores_records_without_completion(tmp_path):
    path = tmp_path / "out.jsonl"
    rec = {"id": "Lang_1", "method_signature": "sig"}
    path.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    assert load_done(str(path)) == set()


def test_load_done_skips_malformed_json(tmp_path):
    path = tmp_path / "out.jsonl"
    path.write_text("not json\n{\"id\":\"x\",\"method_signature\":\"s\",\"completion\":\"c\"}\n",
                    encoding="utf-8")
    done = load_done(str(path))
    assert ("x", "s") in done
    assert len(done) == 1


# ---------------------------------------------------------------------------
# ollama_generate
# ---------------------------------------------------------------------------

def _make_response(body: dict):
    data = json.dumps(body).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = data
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def test_ollama_generate_success():
    mock_resp = _make_response({"response": "generated code"})
    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = ollama_generate(
            "http://localhost:11434", "gemma3:4b", "prompt",
            temperature=0.0, top_p=0.95, num_predict=256,
            timeout=30, retries=1,
        )
    assert result == "generated code"


def test_ollama_generate_retries_on_timeout():
    with patch("urllib.request.urlopen", side_effect=TimeoutError("timeout")):
        with patch("time.sleep"):
            with pytest.raises(RuntimeError, match="falha após"):
                ollama_generate(
                    "http://localhost:11434", "gemma3:4b", "prompt",
                    temperature=0.0, top_p=0.95, num_predict=256,
                    timeout=1, retries=2,
                )


# ---------------------------------------------------------------------------
# main — comportamento de skip e limit via argv
# ---------------------------------------------------------------------------

def test_main_skips_already_done(tmp_path):
    prompt_rec = {"id": "Lang_1", "method_signature": "sig", "prompt": "ask"}
    done_rec = dict(prompt_rec)
    done_rec["completion"] = "existing"

    inp = tmp_path / "prompts.jsonl"
    out = tmp_path / "out.jsonl"
    inp.write_text(json.dumps(prompt_rec) + "\n", encoding="utf-8")
    out.write_text(json.dumps(done_rec) + "\n", encoding="utf-8")

    call_count = {"n": 0}

    def fake_generate(*args, **kwargs):
        call_count["n"] += 1
        return "new"

    with patch("tools.run_ollama_inference.ollama_generate", side_effect=fake_generate):
        from tools.run_ollama_inference import main
        with patch("sys.argv", ["prog", "--input", str(inp), "--output", str(out),
                                "--model", "gemma3:4b"]):
            main()

    assert call_count["n"] == 0


def test_main_respects_limit(tmp_path):
    records = [
        {"id": f"Lang_{i}", "method_signature": f"sig{i}", "prompt": "ask"}
        for i in range(5)
    ]
    inp = tmp_path / "prompts.jsonl"
    out = tmp_path / "out.jsonl"
    inp.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    call_count = {"n": 0}

    def fake_generate(*args, **kwargs):
        call_count["n"] += 1
        return "code"

    with patch("tools.run_ollama_inference.ollama_generate", side_effect=fake_generate):
        from tools.run_ollama_inference import main
        with patch("sys.argv", ["prog", "--input", str(inp), "--output", str(out),
                                "--model", "gemma3:4b", "--limit", "2"]):
            main()

    assert call_count["n"] == 2
