"""Testes unitários para tools/run_groq_inference.py."""
import json
import sys
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.run_groq_inference import (
    DailyLimitExceeded,
    _load_env_file,
    groq_chat,
    load_done,
    parse_gemma_prompt,
    strip_thinking_blocks,
)


# ---------------------------------------------------------------------------
# strip_thinking_blocks
# ---------------------------------------------------------------------------

def test_strip_thinking_blocks_closed():
    text = "before<think>inner reasoning</think>after"
    assert strip_thinking_blocks(text) == "beforeafter"


def test_strip_thinking_blocks_open():
    text = "valid code<think>incomplete reasoning"
    assert strip_thinking_blocks(text) == "valid code"


def test_strip_thinking_blocks_no_block():
    text = "clean output"
    assert strip_thinking_blocks(text) == "clean output"


def test_strip_thinking_blocks_multiple():
    text = "<think>a</think>code<think>b</think>more"
    assert strip_thinking_blocks(text) == "codemore"


# ---------------------------------------------------------------------------
# parse_gemma_prompt
# ---------------------------------------------------------------------------

def test_parse_gemma_prompt_full():
    raw = (
        "<start_of_turn>user\n"
        "Write tests for isEmpty\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
        "import org.junit.Test;\npublic class T {\n"
    )
    user_msg, prefix = parse_gemma_prompt(raw)
    assert "isEmpty" in user_msg
    assert prefix.startswith("import org.junit.Test;")


def test_parse_gemma_prompt_no_model_turn():
    raw = (
        "<start_of_turn>user\n"
        "Write tests\n"
        "<end_of_turn>\n"
    )
    user_msg, prefix = parse_gemma_prompt(raw)
    assert "Write tests" in user_msg
    assert prefix == ""


def test_parse_gemma_prompt_unrecognized_format():
    raw = "plain prompt without gemma markers"
    user_msg, prefix = parse_gemma_prompt(raw)
    assert user_msg == raw
    assert prefix == ""


# ---------------------------------------------------------------------------
# _load_env_file
# ---------------------------------------------------------------------------

def test_load_env_file_finds_key(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('GROQ_API_KEY="gsk_test123"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    with patch("tools.run_groq_inference.__file__", str(tmp_path / "run_groq_inference.py")):
        result = _load_env_file()
    assert result.get("GROQ_API_KEY") == "gsk_test123"


def test_load_env_file_missing_returns_empty(tmp_path):
    with patch("tools.run_groq_inference.__file__", str(tmp_path / "sub" / "run.py")):
        result = _load_env_file()
    assert result == {}


# ---------------------------------------------------------------------------
# groq_chat
# ---------------------------------------------------------------------------

def _mock_urlopen(body: dict):
    data = json.dumps(body).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = data
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def test_groq_chat_success():
    resp_body = {"choices": [{"message": {"content": "generated test"}}]}
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(resp_body)):
        result = groq_chat(
            "gsk_key", "gemma-3-12b-it", "write a test", "",
            max_tokens=512, temperature=0.0, top_p=0.95,
            timeout=30, retries=1,
        )
    assert result == "generated test"


def test_groq_chat_rate_limit_waits():
    http_error = urllib.error.HTTPError(
        url="", code=429, msg="Too Many Requests",
        hdrs=MagicMock(**{"get.return_value": "1"}),
        fp=MagicMock(read=lambda: b"TPM limit"),
    )
    resp_body = {"choices": [{"message": {"content": "ok"}}]}
    responses = [http_error, _mock_urlopen(resp_body)]

    def fake_urlopen(*args, **kwargs):
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        with patch("time.sleep"):
            result = groq_chat(
                "gsk_key", "model", "msg", "",
                max_tokens=256, temperature=0.0, top_p=0.95,
                timeout=30, retries=3,
            )
    assert result == "ok"


def test_groq_chat_daily_limit_raises():
    http_error = urllib.error.HTTPError(
        url="", code=429, msg="Too Many Requests",
        hdrs=MagicMock(**{"get.return_value": "60"}),
        fp=MagicMock(read=lambda: b'{"error": "TPD quota exceeded per day"}'),
    )
    with patch("urllib.request.urlopen", side_effect=http_error):
        with pytest.raises(DailyLimitExceeded):
            groq_chat(
                "gsk_key", "model", "msg", "",
                max_tokens=256, temperature=0.0, top_p=0.95,
                timeout=30, retries=1,
            )


# ---------------------------------------------------------------------------
# main — rotação de chaves e marcador TOO_LARGE
# ---------------------------------------------------------------------------

def test_main_rotates_api_key_on_daily_limit(tmp_path):
    prompt_rec = {"id": "Lang_1", "method_signature": "sig",
                  "prompt": "<start_of_turn>user\nmsg<end_of_turn>\n<start_of_turn>model\n"}
    inp = tmp_path / "prompts.jsonl"
    out = tmp_path / "out.jsonl"
    inp.write_text(json.dumps(prompt_rec) + "\n", encoding="utf-8")

    calls = {"n": 0}

    def fake_chat(api_key, *args, **kwargs):
        calls["n"] += 1
        if api_key == "key1":
            raise DailyLimitExceeded("limit")
        return "code"

    with patch("tools.run_groq_inference.groq_chat", side_effect=fake_chat):
        with patch("time.sleep"):
            from tools.run_groq_inference import main
            with patch("sys.argv", ["prog", "--input", str(inp), "--output", str(out),
                                    "--api-key", "key1,key2", "--limit", "1"]):
                main()

    out_lines = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert out_lines[0]["completion"] == "code"


def test_main_saves_too_large_marker(tmp_path):
    prompt_rec = {"id": "Lang_1", "method_signature": "sig",
                  "prompt": "<start_of_turn>user\nmsg<end_of_turn>\n<start_of_turn>model\n"}
    inp = tmp_path / "prompts.jsonl"
    out = tmp_path / "out.jsonl"
    inp.write_text(json.dumps(prompt_rec) + "\n", encoding="utf-8")

    with patch("tools.run_groq_inference.groq_chat",
               side_effect=RuntimeError("HTTP 413: prompt too large")):
        with patch("time.sleep"):
            from tools.run_groq_inference import main
            with patch("sys.argv", ["prog", "--input", str(inp), "--output", str(out),
                                    "--api-key", "gsk_key", "--limit", "1"]):
                main()

    out_lines = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert out_lines[0]["completion"] == "TOO_LARGE"
