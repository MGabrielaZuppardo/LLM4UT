"""Fixtures compartilhadas entre todos os testes do LLM4UT."""
import json
import pytest


@pytest.fixture
def sample_prompt_record():
    return {
        "id": "Lang_1",
        "project": "Lang",
        "method_signature": "org.apache.commons.lang3#StringUtils#isEmpty(java.lang.String)",
        "is_public": "1",
        "prompt": (
            "<start_of_turn>user\n"
            "Write a unit test for the following Java method.\n"
            "public static boolean isEmpty(String s) { return s == null || s.isEmpty(); }\n"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
            "import org.junit.Test;\nimport static org.junit.Assert.*;\n"
            "public class GeneratedTest {\n"
        ),
        "focal_method": "public static boolean isEmpty(String s) { return s == null || s.isEmpty(); }",
        "format": "comment",
        "strategy": "extend",
        "ablation": "full",
    }


@pytest.fixture
def sample_completion_record(sample_prompt_record):
    rec = dict(sample_prompt_record)
    rec["completion"] = "    @Test\n    public void testIsEmpty() { assertTrue(StringUtils.isEmpty(null)); }\n}"
    rec["model"] = "gemma3:4b"
    return rec


@pytest.fixture
def jsonl_prompts_file(tmp_path, sample_prompt_record):
    path = tmp_path / "prompts.jsonl"
    path.write_text(json.dumps(sample_prompt_record, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def jsonl_output_file(tmp_path, sample_completion_record):
    path = tmp_path / "completions.jsonl"
    path.write_text(json.dumps(sample_completion_record, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
