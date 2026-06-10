"""Testes unitários para tools/collect_fix_info.py e tools/collect_source_methods.py."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.collect_fix_info import (
    HunkChange,
    MethodInfo,
    extract_methods,
    methods_touching,
    parse_unified_diff,
)
from tools.collect_source_methods import (
    build_records_for_class,
    fqn_to_relpath,
    load_focal_classes,
)
import tree_sitter_java
from tree_sitter import Language, Parser

_JAVA_LANGUAGE = Language(tree_sitter_java.language())
_PARSER = Parser(_JAVA_LANGUAGE)

SIMPLE_PATCH = """\
--- a/src/Foo.java
+++ b/src/Foo.java
@@ -10,4 +10,4 @@
-    return a + b;
+    return a - b;
"""

JAVA_SOURCE = """\
package org.example;

public class Foo {
    public int add(int a, int b) {
        return a + b;
    }

    private void helper() {}
}
"""


# ---------------------------------------------------------------------------
# parse_unified_diff
# ---------------------------------------------------------------------------

def test_fix_info_parses_patch_correctly():
    patches = parse_unified_diff(SIMPLE_PATCH)
    assert len(patches) == 1
    assert patches[0].new_path == "src/Foo.java"
    assert len(patches[0].hunks) == 1
    assert patches[0].hunks[0].old_start == 10


def test_parse_unified_diff_ignores_non_java():
    patch = """\
--- a/README.md
+++ b/README.md
@@ -1,1 +1,1 @@
-old
+new
"""
    assert parse_unified_diff(patch) == []


def test_parse_unified_diff_multiple_files():
    patch = """\
--- a/Foo.java
+++ b/Foo.java
@@ -5,3 +5,3 @@
-old
+new
--- a/Bar.java
+++ b/Bar.java
@@ -2,2 +2,2 @@
-x
+y
"""
    patches = parse_unified_diff(patch)
    assert len(patches) == 2


# ---------------------------------------------------------------------------
# extract_methods
# ---------------------------------------------------------------------------

def test_fix_info_extracts_public_methods():
    methods = extract_methods(JAVA_SOURCE)
    names = [m.method_name for m in methods]
    assert "add" in names


def test_fix_info_extracts_private_methods():
    methods = extract_methods(JAVA_SOURCE)
    names = [m.method_name for m in methods]
    assert "helper" in names


def test_fix_info_qualified_class():
    methods = extract_methods(JAVA_SOURCE)
    add_method = next(m for m in methods if m.method_name == "add")
    assert add_method.qualified_class == "org.example.Foo"


# ---------------------------------------------------------------------------
# methods_touching
# ---------------------------------------------------------------------------

def test_fix_info_methods_touching_match():
    methods = [MethodInfo("org.Foo", "add", ["int", "int"], start_line=4, end_line=6)]
    hunks = [HunkChange(old_start=5, old_count=1, new_start=5, new_count=1)]
    matched = methods_touching(hunks, methods)
    assert len(matched) == 1
    assert matched[0].method_name == "add"


def test_fix_info_methods_touching_no_match():
    methods = [MethodInfo("org.Foo", "add", ["int", "int"], start_line=4, end_line=6)]
    hunks = [HunkChange(old_start=20, old_count=1, new_start=20, new_count=1)]
    assert methods_touching(hunks, methods) == []


def test_fix_info_no_duplicate_on_multiple_hunks():
    methods = [MethodInfo("org.Foo", "add", ["int", "int"], start_line=4, end_line=6)]
    hunks = [
        HunkChange(old_start=4, old_count=1, new_start=4, new_count=1),
        HunkChange(old_start=5, old_count=1, new_start=5, new_count=1),
    ]
    matched = methods_touching(hunks, methods)
    assert len(matched) == 1


# ---------------------------------------------------------------------------
# fqn_to_relpath
# ---------------------------------------------------------------------------

def test_source_methods_fqn_to_relpath_simple():
    assert fqn_to_relpath("org.example.Foo") == "org/example/Foo.java"


def test_source_methods_fqn_to_relpath_nested():
    assert fqn_to_relpath("org.example.Foo.Inner") == "org/example/Foo.java"


# ---------------------------------------------------------------------------
# build_records_for_class
# ---------------------------------------------------------------------------

def test_source_methods_output_format():
    src = JAVA_SOURCE.encode("utf-8")
    tree = _PARSER.parse(src)
    type_nodes = [n for n in tree.root_node.children if n.type == "class_declaration"]
    assert type_nodes, "nenhuma class_declaration encontrada"

    records = build_records_for_class(
        type_nodes[0],
        package="org.example",
        imports=["import java.util.List;"],
        project_full_id="Lang_1_fixed",
        src=src,
    )
    required_keys = [
        "source:source_method_code_format",
        "source:source_method_name",
        "source:source_method_signature",
        "source:source_other_method_signature",
        "content:source_class_name",
        "content:source_class_code_format",
        "content:source_class_code_imports",
        "content:source_class_constructors",
        "content:parameter_class_signature",
        "content:parameter_class_constructors",
        "source_class_fields",
        "extra:project_name",
    ]
    for rec in records:
        for key in required_keys:
            assert key in rec, f"chave ausente: {key}"


def test_source_methods_dedup_via_seen_files(tmp_path):
    """load_focal_classes não deve duplicar quando duas classes do mesmo arquivo."""
    bug_id = "Lang_1"
    fix_info = {
        "bug_id": bug_id,
        "project": "Lang",
        "fixing_changes": [
            {"file": "Foo.java", "changed_functions": [{"qualified_names": ["org.example.Foo:add:int,int"]}]},
            {"file": "Foo.java", "changed_functions": [{"qualified_names": ["org.example.Foo:helper:"]}]},
        ],
    }
    fix_dir = tmp_path / bug_id
    fix_dir.mkdir(parents=True)
    (fix_dir / "buggy_fix_info.json").write_text(
        json.dumps(fix_info), encoding="utf-8"
    )

    from tools.collect_source_methods import load_focal_classes
    import tools.collect_source_methods as csm
    original = csm.FIX_INFO_DIR
    csm.FIX_INFO_DIR = str(tmp_path)
    try:
        classes = load_focal_classes(bug_id)
    finally:
        csm.FIX_INFO_DIR = original

    assert "org.example.Foo" in classes
