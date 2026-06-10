"""Testes unitários para utils/ts_compat.py."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.ts_compat import CompatLanguage, CompatParser, JAVA_LANGUAGE

SIMPLE_JAVA = b"""
public class Foo {
    public int add(int a, int b) { return a + b; }
}
"""


# ---------------------------------------------------------------------------
# CompatParser
# ---------------------------------------------------------------------------

def test_compat_parser_parse_returns_tree():
    parser = CompatParser()
    tree = parser.parse(SIMPLE_JAVA)
    assert tree is not None
    assert tree.root_node is not None
    assert tree.root_node.type == "program"


def test_compat_parser_set_language_noop():
    parser = CompatParser()
    parser.set_language(None)
    tree = parser.parse(SIMPLE_JAVA)
    assert tree.root_node.type == "program"


def test_compat_parser_parse_invalid_java_still_returns_tree():
    parser = CompatParser()
    tree = parser.parse(b"not valid java ??? {{{")
    assert tree is not None
    assert tree.root_node is not None


# ---------------------------------------------------------------------------
# CompatLanguage / CompatQuery
# ---------------------------------------------------------------------------

def test_compat_language_query_returns_tuples():
    query_src = "(method_declaration name: (identifier) @method_name)"
    captures = JAVA_LANGUAGE.query(query_src).captures(
        CompatParser().parse(SIMPLE_JAVA).root_node
    )
    assert isinstance(captures, list)
    assert len(captures) > 0
    node, name = captures[0]
    assert name == "method_name"
    assert node.text == b"add"


def test_compat_query_sorted_by_position():
    java = b"""
    public class Bar {
        public void first() {}
        public void second() {}
        public void third() {}
    }
    """
    query_src = "(method_declaration name: (identifier) @name)"
    captures = JAVA_LANGUAGE.query(query_src).captures(
        CompatParser().parse(java).root_node
    )
    names = [n.text.decode() for n, _ in captures]
    assert names == sorted(names, key=lambda x: ["first", "second", "third"].index(x))


# ---------------------------------------------------------------------------
# JAVA_LANGUAGE singleton
# ---------------------------------------------------------------------------

def test_java_language_singleton():
    assert isinstance(JAVA_LANGUAGE, CompatLanguage)


def test_java_language_delegates_attrs():
    assert hasattr(JAVA_LANGUAGE, "raw")
    from tree_sitter import Language
    assert isinstance(JAVA_LANGUAGE.raw, Language)
