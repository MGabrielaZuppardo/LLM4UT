# Executar dentro do container:
#   pytest docker-tests/smoke_test.py -v
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_python_version():
    assert sys.version_info >= (3, 9)


def test_pacotes_instalados():
    import pandas
    import numpy
    import scipy
    import javalang
    import tree_sitter
    import openai
    import groq
    import ollama


def test_tree_sitter_parse_java():
    from tree_sitter import Language, Parser
    import tree_sitter_java as tsjava

    parser = Parser(Language(tsjava.language()))
    tree = parser.parse(b"public class Foo { int x; }")

    assert tree.root_node.type == "program"


def test_javalang_parse():
    import javalang

    tree = javalang.parse.parse("public class Foo { int somar(int a, int b) { return a+b; } }")
    classe = tree.types[0]

    assert classe.name == "Foo"
    assert classe.methods[0].name == "somar"


def test_utils_ts_compat():
    from utils.ts_compat import JAVA_LANGUAGE

    assert JAVA_LANGUAGE is not None
