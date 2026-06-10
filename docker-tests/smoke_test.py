# Executar dentro do container:
#   pytest docker-tests/smoke_test.py -v -s
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_python_version():
    print(f"\nPython: {sys.version}")
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

    print(f"\n  pandas       {pandas.__version__}")
    print(f"  numpy        {numpy.__version__}")
    print(f"  scipy        {scipy.__version__}")
    from importlib.metadata import version
    print(f"  tree-sitter  {version('tree-sitter')}")
    print(f"  openai       {openai.__version__}")


def test_tree_sitter_parse_java():
    from tree_sitter import Language, Parser
    import tree_sitter_java as tsjava

    parser = Parser(Language(tsjava.language()))
    tree = parser.parse(b"public class Foo { int x; }")

    print(f"\n  nó raiz: {tree.root_node.type}")
    assert tree.root_node.type == "program"


def test_javalang_parse():
    import javalang

    tree = javalang.parse.parse("public class Foo { int somar(int a, int b) { return a+b; } }")
    classe = tree.types[0]

    print(f"\n  classe: {classe.name}, método: {classe.methods[0].name}")
    assert classe.name == "Foo"
    assert classe.methods[0].name == "somar"


def test_utils_ts_compat():
    from utils.ts_compat import JAVA_LANGUAGE

    print(f"\n  JAVA_LANGUAGE: {JAVA_LANGUAGE}")
    assert JAVA_LANGUAGE is not None
