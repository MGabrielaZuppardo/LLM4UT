"""
Coletor de informações de fix para o LLM4UT.

Para cada bug listado em ``data/d4j_fixed_project_list``, gera::

    data/d4j2_fix_info/{Project}_{bug}/buggy_fix_info.json

contendo a lista de métodos alterados pelo patch que corrige o bug.
O formato corresponde ao consumido por ``rq1/generate_prompts_gemma.py``::

    {
      "fixing_changes": [
        {
          "changed_functions": [
            {
              "qualified_names": [
                "<fully.qualified.ClassName>:<methodName>:..."
              ]
            }
          ]
        }
      ]
    }

Fonte dos dados:
    - Patches: ``$D4J_HOME/framework/projects/<Project>/patches/<bug>.src.patch``
    - Checkouts: ``$D4J_HOME/d4j_projects/<Project>_<bug>_fixed/`` (origem do
      arquivo Java pós-fix, usado para mapear linhas modificadas → métodos via
      tree-sitter)

Uso::

    # Um bug isolado (smoke test)
    python tools/collect_fix_info.py --bug Chart_1

    # Tudo
    python tools/collect_fix_info.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

sys.path.extend([".", ".."])

import tree_sitter_java  # noqa: E402
from tree_sitter import Language, Parser  # noqa: E402

# Parser global (tree-sitter >= 0.22 / API moderna).
# Não dependemos do data/configuration.py porque ele usa Language.build_library()
# que foi removido em tree-sitter 0.22+.
JAVA_LANGUAGE = Language(tree_sitter_java.language())
_PARSER = Parser(JAVA_LANGUAGE)


# Configuração: tenta importar do data/configuration.py; senão usa defaults
# baseados em variáveis de ambiente / convenções deste setup.
try:
    from data.configuration import code_base, d4j_home, d4j_proj_base  # type: ignore
except Exception:  # pragma: no cover
    code_base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    d4j_home = os.environ.get("D4J_HOME", os.path.expanduser("~/defects4j"))
    d4j_proj_base = os.environ.get("D4J_PROJ_BASE", f"{d4j_home}/d4j_projects")


PATCHES_DIR_TEMPLATE = "{d4j_home}/framework/projects/{project}/patches"
CHECKOUT_DIR_TEMPLATE = "{base}/{project}_{bug}_fixed"
OUTPUT_DIR = os.path.join(code_base, "data", "d4j2_fix_info")
PROJECT_LIST = os.path.join(code_base, "data", "d4j_fixed_project_list")


# ---------------------------------------------------------------------------
# Parsing de patches unified diff
# ---------------------------------------------------------------------------

@dataclass
class HunkChange:
    """Faixas de linhas cobertas por um hunk, em AMBOS os lados do diff.

    Atenção: os patches do Defects4J (``<id>.src.patch``) são orientados
    FIXED -> BUGGY (aplicá-los reverte o fix). Logo, no unified diff:
        - lado ``---`` / "old" / ``a`` = versão FIXED
        - lado ``+++`` / "new" / ``b`` = versão BUGGY
    Como mapeamos métodos lendo o checkout FIXED, usamos o lado OLD.
    """
    old_start: int   # 1-based (lado FIXED)
    old_count: int
    new_start: int   # 1-based (lado BUGGY)
    new_count: int


@dataclass
class FilePatch:
    """Patch de um arquivo individual."""
    new_path: str                  # caminho relativo após "+++ b/"
    hunks: list[HunkChange]


_HUNK_HEADER_RE = re.compile(
    r"^@@ -(?P<ostart>\d+)(?:,(?P<ocount>\d+))? \+(?P<nstart>\d+)(?:,(?P<ncount>\d+))? @@"
)


def parse_unified_diff(patch_text: str) -> list[FilePatch]:
    """Parser leve de unified diff.

    Retorna a lista de arquivos modificados, cada um com seus hunks no
    referencial do arquivo NOVO. Não tentamos reconstruir o conteúdo, apenas
    saber QUAIS LINHAS mudaram para depois mapeá-las a métodos.
    """
    files: list[FilePatch] = []
    current: FilePatch | None = None

    pending_old_path: str | None = None
    for line in patch_text.splitlines():
        if line.startswith("--- "):
            # "--- a/source/foo/Bar.java", "--- source/foo/Bar.java (revision N)"
            raw = line[4:].strip()
            raw = raw.split("\t")[0].strip()        # remove "(revision N)" após TAB
            raw = re.sub(r"\s+\(revision \d+\)$", "", raw)
            if raw.startswith("a/"):
                raw = raw[2:]
            pending_old_path = raw
        elif line.startswith("+++ "):
            # usamos o caminho do lado FIXED (old); se ausente, cai no new
            raw = line[4:].strip().split("\t")[0].strip()
            raw = re.sub(r"\s+\(revision \d+\)$", "", raw)
            if raw.startswith("b/"):
                raw = raw[2:]
            path = pending_old_path or raw
            if path in ("/dev/null", ""):
                path = raw
            current = FilePatch(new_path=path, hunks=[])
            files.append(current)
            pending_old_path = None
        elif line.startswith("@@") and current is not None:
            m = _HUNK_HEADER_RE.match(line)
            if not m:
                continue
            current.hunks.append(HunkChange(
                old_start=int(m.group("ostart")),
                old_count=int(m.group("ocount") or "1"),
                new_start=int(m.group("nstart")),
                new_count=int(m.group("ncount") or "1"),
            ))

    # filtra arquivos sem hunks (cabeçalho órfão) e arquivos não-Java
    return [f for f in files if f.hunks and f.new_path.endswith(".java")]


# ---------------------------------------------------------------------------
# Mapeamento linha -> método via tree-sitter
# ---------------------------------------------------------------------------

@dataclass
class MethodInfo:
    qualified_class: str   # ex.: org.jfree.chart.Foo
    method_name: str
    param_types: list[str]
    start_line: int        # 1-based (inclusive)
    end_line: int          # 1-based (inclusive)


def _text(node) -> str:
    return node.text.decode("utf-8")


def _package_name(root) -> str:
    for child in root.children:
        if child.type == "package_declaration":
            for sub in child.children:
                if sub.type == "scoped_identifier" or sub.type == "identifier":
                    return _text(sub)
    return ""


_TYPE_DECL_TYPES = (
    "class_declaration", "interface_declaration",
    "enum_declaration", "record_declaration", "annotation_type_declaration",
)
_METHOD_TYPES = ("method_declaration", "constructor_declaration")


def _enclosing_qualified_class(method_node, package: str) -> str | None:
    """Sobe pela árvore a partir de um nó de método e monta o nome qualificado
    do tipo nomeado mais próximo (ignora classes anônimas). Cobre métodos
    dentro de classes, interfaces, enums, constantes de enum, e tipos
    aninhados em qualquer profundidade."""
    names: list[str] = []
    node = method_node.parent
    while node is not None:
        if node.type in _TYPE_DECL_TYPES:
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                names.append(_text(name_node))
        node = node.parent
    if not names:
        return None
    names.reverse()
    qualified = ".".join(names)
    return f"{package}.{qualified}" if package else qualified


def _iter_method_nodes(node) -> Iterable[object]:
    """Varre recursivamente toda a árvore retornando nós de método/construtor."""
    for child in node.children:
        if child.type in _METHOD_TYPES:
            yield child
        # recursa sempre: métodos podem estar dentro de enum_constant,
        # enum_body_declarations, class_body, etc.
        yield from _iter_method_nodes(child)


def _method_param_types(method_node) -> list[str]:
    params = method_node.child_by_field_name("parameters")
    if params is None:
        return []
    types: list[str] = []
    for param in params.children:
        if param.type == "formal_parameter":
            type_node = param.child_by_field_name("type")
            if type_node is not None:
                types.append(_text(type_node))
        elif param.type == "spread_parameter":
            type_node = param.child_by_field_name("type")
            if type_node is not None:
                types.append(_text(type_node) + "...")
    return types


def extract_methods(java_source: str) -> list[MethodInfo]:
    """Extrai todos os métodos (e construtores) de um arquivo Java, com
    package + classe completa e faixas de linhas (1-based). Cobre tipos
    aninhados, enums e métodos dentro de constantes de enum."""
    tree = _PARSER.parse(java_source.encode("utf-8"))
    root = tree.root_node
    pkg = _package_name(root)

    methods: list[MethodInfo] = []
    for member in _iter_method_nodes(root):
        qualified_class = _enclosing_qualified_class(member, pkg)
        if qualified_class is None:
            continue
        name_node = member.child_by_field_name("name")
        if member.type == "constructor_declaration":
            method_name = _text(name_node) if name_node else qualified_class.split(".")[-1]
        else:
            if name_node is None:
                continue
            method_name = _text(name_node)
        methods.append(MethodInfo(
            qualified_class=qualified_class,
            method_name=method_name,
            param_types=_method_param_types(member),
            # tree-sitter dá 0-based; convertemos para 1-based
            start_line=member.start_point[0] + 1,
            end_line=member.end_point[0] + 1,
        ))
    return methods


def methods_touching(hunks: list[HunkChange], methods: list[MethodInfo]) -> list[MethodInfo]:
    """Retorna os métodos cujos intervalos de linha interceptam qualquer hunk.

    Usa o lado OLD do hunk (= versão FIXED), pois lemos o checkout fixed."""
    matched: list[MethodInfo] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for m in methods:
        for h in hunks:
            h_start = h.old_start
            h_end = h.old_start + max(h.old_count - 1, 0)
            if h_end < m.start_line or h_start > m.end_line:
                continue
            key = (m.qualified_class, m.method_name, tuple(m.param_types))
            if key in seen:
                break
            seen.add(key)
            matched.append(m)
            break
    return matched


# ---------------------------------------------------------------------------
# Localização dos arquivos no checkout (resolve raiz das sources)
# ---------------------------------------------------------------------------

def find_source_file(checkout_dir: Path, rel_path_from_patch: str) -> Path | None:
    """Tenta localizar o arquivo Java referenciado pelo patch dentro do
    checkout. O patch usa caminho relativo à raiz do repositório (ex.:
    ``source/org/jfree/...``), que pode ou não coincidir com a raiz das
    sources usadas pelo Defects4J. Tentamos:

    1. ``<checkout>/<rel_path_from_patch>`` (geralmente funciona)
    2. ``find`` por nome de arquivo dentro do checkout como fallback
    """
    direct = checkout_dir / rel_path_from_patch
    if direct.is_file():
        return direct
    # fallback: busca pelo nome
    basename = Path(rel_path_from_patch).name
    for candidate in checkout_dir.rglob(basename):
        # heurística: preferir caminhos cuja terminação bate com o do patch
        if str(candidate).replace("\\", "/").endswith(rel_path_from_patch):
            return candidate
    # último recurso: primeiro match pelo nome
    for candidate in checkout_dir.rglob(basename):
        return candidate
    return None


# ---------------------------------------------------------------------------
# Orquestração por bug
# ---------------------------------------------------------------------------

def collect_for_bug(bug_id: str, *, verbose: bool = False) -> dict | None:
    """Processa um bug (ex.: 'Chart_1') e retorna o dicionário a ser salvo.
    Retorna ``None`` se não houver dados utilizáveis (patch ausente,
    checkout ausente, etc.)."""
    parts = bug_id.split("_")
    if len(parts) < 2:
        print(f"[skip] {bug_id}: id malformado")
        return None
    project, bug_num = parts[0], parts[1]

    patch_path = Path(PATCHES_DIR_TEMPLATE.format(d4j_home=d4j_home, project=project)) / f"{bug_num}.src.patch"
    if not patch_path.is_file():
        print(f"[skip] {bug_id}: patch ausente em {patch_path}")
        return None

    checkout_dir = Path(CHECKOUT_DIR_TEMPLATE.format(base=d4j_proj_base, project=project, bug=bug_num))
    if not checkout_dir.is_dir():
        print(f"[skip] {bug_id}: checkout ausente em {checkout_dir}")
        return None

    patch_text = patch_path.read_text(encoding="utf-8", errors="replace")
    file_patches = parse_unified_diff(patch_text)

    fixing_changes = []
    for fp in file_patches:
        java_file = find_source_file(checkout_dir, fp.new_path)
        if java_file is None:
            if verbose:
                print(f"  [warn] {bug_id}: arquivo {fp.new_path} não localizado no checkout")
            continue
        source = java_file.read_text(encoding="utf-8", errors="replace")
        all_methods = extract_methods(source)
        matched = methods_touching(fp.hunks, all_methods)
        if not matched:
            if verbose:
                print(f"  [info] {bug_id}: nenhum método casa com hunks em {fp.new_path}")
            continue
        qualified_names = [
            f"{m.qualified_class}:{m.method_name}:{','.join(m.param_types)}"
            for m in matched
        ]
        fixing_changes.append({
            "file": fp.new_path,
            "changed_functions": [{"qualified_names": qualified_names}],
        })

    if not fixing_changes:
        print(f"[warn] {bug_id}: nenhum método identificado a partir do patch")
        return None

    return {
        "bug_id": bug_id,
        "project": project,
        "fixing_changes": fixing_changes,
    }


def write_output(bug_id: str, data: dict) -> Path:
    out_dir = Path(OUTPUT_DIR) / bug_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "buggy_fix_info.json"
    out_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_file


def load_project_list() -> list[str]:
    bugs: list[str] = []
    with open(PROJECT_LIST, "r", encoding="utf-8") as fh:
        for line in fh:
            entry = line.strip().rstrip("\r")
            if not entry:
                continue
            # 'Chart_10_fixed' -> 'Chart_10'
            if entry.endswith("_fixed"):
                entry = entry[: -len("_fixed")]
            bugs.append(entry)
    return bugs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--bug", help="processa apenas este bug (ex.: Chart_1)")
    g.add_argument("--all", action="store_true", help="processa todos os bugs do project_list")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--print", action="store_true", help="só imprime o JSON, não escreve em disco")
    args = ap.parse_args()

    if args.bug:
        bugs = [args.bug]
    else:
        bugs = load_project_list()

    ok = 0
    skipped = 0
    for bug in bugs:
        data = collect_for_bug(bug, verbose=args.verbose)
        if data is None:
            skipped += 1
            continue
        if args.print:
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            out = write_output(bug, data)
            if args.verbose:
                print(f"[ok] {bug} -> {out}")
        ok += 1
    print(f"\n=== resumo: {ok} ok / {skipped} skipped / total {len(bugs)} ===")
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
