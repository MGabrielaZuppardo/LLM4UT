"""
Coletor de métodos-fonte para o LLM4UT.

Para cada bug listado em ``data/d4j_fixed_project_list``, gera::

    data/fixed_projects_source/{Project}/{Project}_{bug}_fixed.jsonl

Cada linha do JSONL é ``{chave: item}`` onde ``item`` contém todas as chaves
consumidas por ``rq1/generate_prompts_gemma.py`` / ``PromptFormatter.apply_format``:

    source:source_method_code_format     (str)  código do método focal
    source:source_method_name            (str)  nome do método
    source:source_method_signature       (str)  "<package>#<NestedClass>#<method>"
    source:source_other_method_signature (list) assinaturas dos demais métodos
    content:source_class_name            (str)  nome simples da classe
    content:source_class_code_format     (str)  código completo da classe
    content:source_class_code_imports    (list) imports do arquivo
    content:source_class_constructors    (list) código dos construtores
    content:parameter_class_signature    (list) [] (resolvido graciosamente p/ tipos crus)
    content:parameter_class_constructors (list) []
    source_class_fields                  (list) declarações de campos (str)
    extra:project_name                   (str)  "<Project>_<bug>_fixed"

OTIMIZAÇÃO: como ``data/d4j2_fix_info/<bug>/buggy_fix_info.json`` já indica as
classes focais de cada bug, extraímos métodos APENAS dessas classes (o
consumidor filtra para os métodos focais de qualquer forma). Isso reduz
drasticamente o volume de saída e o tempo de execução. Use ``--all-classes``
para extrair todos os métodos de todas as classes do projeto (modo fiel ao
intento original, porém muito mais pesado).

Uso::

    python tools/collect_source_methods.py --bug Chart_1
    python tools/collect_source_methods.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

sys.path.extend([".", ".."])

import tree_sitter_java  # noqa: E402
from tree_sitter import Language, Parser  # noqa: E402

JAVA_LANGUAGE = Language(tree_sitter_java.language())
_PARSER = Parser(JAVA_LANGUAGE)

try:
    from data.configuration import code_base, d4j_proj_base  # type: ignore
except Exception:  # pragma: no cover
    code_base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    _d4j_home = os.environ.get("D4J_HOME", os.path.expanduser("~/defects4j"))
    d4j_proj_base = os.environ.get("D4J_PROJ_BASE", f"{_d4j_home}/d4j_projects")

FIX_INFO_DIR = os.path.join(code_base, "data", "d4j2_fix_info")
OUTPUT_DIR = os.path.join(code_base, "data", "fixed_projects_source")
PROJECT_LIST = os.path.join(code_base, "data", "d4j_fixed_project_list")
CHECKOUT_DIR_TEMPLATE = "{base}/{project}_{bug}_fixed"


_TYPE_DECL_TYPES = (
    "class_declaration", "interface_declaration",
    "enum_declaration", "record_declaration",
)
_METHOD_TYPES = ("method_declaration", "constructor_declaration")


# ---------------------------------------------------------------------------
# helpers tree-sitter
# ---------------------------------------------------------------------------

def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _package_name(root, src: bytes) -> str:
    for child in root.children:
        if child.type == "package_declaration":
            for sub in child.children:
                if sub.type in ("scoped_identifier", "identifier"):
                    return _text(sub, src)
    return ""


def _imports(root, src: bytes) -> list[str]:
    out = []
    for child in root.children:
        if child.type == "import_declaration":
            out.append(_text(child, src))
    return out


def _iter_type_decls(node) -> Iterable[object]:
    """Todos os tipos nomeados (classe/interface/enum/record), em qq profundidade."""
    for child in node.children:
        if child.type in _TYPE_DECL_TYPES:
            yield child
        yield from _iter_type_decls(child)


def _nested_class_name(type_node, src: bytes) -> str:
    """Nome qualificado do tipo dentro do arquivo (sem package), com '.' entre
    níveis aninhados. Ex.: 'Outer.Inner'."""
    names = []
    node = type_node
    while node is not None:
        if node.type in _TYPE_DECL_TYPES:
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                names.append(_text(name_node, src))
        node = node.parent
    names.reverse()
    return ".".join(names)


def _direct_members(type_node):
    """Itera membros diretos do corpo de um tipo (lida com class_body e
    enum_body / enum_body_declarations)."""
    body = type_node.child_by_field_name("body")
    if body is None:
        return
    for child in body.children:
        if child.type == "enum_body_declarations":
            for sub in child.children:
                yield sub
        else:
            yield child


def _param_types(method_node, src: bytes) -> list[str]:
    params = method_node.child_by_field_name("parameters")
    if params is None:
        return []
    types = []
    for p in params.children:
        if p.type in ("formal_parameter", "spread_parameter"):
            t = p.child_by_field_name("type")
            if t is not None:
                suffix = "..." if p.type == "spread_parameter" else ""
                types.append(_text(t, src) + suffix)
    return types


def _method_header(method_node, src: bytes) -> str:
    """Assinatura textual = tudo antes do corpo. Para métodos abstratos/sem
    corpo, usa o texto inteiro sem o ';' final."""
    body = method_node.child_by_field_name("body")
    if body is not None:
        header = src[method_node.start_byte:body.start_byte].decode("utf-8", errors="replace")
    else:
        header = _text(method_node, src)
    return " ".join(header.split()).rstrip("{").strip()


def _field_declarations(type_node, src: bytes) -> list[str]:
    out = []
    for member in _direct_members(type_node):
        if member.type == "field_declaration":
            out.append(" ".join(_text(member, src).split()))
    return out


def _constructors(type_node, src: bytes) -> list[str]:
    out = []
    for member in _direct_members(type_node):
        if member.type == "constructor_declaration":
            out.append(_text(member, src))
    return out


# ---------------------------------------------------------------------------
# construção dos registros
# ---------------------------------------------------------------------------

def build_records_for_class(type_node, *, package: str, imports: list[str],
                            project_full_id: str, src: bytes) -> list[dict]:
    """Gera um registro por método (incl. construtores) da classe ``type_node``."""
    simple_name = ""
    nm = type_node.child_by_field_name("name")
    if nm is not None:
        simple_name = _text(nm, src)
    nested_name = _nested_class_name(type_node, src)
    class_code = _text(type_node, src)
    fields = _field_declarations(type_node, src)
    constructors = _constructors(type_node, src)

    # coleta métodos diretos da classe
    methods = [m for m in _direct_members(type_node) if m.type in _METHOD_TYPES]

    records = []
    for m in methods:
        name_node = m.child_by_field_name("name")
        if m.type == "constructor_declaration":
            method_name = _text(name_node, src) if name_node else simple_name
        else:
            if name_node is None:
                continue
            method_name = _text(name_node, src)

        method_code = _text(m, src)
        signature = f"{package}#{nested_name}#{method_name}"

        # demais métodos da classe (assinaturas textuais), exceto o atual
        other_sigs = [
            _method_header(o, src) for o in methods if o is not m
        ]

        records.append({
            "source:source_method_code_format": method_code,
            "source:source_method_name": method_name,
            "source:source_method_signature": signature,
            "source:source_other_method_signature": other_sigs,
            "content:source_class_name": simple_name,
            "content:source_class_code_format": class_code,
            "content:source_class_code_imports": imports,
            "content:source_class_constructors": constructors,
            "content:parameter_class_signature": [],
            "content:parameter_class_constructors": [],
            "source_class_fields": fields,
            "extra:project_name": project_full_id,
            # metadado interno (não usado pelo formatter)
            "_param_types": _param_types(m, src),
        })
    return records


# ---------------------------------------------------------------------------
# localização de arquivos focais
# ---------------------------------------------------------------------------

def load_focal_classes(bug_id: str) -> set[str]:
    """Lê o fix_info do bug e retorna o conjunto de classes focais (FQN)."""
    info_path = Path(FIX_INFO_DIR) / bug_id / "buggy_fix_info.json"
    if not info_path.is_file():
        return set()
    info = json.loads(info_path.read_text(encoding="utf-8"))
    classes: set[str] = set()
    for change in info.get("fixing_changes", []):
        for cf in change.get("changed_functions", []):
            for qn in cf.get("qualified_names", []):
                fqn_class = qn.split(":")[0]
                classes.add(fqn_class)
    return classes


def fqn_to_relpath(fqn_class: str) -> str:
    """org.jfree.Foo.Inner -> org/jfree/Foo.java (top-level class do arquivo)."""
    # remove níveis aninhados: o arquivo tem o nome da classe top-level
    # heurística: o arquivo é o primeiro segmento com inicial maiúscula
    parts = fqn_class.split(".")
    for i, p in enumerate(parts):
        if p[:1].isupper():
            return "/".join(parts[: i + 1]) + ".java"
    return fqn_class.replace(".", "/") + ".java"


def find_class_file(checkout_dir: Path, fqn_class: str) -> Path | None:
    rel = fqn_to_relpath(fqn_class)
    # tentativa direta em raízes comuns
    for root in ("src/main/java", "src/java", "src", "source", "gson/src/main/java", "."):
        cand = checkout_dir / root / rel
        if cand.is_file():
            return cand
    # fallback: busca pelo basename
    basename = Path(rel).name
    for cand in checkout_dir.rglob(basename):
        if str(cand).replace("\\", "/").endswith(rel):
            return cand
    for cand in checkout_dir.rglob(basename):
        return cand
    return None


# ---------------------------------------------------------------------------
# orquestração
# ---------------------------------------------------------------------------

def collect_for_bug(bug_id: str, *, verbose: bool = False) -> list[dict] | None:
    parts = bug_id.split("_")
    if len(parts) < 2:
        return None
    project, bug_num = parts[0], parts[1]
    project_full_id = f"{project}_{bug_num}_fixed"

    checkout_dir = Path(CHECKOUT_DIR_TEMPLATE.format(base=d4j_proj_base, project=project, bug=bug_num))
    if not checkout_dir.is_dir():
        print(f"[skip] {bug_id}: checkout ausente")
        return None

    focal_classes = load_focal_classes(bug_id)
    if not focal_classes:
        print(f"[skip] {bug_id}: sem fix_info / sem classes focais")
        return None

    all_records: list[dict] = []
    seen_files: set[Path] = set()
    for fqn in focal_classes:
        java_file = find_class_file(checkout_dir, fqn)
        if java_file is None:
            if verbose:
                print(f"  [warn] {bug_id}: arquivo de {fqn} não encontrado")
            continue
        if java_file in seen_files:
            continue
        seen_files.add(java_file)

        src = java_file.read_bytes()
        tree = _PARSER.parse(src)
        root = tree.root_node
        package = _package_name(root, src)
        imports = _imports(root, src)

        for type_node in _iter_type_decls(root):
            recs = build_records_for_class(
                type_node, package=package, imports=imports,
                project_full_id=project_full_id, src=src,
            )
            all_records.extend(recs)

    if not all_records:
        print(f"[warn] {bug_id}: nenhum método extraído")
        return None
    return all_records


def write_output(bug_id: str, records: list[dict]) -> Path:
    project = bug_id.split("_")[0]
    out_dir = Path(OUTPUT_DIR) / project
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{bug_id}_fixed.jsonl"
    with open(out_file, "w", encoding="utf-8") as fh:
        for rec in records:
            key = rec["source:source_method_signature"]
            fh.write(json.dumps({key: rec}, ensure_ascii=False) + "\n")
    return out_file


def load_project_list() -> list[str]:
    bugs = []
    with open(PROJECT_LIST, "r", encoding="utf-8") as fh:
        for line in fh:
            entry = line.strip().rstrip("\r")
            if not entry:
                continue
            if entry.endswith("_fixed"):
                entry = entry[: -len("_fixed")]
            bugs.append(entry)
    return bugs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--bug", help="processa apenas este bug (ex.: Chart_1)")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--print", action="store_true", help="imprime 1ª linha do JSONL e não grava")
    args = ap.parse_args()

    bugs = [args.bug] if args.bug else load_project_list()

    ok = skipped = 0
    total_records = 0
    for bug in bugs:
        records = collect_for_bug(bug, verbose=args.verbose)
        if not records:
            skipped += 1
            continue
        total_records += len(records)
        if args.print:
            key = records[0]["source:source_method_signature"]
            print(json.dumps({key: records[0]}, ensure_ascii=False, indent=2)[:2000])
            print(f"... ({len(records)} métodos extraídos para {bug})")
        else:
            out = write_output(bug, records)
            if args.verbose:
                print(f"[ok] {bug} -> {out} ({len(records)} métodos)")
        ok += 1
    print(f"\n=== resumo: {ok} ok / {skipped} skipped / total {len(bugs)} | {total_records} registros ===")
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
