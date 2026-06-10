"""Camada de compatibilidade tree-sitter.

O código original do LLM4UT foi escrito para tree-sitter <= 0.21, cuja API
mudou bastante a partir da 0.22:

  * ``Language.build_library(...)`` foi removido (a gramática agora vem como
    wheel: ``tree_sitter_java.language()``).
  * ``Parser()`` + ``parser.set_language(lang)`` virou ``Parser(lang)``
    (``set_language`` foi removido).
  * ``language.query(src).captures(node)`` deixou de existir nesse formato;
    agora retorna ``dict[capture_name, list[Node]]`` em vez da antiga lista
    de tuplas ``[(node, capture_name), ...]`` ordenada por posição.

Este módulo expõe wrappers que restauram a API antiga, permitindo que
``utils/java_parser.py`` (e o ``data/configuration.py``) funcionem sem
reescrever os ~26 pontos de uso. Basta:

    # em data/configuration.py
    from utils.ts_compat import JAVA_LANGUAGE

    # em utils/java_parser.py
    from utils.ts_compat import CompatParser as Parser
"""

from __future__ import annotations

import tree_sitter_java
# QueryCursor aparece na documentação do branch principal do tree-sitter,
# mas nunca foi lançado no PyPI (versão mais recente: 0.23.2).
# Na 0.23, Query.captures(node) já retorna dict[str, list[Node]] diretamente,
# tornando QueryCursor desnecessário.
from tree_sitter import Language, Parser, Query

# Gramática Java (API moderna). Fonte única da verdade.
RAW_JAVA_LANGUAGE = Language(tree_sitter_java.language())


class CompatQuery:
    """Reproduz ``query.captures(node)`` retornando a antiga lista de tuplas
    ``[(node, capture_name), ...]`` ordenada por posição no código-fonte."""

    def __init__(self, query: Query):
        self._q = query

    def captures(self, node):
        d = self._q.captures(node)  # dict[str, list[Node]] — API tree-sitter 0.23
        out = []
        for name, nodes in d.items():
            for n in nodes:
                out.append((n, name))
        # ordena por posição: replica o comportamento da API antiga, do qual
        # o java_parser depende para agrupar capturas (groups de 3/4).
        out.sort(key=lambda t: (t[0].start_byte, t[0].end_byte))
        return out


class CompatLanguage:
    """Wrapper de Language que reexpõe ``.query(src)`` no formato antigo."""

    def __init__(self, raw: Language):
        self._raw = raw

    def query(self, source: str) -> CompatQuery:
        return CompatQuery(Query(self._raw, source))

    @property
    def raw(self) -> Language:
        return self._raw

    def __getattr__(self, item):
        # delega qualquer outro atributo para a Language real
        return getattr(self._raw, item)


class CompatParser:
    """Reproduz ``Parser()`` + ``parser.set_language(lang)`` da API antiga.

    Como o projeto é exclusivamente Java, o parser sempre usa a gramática Java
    (``RAW_JAVA_LANGUAGE``); ``set_language`` é aceito mas é no-op."""

    def __init__(self, language=None):
        self._p = Parser(RAW_JAVA_LANGUAGE)

    def set_language(self, language):  # no-op (compat)
        return None

    def parse(self, data, *args, **kwargs):
        return self._p.parse(data, *args, **kwargs)

    def __getattr__(self, item):
        return getattr(self._p, item)


# Objeto compatível usado em todo o projeto no lugar do antigo JAVA_LANGUAGE.
JAVA_LANGUAGE = CompatLanguage(RAW_JAVA_LANGUAGE)
