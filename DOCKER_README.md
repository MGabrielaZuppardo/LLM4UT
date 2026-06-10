# Docker — Guia de Uso

Este documento explica como construir a imagem Docker do projeto e verificar se o ambiente Python e Java 11 estão funcionando corretamente.

---

## Pré-requisitos

- [Docker](https://docs.docker.com/get-docker/) ≥ 24
- [Docker Compose](https://docs.docker.com/compose/) ≥ 2.20 (incluso no Docker Desktop)

---

## 1. Construir a imagem

```bash
# Na raiz do projeto
docker compose build
```

Ou sem o Compose:

```bash
docker build -t llm4ut:latest .
```

> **Tempo estimado:** 5–10 min (baixa a imagem base + instala ~60 pacotes Python).

---

## 2. Iniciar o container

```bash
docker compose run --rm llm4ut bash
```

Você entrará num shell interativo com o projeto montado em `/app` e o `PYTHONPATH` já configurado.

---

## 3. Verificar o ambiente Python

Execute o teste de smoke dentro do container:

```bash
# Dentro do container (prompt: root@<id>:/app#)
pytest docker-tests/smoke_test.py -v
```

**Saída esperada:**

```
docker-tests/smoke_test.py::test_python_version         PASSED
docker-tests/smoke_test.py::test_pacotes_instalados     PASSED
docker-tests/smoke_test.py::test_tree_sitter_parse_java PASSED
docker-tests/smoke_test.py::test_javalang_parse         PASSED
docker-tests/smoke_test.py::test_utils_ts_compat        PASSED

5 passed in X.XXs
```

O que cada teste valida:

| Teste | O que verifica |
|---|---|
| `test_python_version` | Python ≥ 3.9 instalado |
| `test_pacotes_instalados` | pandas, numpy, scipy, javalang, tree-sitter, openai, groq, ollama importáveis |
| `test_tree_sitter_parse_java` | tree-sitter-java faz parse de código Java |
| `test_javalang_parse` | javalang extrai classe e método de um snippet Java |
| `test_utils_ts_compat` | módulo `utils/ts_compat.py` do projeto funciona |

---

## 4. Verificar o ambiente Java 11

### 4a. Compilar

```bash
# Dentro do container
javac docker-tests/HelloTest.java -d docker-tests/
```

### 4b. Executar

```bash
java -cp docker-tests HelloTest
```

**Saída esperada:**

```
Java version: 11.x.x
Todos os testes passaram!
```

O que o teste valida:

| Verificação | O que testa |
|---|---|
| `java.version` | Java 11+ instalado |
| `strip()` / `repeat()` / `isBlank()` | métodos introduzidos no Java 11 |
| `var` | inferência de tipo local (Java 10+) |
| `somar(7, 8)` | lógica aritmética básica |

---

## 5. Verificar a versão do Java

```bash
java -version
javac -version
```

Saída esperada: `openjdk version "11.x.x" ...`

---

## 6. Executar a suite de testes do projeto

```bash
# Testes unitários (não precisam do Defects4J)
pytest tests/units/ -v

# Todos os testes
pytest tests/ -v
```

---

## 7. Uso do container para o projeto

```bash
# Gerar prompts
python rq1/generate_prompts.py

# Inferência local com Ollama (serviço Ollama deve estar acessível)
python tools/run_ollama_inference.py \
  --input data/outputs/gemma3_4b_comment_extend_full.jsonl \
  --model gemma3:4b

# Análise de resultados
python rq1/analyze/compile_and_pass_rates.py
python rq1/analyze/coverage_rates.py
python rq1/analyze/bug_detection_summary.py
```

---

## 8. Estrutura dos arquivos Docker

```
LLM4UT/
├── Dockerfile               # Imagem: python:3.9-slim + OpenJDK 11
├── docker-compose.yml       # Serviço llm4ut com volume montado
├── docker-requirements.txt  # Pacotes pip (versão leve, sem torch)
├── .dockerignore            # Exclui data/, logs/, cache do contexto de build
├── DOCKER_README.md         # Este documento
└── docker-tests/            # Testes de smoke do ambiente Docker
    ├── smoke_test.py          # Smoke test Python
    └── HelloTest.java         # Smoke test Java 11
```

---

## 9. Notas

- **torch/transformers não estão incluídos** por padrão para manter a imagem ~2 GB em vez de ~7 GB. Para habilitá-los, descomente as linhas correspondentes em [docker-requirements.txt](docker-requirements.txt) e reconstrua.
- O **Defects4J** (ferramenta externa Java) não está incluído na imagem. Para a pipeline completa de avaliação, siga as instruções do [README.md](README.md) principal.
- O volume `.:/app` monta o código no container e `./data:/app/data` monta os dados separadamente — edições feitas fora refletem imediatamente dentro, sem necessidade de rebuild.
