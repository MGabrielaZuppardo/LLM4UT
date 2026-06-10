# LLM4UT — Replicação e Extensão do Estudo ASE'24

Este repositório é uma releitura acadêmica do artigo:

> **"On the Evaluation of Large Language Models in Unit Test Generation"**
> Yang et al., ASE 2024 — [Repositório original](https://github.com/LeonYang95/LLM4UT) | Autor: Leon Yang | Licença: Mulan PSL v2

O estudo original avalia LLMs na geração de testes unitários usando o benchmark Defects4J 2.0.
Esta versão replica a metodologia com modelos pós-2024 executados via Ollama (local) e Groq API,
adaptando o pipeline de inferência para ambientes sem GPU de alto desempenho.

**Instituição:** Programa de Pós-Graduação em Ciência da Computação — UFPE
**Disciplina:** Engenharia de Software

---

## Modelos avaliados

| Modelo | Inferência | Status |
|---|---|---|
| `gemma3:4b` | Ollama local | Completo (777/777) |
| `mistral:7b` | Ollama local | Completo (777/777) |
| `deepseek-r1:8b` | Ollama local | Completo (777/777) |
| `deepseek-r1:1.5b` | Ollama local | Completo (777/777) |
| `llama4-scout:17b` | Groq API | Completo (775/777) |

---

## Estrutura do repositório

```
LLM4UT/
├── data/
│   ├── configuration.py          # Configuração central (modelos, caminhos, projetos)
│   ├── outputs/                  # Prompts gerados por generate_prompts.py
│   ├── rq1/
│   │   └── results/              # Resultados da avaliação — um subdiretório por modelo
│   │       ├── gemma3_4b/
│   │       ├── mistral_7b/
│   │       ├── deepseek_r1_8b/
│   │       ├── deepseek_r1_1.5b/
│   │       └── llama4_scout_17b/
│   └── fixed_projects_source/    # Métodos extraídos das classes focais (tree-sitter)
├── rq1/
│   ├── generate_prompts.py       # Geração de prompts a partir dos métodos focais
│   ├── rq1.py                    # Loop de avaliação principal
│   ├── assistant_methods.py      # Compilação iterativa, cobertura e detecção de defeitos
│   └── analyze/
│       ├── compile_and_pass_rates.py   # Calcula CSR (Compilation Success Rate)
│       ├── coverage_rates.py           # Calcula CovL e CovB (cobertura de linha e branch)
│       └── bug_detection_summary.py    # Calcula NDD (Number of Detected Defects)
├── tools/
│   ├── run_ollama_inference.py   # Inferência via Ollama (modelos locais, idempotente)
│   ├── run_groq_inference.py     # Inferência via Groq / Gemini / OpenRouter API
│   ├── collect_fix_info.py       # Extrai métodos alterados por cada correção de bug
│   ├── collect_source_methods.py # Extrai todos os métodos das classes focais
│   └── monitor.py                # Monitor de progresso da avaliação em tempo real
├── utils/
│   └── ts_compat.py              # Camada de compatibilidade tree-sitter 0.22+
├── tests/
│   ├── conftest.py
│   ├── units/                    # Testes unitários por módulo
│   └── integration/              # Testes de integração do pipeline
├── baselines/                    # Scripts EvoSuite (baseline do paper original)
├── running_examples/             # Exemplos dos formatos de prompt (CL, NL, CoT, RAG)
└── requirements.txt
```

---

## Pré-requisitos

### 1. Defects4J 2.0 (Linux / WSL obrigatório)

```bash
git clone https://github.com/rjust/defects4j.git ~/defects4j
cd ~/defects4j && ./init.sh
export PATH=$PATH:~/defects4j/framework/bin
defects4j info -p Lang   # verifica se a instalação funcionou
```

Faça o checkout de todos os projetos (fixed + buggy) — aproximadamente 1.600 checkouts:

```bash
# Exemplo para um bug
defects4j checkout -p Lang -v 1f -w ~/defects4j/d4j_projects/Lang_1/fixed
defects4j checkout -p Lang -v 1b -w ~/defects4j/d4j_projects/Lang_1/buggy
```

A estrutura esperada é `{Bug_id}/fixed` e `{Bug_id}/buggy` dentro de `d4j_proj_base`.

Adicione as dependências de compilação dos testes em `~/defects4j/framework/projects/lib/`:

- JUnit 4
- Mockito 5
- PowerMock 2
- Hamcrest 2.1

### 2. Python

```bash
pip install -r requirements.txt
```

Para rodar os testes que dependem de `tree-sitter-java` (requer Linux/WSL):

```bash
python3 -m venv venv && source venv/bin/activate
pip install pytest tree-sitter tree-sitter-java
```

### 3. Ollama (inferência local)

Instale a partir de [ollama.com](https://ollama.com) e baixe os modelos:

```bash
ollama pull gemma3:4b
ollama pull mistral:7b
ollama pull deepseek-r1:8b
ollama pull deepseek-r1:1.5b
```

### 4. Groq API (para modelos maiores via nuvem)

Crie um arquivo `.env` na raiz do projeto com sua chave:

```
GROQ_API_KEY=gsk_...
```

Obtenha uma chave gratuita em [console.groq.com](https://console.groq.com).

---

## Configuração

Edite `data/configuration.py` conforme seu ambiente:

```python
d4j_home      = "/home/usuario/defects4j"
d4j_proj_base = f"{d4j_home}/d4j_projects"
output_dir    = "data/rq1/results"      # diretório de saída da avaliação

target_models = [
    "gemma3_4b",
    "mistral_7b",
    "deepseek_r1_8b",
    "deepseek_r1_1.5b",
    "llama4_scout_17b",
]

projects  = ["Lang", "Math", "Cli", "Chart", "Closure", ...]  # projetos Defects4J
formats   = ["comment"]   # "comment" = Code-Language-Description
                          # "natural" = Natural-Language-Description
strategies = ["extend"]
ablations  = ["full"]
```

---

## Execução passo a passo

### Passo 1 — Extrair informações dos bugs e métodos focais

```bash
# Extrai quais métodos foram alterados em cada correção de bug
python tools/collect_fix_info.py

# Extrai todos os métodos das classes focais (entrada para geração de prompts)
python tools/collect_source_methods.py
```

Saída: `data/d4j2_fix_info/{Bug_id}/buggy_fix_info.json` e `data/fixed_projects_source/`.

### Passo 2 — Gerar prompts

```bash
python rq1/generate_prompts.py
```

Saída: `data/outputs/{model}_{format}_{strategy}_{ablation}.jsonl`

Cada linha do JSONL contém os campos necessários para a inferência, incluindo `prompt`, `id`, `project`, `method_signature`, `is_public`, `format`, `strategy` e `ablation`.

### Passo 3 — Rodar inferência

**Ollama (modelos locais):**

```bash
python tools/run_ollama_inference.py \
  --input  data/outputs/gemma3_4b_comment_extend_full.jsonl \
  --output data/outputs/gemma3_4b_comment_extend_full.jsonl \
  --model  gemma3:4b
```

O script é idempotente: registros já processados são ignorados em caso de retomada.

**Groq API (modelos via nuvem):**

```bash
python tools/run_groq_inference.py \
  --input   data/outputs/llama4_scout_17b_comment_extend_full.jsonl \
  --output  data/outputs/llama4_scout_17b_comment_extend_full.jsonl \
  --model   meta-llama/llama-4-scout-17b-16e-instruct \
  --api-key gsk_... \
  --temperature 0.0
```

Para múltiplas chaves com rotação automática ao atingir o limite diário:

```bash
python tools/run_groq_inference.py \
  --input   data/outputs/llama4_scout_17b_comment_extend_full.jsonl \
  --output  data/outputs/llama4_scout_17b_comment_extend_full.jsonl \
  --model   meta-llama/llama-4-scout-17b-16e-instruct \
  --api-key gsk_chave1,gsk_chave2,gsk_chave3
```

**Monitorar progresso em tempo real:**

```bash
python tools/monitor.py
```

### Passo 4 — Avaliação RQ1 (compilação, cobertura e detecção de defeitos)

```bash
# Avalia todos os modelos e projetos configurados em data/configuration.py
python rq1/rq1_starter.py
```

O script percorre cada combinação `(modelo, projeto, formato, estratégia, ablação)`, compila iterativamente os testes gerados e coleta métricas via JaCoCo. Resultados são salvos em `data/rq1/results/{modelo}/{Projeto}_{formato}_{estratégia}_{ablação}.jsonl`.

Para avaliar um projeto específico:

```bash
python rq1/rq1.py \
  --model gemma3_4b \
  --project Lang \
  --format comment \
  --strategy extend \
  --ablation full
```

### Passo 5 — Análise dos resultados

```bash
# CSR — Compilation Success Rate
python rq1/analyze/compile_and_pass_rates.py

# CovL / CovB — cobertura de linha e branch (requer execução com JaCoCo concluída)
python rq1/analyze/coverage_rates.py

# NDD — Number of Detected Defects
python rq1/analyze/bug_detection_summary.py
```

Critério de detecção de defeito: o teste gerado **passa na versão corrigida** e **falha na versão com bug**.

---

## Testes do pipeline

```bash
# Testes unitários (rodam no Windows/Linux sem dependências externas)
python -m pytest tests/units/test_ollama_inference.py
python -m pytest tests/units/test_groq_inference.py
python -m pytest tests/units/test_monitor.py
python -m pytest tests/integration/

# Testes que requerem tree-sitter-java (Linux/WSL com venv ativado)
source venv/bin/activate
python3 -m pytest tests/units/test_ts_compat.py tests/units/test_collect.py -v
```

---

## Métricas avaliadas

| Sigla | Nome | Descrição |
|---|---|---|
| CSR | Compilation Success Rate | % de testes que compilam com sucesso |
| CovL | Line Coverage | % de linhas do método focal cobertas |
| CovB | Branch Coverage | % de branches do método focal cobertos |
| NDD | Number of Detected Defects | Quantidade de bugs detectados pelo conjunto de testes |

---

## Referência

```bibtex
@inproceedings{yang2024llm4ut,
  title     = {On the Evaluation of Large Language Models in Unit Test Generation},
  author    = {Yang, Lin and others},
  booktitle = {Proceedings of the 39th IEEE/ACM International Conference on
               Automated Software Engineering (ASE)},
  year      = {2024}
}
```
