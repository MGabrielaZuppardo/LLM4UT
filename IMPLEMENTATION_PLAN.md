# Plano de Implementação — LLM4UT com Small LLMs

## Objetivo

Reproduzir o paper LLM4UT usando modelos pequenos (≤8B parâmetros), com um pipeline
simples de rodar, parametrizável por modelo e subconjunto de bugs, sem precisar refazer
checkouts a cada execução.

---

## Decisões de arquitetura

### Por que NÃO usar imagens Docker com checkouts dentro

Cada checkout do Defects4J é ~100MB de código fonte. 835 bugs × 2 versões = 1670
checkouts = ~167GB. Uma imagem com tudo isso é impraticável para distribuir ou versionar.

O próprio Dockerfile do Defects4J documenta essa decisão:
> *"Running init.sh during build would bake those files into the image layer,
> making the image huge and non-persistent across rebuilds."*

### Solução adotada: imagem leve + volume Docker persistente

- **Imagem** (`llm4ut:latest`): Java 11 + Python 3.9 + Defects4J + dependências do projeto. ~2GB. Distribuível.
- **Volume** (`d4j_projects`): checkouts persistidos fora da imagem. Populado uma única vez por um script idempotente.
- **Checkout script**: paralelo (usa os 12 núcleos disponíveis), pula o que já existe. ~20-30 min na primeira vez.

**Vantagem adicional — portabilidade real:** como o volume é uma pasta comum no host,
você pode zipá-la, copiar para outro PC e o pipeline roda imediatamente sem refazer
nenhum checkout. Isso é mais prático do que qualquer imagem Docker: a imagem você
distribui pelo registry, mas o volume (os dados) você distribui como um arquivo.

```bash
# Exportar (PC origem)
tar -czf d4j_projects.tar.gz /caminho/para/d4j_projects/

# Importar (PC destino)
tar -xzf d4j_projects.tar.gz
# ajustar D4J_PROJ_BASE em configuration.py para o novo caminho
```

---

## Simplificações

Regra: **só remover adições do fork que sejam desnecessárias**. Tudo que veio do
projeto original permanece intocado, mesmo que não seja usado diretamente.

### Adições do fork a remover

| O que remover | Por quê é desnecessário |
|---|---|
| `data/d4j2_fix_info/` | Adicionado pelo fork. Só necessário para regenerar prompts do zero — os prompts já estão em `d4j_assistant.jsonl` |
| `data/fixed_projects_source/` | Idem |
| `tools/collect_fix_info.py` | Script que gerou o `d4j2_fix_info/` — se o dado em si não fica, o script também não |
| `tools/collect_source_methods.py` | Script que gerou o `fixed_projects_source/` — mesmo motivo |
| `.idea/` | Configuração do IntelliJ, não pertence ao repo |
| `.pytest_cache/` | Cache local de testes, deve estar no `.gitignore` |

### Adições do fork a manter

Tudo o mais que o fork adicionou é útil e fica:

| O que manter | Por quê |
|---|---|
| `Dockerfile`, `docker-compose.yml`, etc. | Base do nosso plano de infra |
| `utils/ts_compat.py` | Corrige incompatibilidade com tree-sitter >= 0.22 — sem isso o pipeline não roda |
| `tools/run_ollama_inference.py` | Inferência local — necessário |
| `tools/run_groq_inference.py` | Inferência via API — necessário |
| `tools/monitor.py` | Acompanhamento do progresso em tempo real |
| `tools/checkout_buggy.sh` | Base do script de checkout — será estendido |
| `tools/show_results.py` | Útil para visualizar resultados |
| `rq1/analyze/statistical_analysis.py` | Análise estatística dos resultados |
| `tests/` | Suite de testes do ambiente |
| `data/outputs/*.jsonl` | Resultados de inferência já obtidos para 5 modelos — dados valiosos |
| `data/rq1/results/` | Resultados de avaliação já obtidos — dados valiosos |
| `data/prompts/rq1/Gemma-7b-it_comment_extend_full.jsonl` | Prompts no formato Gemma já gerados |

### O que NÃO tocar (veio do projeto original)

`rq1/generate_prompts*.py`, `rq3/`, `baselines/`, `running_examples/` e demais
arquivos do projeto original permanecem — mesmo sem uso direto no objetivo atual.

---

## Estrutura alvo

```
LLM4UT/
├── scripts/
│   ├── setup_defects4j.sh      ← clona D4J e roda init.sh (uma vez)
│   ├── checkout.sh             ← faz os checkouts no volume (idempotente, paralelo)
│   └── run_pipeline.sh         ← wrapper conveniente para o pipeline completo
├── docker/
│   ├── Dockerfile              ← imagem base (Java 11 + Python + D4J + deps)
│   └── docker-compose.yml      ← sobe llm4ut + ollama, monta volume d4j_projects
├── tools/
│   ├── run_ollama_inference.py
│   ├── run_groq_inference.py
│   └── monitor.py
├── rq1/
│   ├── rq1_starter.py          ← aceita --projects --bugs --model
│   ├── rq1.py
│   └── assistant_methods.py
├── utils/
├── data/
└── IMPLEMENTATION_PLAN.md      ← este arquivo
```

---

## O que falta implementar

### 1. `scripts/setup_defects4j.sh`

Responsabilidade:
- Clonar o Defects4J em `../defects4j` (relativo ao projeto) se não existir
- Rodar `init.sh` do D4J (baixa os `project_repos/` e ferramentas externas)
- Exige: git, perl, Java 8 ou 11 instalados no host

```bash
# Comportamento esperado
./scripts/setup_defects4j.sh
# → clona https://github.com/rjust/defects4j em ../defects4j
# → executa ../defects4j/init.sh
# → imprime path para usar na configuration.py
```

### 2. `scripts/checkout.sh`

Responsabilidade:
- Ler `data/d4j_fixed_project_list`
- Para cada bug, fazer checkout da versão buggy E fixed no volume `d4j_projects`
- Pular se o diretório já existir (idempotente)
- Rodar em paralelo (N workers configurável, padrão = nproc)
- Aceitar filtro de projeto: `./scripts/checkout.sh --projects Chart Lang`

Baseado em `tools/checkout_buggy.sh` (já existe no fork) — estender para cobrir
também a versão fixed e adicionar o paralelismo e o filtro.

### 3. `docker/Dockerfile` (atualizar o existente)

O Dockerfile atual (`Dockerfile` na raiz) já tem Java 11 + Python. Precisa:
- Adicionar Perl (dependência do D4J para rodar os testes)
- Montar o D4J via volume (não copiar para dentro da imagem)
- Ajustar `PYTHONPATH` e variáveis de ambiente do D4J

### 4. `docker/docker-compose.yml` (atualizar o existente)

O `docker-compose.yml` atual já tem os serviços `llm4ut` e `ollama`. Precisa:
- Adicionar o volume `d4j_projects` apontando para o diretório de checkouts do host
- Adicionar o volume do D4J (`../defects4j:/defects4j`)
- Passar as variáveis `D4J_HOME` e `D4J_PROJ_BASE` para o container

### 5. Parametrização do pipeline (`rq1/rq1_starter.py`)

Adicionar argumentos CLI:
```
--projects  Chart Lang Math     (default: todos da configuration.py)
--bugs      Chart_10 Chart_11   (default: todos do projeto)
--model     gemma3_4b           (default: todos de configuration.target_models)
--limit     10                  (pegar os primeiros N bugs — útil para smoke test)
```

### 6. `data/configuration.py` (ajustar paths)

Substituir os paths hardcoded de `gabriela_zuppardo` por variáveis de ambiente
com fallbacks sensatos:

```python
d4j_home = os.environ.get("D4J_HOME", os.path.abspath("../defects4j"))
d4j_proj_base = os.environ.get("D4J_PROJ_BASE", f"{d4j_home}/d4j_projects")
```

---

## Fluxo completo após implementação

```
# Uma vez (setup do ambiente):
./scripts/setup_defects4j.sh
./scripts/checkout.sh --projects Chart          # ou sem filtro para tudo

# Inferência (gera os outputs do LLM):
python tools/run_ollama_inference.py \
    --input  data/d4j_assistant.jsonl \
    --output data/outputs/gemma3_4b_comment_extend_full.jsonl \
    --model  gemma3:4b \
    --projects Chart                            # subset opcional

# Avaliação:
python rq1/rq1_starter.py \
    --projects Chart \
    --model gemma3_4b \
    --limit 10                                  # smoke test com 10 bugs

# Monitoramento (em outro terminal):
python tools/monitor.py
```

---

## Modelos alvo (small LLMs ≤ 8B)

Rodar via Ollama local:

| Modelo | Parâmetros | Tag Ollama |
|---|---|---|
| Gemma 3 4B | 4B | `gemma3:4b` |
| Mistral 7B | 7B | `mistral:7b` |
| DeepSeek-R1 | 8B | `deepseek-r1:8b` |
| LLaMA 3.1 | 8B | `llama3.1:8b` |
| Qwen 3 | 4B / 8B | `qwen3:4b` |

Rodar via Groq API (gratuito até certo limite):
- `gemma2-9b-it`, `llama3-8b-8192`

---

## Notas de contexto

- O fork já tem `data/d4j_assistant.jsonl` com **1.243 prompts prontos** — não é necessário
  regenerar a partir do Defects4J.
- Os 1.243 prompts cobrem **710 bugs únicos** de 835 totais (os demais foram filtrados
  por não terem métodos públicos).
- Um bug pode ter múltiplos métodos focais (ex: `Chart_14` tem 8), daí mais entradas que bugs.
- A estratégia usada nos prompts do `d4j_assistant.jsonl` é `extend` + `comment` + `full`.
  Para usar outra estratégia seria necessário regenerar os prompts (aí sim o `d4j2_fix_info`
  seria necessário).
