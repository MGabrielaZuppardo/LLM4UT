"""
Inferência via Groq API para os prompts gerados pelo LLM4UT.

Lê o JSONL de prompts (saída de ``rq1/generate_prompts_gemma.py``), envia cada
prompt para um modelo hospedado na Groq e grava um JSONL acrescentando a chave
``completion`` (exigida pelo pipeline de avaliação do LLM4UT).

Comportamento para estratégia "extend":
  O prompt Gemma termina com ``<start_of_turn>model\\n[cabeçalho parcial da
  classe de teste]``.  O script extrai esse prefixo, envia como assistente
  prefill na chamada de chat (Groq suporta esse padrão), e reconstrói a
  completion completa = prefixo + resposta da API.  Assim o pipeline de
  avaliação (rq1.py → assistant_methods.py) recebe um arquivo Java completo.

Idempotente/retomável: pula prompts cujo (id, method_signature) já estiver no
arquivo de saída com a chave ``completion`` preenchida.

Para listar os modelos disponíveis na sua conta Groq::

    curl -s https://api.groq.com/openai/v1/models \\
         -H "Authorization: Bearer $GROQ_API_KEY" | python -m json.tool

Uso típico::

    # teste rápido com 3 prompts
    python tools/run_groq_inference.py \\
        --input  data/prompts/rq1/Gemma-7b-it_comment_extend_full.jsonl \\
        --output data/outputs/gemma-3-12b-it_comment_extend_full.jsonl \\
        --model  gemma-3-12b-it \\
        --api-key gsk_... \\
        --limit 3

    # tudo
    GROQ_API_KEY=gsk_... python tools/run_groq_inference.py \\
        --input  data/prompts/rq1/Gemma-7b-it_comment_extend_full.jsonl \\
        --output data/outputs/gemma-3-12b-it_comment_extend_full.jsonl \\
        --model  gemma-3-12b-it
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request


def strip_thinking_blocks(text: str) -> str:
    """Remove blocos <think>...</think> gerados pelo raciocínio do Qwen3/R1.

    Trata dois casos:
    - Bloco fechado:   <think>...</think>  → removido inteiro
    - Bloco aberto:    <think>... (cortado por max_tokens) → tudo até o fim
      removido; o código Java válido fica ANTES do <think>
    """
    # Remove blocos fechados
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove bloco aberto restante (se o modelo foi cortado antes de fechar)
    if "<think>" in text:
        text = text[: text.index("<think>")]
    return text.strip()

GROQ_API_URL    = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_API_URL  = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
MODEL_DEFAULT = "qwen/qwen3-32b"

# Modelos Groq com 7B+ parâmetros (verificar disponibilidade em console.groq.com):
#   qwen/qwen3-32b                          — Qwen3 32B  (500K TPD) ← padrão
#   meta-llama/llama-4-scout-17b-16e-instruct — Llama 4 Scout 17B (500K TPD)
#   llama-3.3-70b-versatile                 — Llama 3.3 70B (100K TPD)
#   openai/gpt-oss-120b                     — GPT-OSS 120B (200K TPD)
#   gemma2-9b-it                            — Gemma 2 9B  (verificar TPD)
#   deepseek-r1-distill-llama-70b           — DeepSeek R1 70B (verificar TPD)


class DailyLimitExceeded(Exception):
    """Limite diário de tokens (TPD) da chave Groq atingido.

    Sinaliza para o chamador que deve trocar para a próxima chave disponível.
    """


def _load_env_file() -> dict[str, str]:
    """Procura um arquivo .env subindo a árvore de diretórios e lê as variáveis.

    Suporta tanto o formato padrão (``KEY=value``) quanto o formato Python
    usado neste projeto (``key = "value"``).
    """
    cur = os.path.abspath(os.path.dirname(__file__))
    for _ in range(6):  # sobe no máximo 6 níveis
        candidate = os.path.join(cur, ".env")
        if os.path.exists(candidate):
            env: dict[str, str] = {}
            with open(candidate, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    # aceita: KEY=value  |  KEY = "value"  |  KEY = 'value'
                    m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*["\']?(.*?)["\']?\s*$', line)
                    if m:
                        env[m.group(1)] = m.group(2)
            return env
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return {}

# Intervalo mínimo entre requisições (segundos) para respeitar rate-limits do
# tier gratuito da Groq (~30 req/min para modelos Gemma).
_MIN_REQUEST_INTERVAL_GROQ   = 2.1   # ~30 RPM
_MIN_REQUEST_INTERVAL_GEMINI = 6.0   # ~10 RPM — margem segura abaixo do limite de 15 RPM


def _key(rec: dict) -> tuple:
    return (rec.get("id", ""), rec.get("method_signature", ""))


def load_done(output_path: str) -> set:
    done: set = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Considera "done" tanto respostas válidas quanto TOO_LARGE
            if "completion" in rec:
                done.add(_key(rec))
    return done


def parse_gemma_prompt(raw_prompt: str) -> tuple[str, str]:
    """Separa prompt Gemma em (mensagem_usuario, prefixo_assistente).

    Formato esperado::

        <start_of_turn>user
        {USER_MSG}<end_of_turn>
        <start_of_turn>model
        {ASSISTANT_PREFIX}

    Retorna (user_msg, assistant_prefix).  Se o formato não for reconhecido,
    retorna (raw_prompt, "").
    """
    USER_START  = "<start_of_turn>user\n"
    USER_END    = "<end_of_turn>"
    MODEL_START = "<start_of_turn>model\n"

    if USER_START not in raw_prompt:
        return raw_prompt, ""

    after_user = raw_prompt[raw_prompt.index(USER_START) + len(USER_START):]

    if USER_END not in after_user:
        return after_user.strip(), ""

    user_msg = after_user[: after_user.index(USER_END)]
    rest     = after_user[after_user.index(USER_END) + len(USER_END):]

    assistant_prefix = ""
    if MODEL_START in rest:
        assistant_prefix = rest[rest.index(MODEL_START) + len(MODEL_START):]

    return user_msg.strip(), assistant_prefix


def groq_chat(
    api_key: str,
    model: str,
    user_message: str,
    assistant_prefix: str,
    *,
    api_url: str = GROQ_API_URL,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: int,
    retries: int,
    no_thinking: bool = False,
) -> str:
    """Chama a API de chat da Groq e retorna o conteúdo gerado.

    Se ``assistant_prefix`` não for vazio, ele é incluído como última mensagem
    do assistente (prefill), e a API continua a partir desse ponto.
    A completion retornada é apenas a *continuação* — o chamador deve
    prefixa-la com ``assistant_prefix`` para obter o texto completo.
    """
    # /no_think desabilita o raciocínio interno do Qwen3 — reduz tokens de
    # ~1500 para ~400 por resposta, evitando bater o TPM do tier gratuito.
    content = f"/no_think\n{user_message}" if no_thinking else user_message
    messages: list[dict] = [{"role": "user", "content": content}]
    if assistant_prefix:
        messages.append({"role": "assistant", "content": assistant_prefix})

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    data = json.dumps(payload).encode("utf-8")
    # Headers: User-Agent do SDK Groq para passar pelo Cloudflare;
    # para outros providers (Gemini, HF) o header padrão funciona.
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "groq-python/0.18.0",
        "X-Stainless-Lang": "python",
        "X-Stainless-Package-Version": "0.18.0",
        "X-Stainless-Runtime": "CPython",
        "X-Stainless-Runtime-Version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }

    last_err: object = None
    errors = 0  # conta apenas falhas de rede/5xx — 429 não conta
    while True:
        try:
            req = urllib.request.Request(api_url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            last_err = f"HTTP {exc.code}: {body_text[:400]}"
            # 4xx não-transitórios: falha imediata, sem retry
            # 413 = prompt maior que o limite de tokens do modelo
            if exc.code in (400, 401, 403, 404, 413, 422):
                raise RuntimeError(last_err) from exc
            # 429 = rate-limit: distingue TPM (espera) de TPD (troca de chave)
            if exc.code == 429:
                is_daily = ("TPD" in body_text or "per day" in body_text
                            or "RESOURCE_EXHAUSTED" in body_text and "quota" in body_text.lower())
                if is_daily:
                    raise DailyLimitExceeded(
                        f"Limite diário atingido: {body_text[:200]}"
                    ) from exc
                # Tenta ler retryDelay do corpo (formato Gemini: "retryDelay": "30s")
                retry_delay_match = re.search(r'"retryDelay"\s*:\s*"(\d+)s"', body_text)
                if retry_delay_match:
                    wait = int(retry_delay_match.group(1))
                else:
                    wait = int(exc.headers.get("Retry-After", 60))
                print(f"    [rate-limit] aguardando {wait}s…", file=sys.stderr)
                time.sleep(wait)
                continue
            # 5xx e outros: conta como erro e aplica backoff
            errors += 1
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_err = exc
            errors += 1

        if errors >= retries:
            raise RuntimeError(f"falha após {retries} tentativas de rede: {last_err}")

        wait = min(2 ** errors, 30)
        print(
            f"    [retry {errors}/{retries}] erro: {last_err}; aguardando {wait}s",
            file=sys.stderr,
        )
        time.sleep(wait)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input",  required=True, help="JSONL de prompts")
    ap.add_argument("--output", required=True, help="JSONL de saída (com 'completion')")
    ap.add_argument("--model",  default=MODEL_DEFAULT,
                    help=f"ID do modelo na Groq (default: {MODEL_DEFAULT})")
    ap.add_argument("--api-key", default="",
                    help="API key(s) separadas por vírgula. Troca automaticamente "
                         "ao atingir limite diário. Ordem: --api-key > GROQ_API_KEYS "
                         "> GROQ_API_KEY > GEMINI_API_KEY > .env")
    ap.add_argument("--base-url", default="",
                    help="URL base do endpoint OpenAI-compatible. "
                         "Padrão: Groq. Atalhos: 'gemini' ou URL completa. "
                         "Ex: --base-url gemini")
    ap.add_argument("--limit",   type=int, default=0,
                    help="para após N *tentativas* de requisição (0 = todas)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p",       type=float, default=0.95)
    ap.add_argument("--max-tokens",  type=int,   default=4096,
                    help="máx. de tokens gerados por resposta (padrão 4096 para "
                         "acomodar o raciocínio interno do Qwen3/R1)")
    ap.add_argument("--no-thinking", action="store_true",
                    help="desabilita o raciocínio interno do Qwen3 via /no_think "
                         "(reduz ~1500 para ~400 tokens/resposta, evita bater o "
                         "TPM do tier gratuito — use com --max-tokens 1024)")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    # Resolve URL do endpoint
    base_url = args.base_url.strip().lower()
    if base_url == "gemini":
        api_url = GEMINI_API_URL
    elif base_url:
        api_url = base_url  # URL completa fornecida pelo usuário
    else:
        api_url = GROQ_API_URL
    # Intervalo padrão por provider
    default_interval = (_MIN_REQUEST_INTERVAL_GEMINI
                        if api_url == GEMINI_API_URL
                        else _MIN_REQUEST_INTERVAL_GROQ)
    print(f"Endpoint: {api_url}")

    # Resolve API keys — ordem depende do provider
    env_vars = _load_env_file()
    raw_keys = args.api_key
    if not raw_keys:
        if api_url == GEMINI_API_URL:
            # Gemini: prioriza GEMINI_API_KEY
            raw_keys = (os.environ.get("GEMINI_API_KEY", "")
                        or env_vars.get("GEMINI_API_KEY", ""))
        else:
            # Groq / outros: prioriza GROQ_API_KEYS
            raw_keys = (os.environ.get("GROQ_API_KEYS", "")
                        or os.environ.get("GROQ_API_KEY", "")
                        or env_vars.get("GROQ_API_KEYS", "")
                        or env_vars.get("GROQ_API_KEY", "")
                        or env_vars.get("groq_api", ""))
    if not raw_keys:
        print("ERRO: API key não encontrada. Use --api-key, GROQ_API_KEY, "
              "GEMINI_API_KEY ou .env", file=sys.stderr)
        return 1

    api_keys: list[str] = [k.strip() for k in raw_keys.split(",") if k.strip()]
    print(f"Chaves disponíveis: {len(api_keys)}")
    if not os.path.exists(args.input):
        print(f"ERRO: input não encontrado: {args.input}", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    done = load_done(args.output)
    print(f"Já respondidos: {len(done)}")

    with open(args.input, "r", encoding="utf-8") as fh:
        prompts = [json.loads(line) for line in fh if line.strip()]
    total = len(prompts)
    print(f"Total de prompts: {total} | modelo: {args.model}")

    processed = 0
    attempted = 0  # conta requisições enviadas (para o --limit)
    last_req_time = 0.0
    t0 = time.time()
    key_idx = 0  # índice da chave ativa

    with open(args.output, "a", encoding="utf-8") as out:
        for i, rec in enumerate(prompts, 1):
            if _key(rec) in done:
                continue
            if args.limit and attempted >= args.limit:
                break
            attempted += 1

            raw_prompt = rec.get("prompt", "")
            if not raw_prompt:
                continue

            user_msg, assistant_prefix = parse_gemma_prompt(raw_prompt)

            # Respeita rate-limit do tier gratuito
            elapsed_since_last = time.time() - last_req_time
            if elapsed_since_last < default_interval:
                time.sleep(default_interval - elapsed_since_last)

            t = time.time()
            try:
                response = groq_chat(
                    api_keys[key_idx],
                    args.model,
                    user_msg,
                    assistant_prefix,
                    api_url=api_url,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    timeout=args.timeout,
                    retries=args.retries,
                    no_thinking=args.no_thinking,
                )
            except DailyLimitExceeded as exc:
                print(f"\n[chave {key_idx + 1}/{len(api_keys)}] Limite diário (TPD) "
                      f"atingido: {exc}", file=sys.stderr)
                key_idx += 1
                if key_idx >= len(api_keys):
                    print("ERRO: todas as chaves atingiram o limite diário. "
                          "Execute novamente amanhã.", file=sys.stderr)
                    break
                print(f"    → trocando para chave {key_idx + 1}/{len(api_keys)}…"
                      f" retentando prompt {rec.get('id')}…", file=sys.stderr)
                attempted -= 1  # não conta esta tentativa
                # Retenta o mesmo prompt com a nova chave
                try:
                    response = groq_chat(
                        api_keys[key_idx],
                        args.model,
                        user_msg,
                        assistant_prefix,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        timeout=args.timeout,
                        retries=args.retries,
                        no_thinking=args.no_thinking,
                    )
                except Exception as exc2:  # noqa: BLE001
                    print(f"[{i}/{total}] FALHA {rec.get('id')} após troca de chave "
                          f":: {exc2}", file=sys.stderr)
                    continue
            except Exception as exc:  # noqa: BLE001
                exc_str = str(exc)
                print(f"[{i}/{total}] FALHA {rec.get('id')} :: {exc_str[:120]}",
                      file=sys.stderr)
                # Prompts grandes demais (413): salva registro para não tentar de novo
                if "413" in exc_str:
                    rec_out = dict(rec)
                    rec_out["completion"] = "TOO_LARGE"
                    rec_out["model"] = args.model
                    rec_out["error"] = "prompt_too_large"
                    out.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
                    out.flush()
                    print(f"    → salvo como TOO_LARGE (será ignorado na avaliação)",
                          file=sys.stderr)
                continue
            finally:
                last_req_time = time.time()

            # Remove <think>...</think> (raciocínio interno do Qwen3/R1)
            response = strip_thinking_blocks(response)

            # Para estratégia "extend": completion = prefixo_da_classe + continuação
            completion = (assistant_prefix + response) if assistant_prefix else response

            rec_out = dict(rec)
            rec_out["completion"] = completion
            rec_out["model"] = args.model
            out.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            out.flush()
            processed += 1
            dt = time.time() - t
            print(f"[{i}/{total}] {rec.get('id')} :: {dt:.1f}s ({len(completion)} chars)")

    elapsed = time.time() - t0
    print(
        f"\n=== concluído: {processed} novos em {elapsed:.0f}s "
        f"(total no arquivo: {len(load_done(args.output))}) ===",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
