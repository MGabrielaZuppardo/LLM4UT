"""
Inferência local via Ollama para os prompts gerados pelo LLM4UT.

Lê o JSONL de prompts (saída de ``rq1/generate_prompts_gemma.py``), envia cada
prompt para um modelo servido pelo Ollama e grava um novo JSONL acrescentando a
chave ``completion`` (exigida pelo pipeline de avaliação do LLM4UT).

Os prompts já vêm no template do Gemma (``<start_of_turn>...``) e terminam no
início da classe de teste (estratégia "extend"); por isso usamos o endpoint
``/api/generate`` com ``raw=true``, fazendo o modelo CONTINUAR o texto sem que o
Ollama aplique outro template por cima.

Características:
  * Idempotente/retomável: relê o output e pula prompts já respondidos (por id +
    method_signature).
  * Robusto: timeout, tentativas, e gravação incremental (uma linha por vez).

Uso::

    # teste rápido
    python tools/run_ollama_inference.py \
        --input data/prompts/rq1/Gemma-7b-it_comment_extend_full.jsonl \
        --output data/rq1/results_gemma3_4b/completions.jsonl \
        --model gemma3:4b --limit 3

    # tudo (use dentro de tmux)
    python tools/run_ollama_inference.py \
        --input data/prompts/rq1/Gemma-7b-it_comment_extend_full.jsonl \
        --output data/rq1/results_gemma3_4b/completions.jsonl \
        --model gemma3:4b
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request


def _key(rec: dict) -> tuple:
    return (rec.get("id", ""), rec.get("method_signature", ""))


def load_done(output_path: str) -> set:
    done = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "completion" in rec:
                    done.add(_key(rec))
    return done


def ollama_generate(host: str, model: str, prompt: str, *,
                    temperature: float, top_p: float, num_predict: int,
                    timeout: int, retries: int) -> str:
    url = f"{host.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "raw": True,          # não aplica template extra; o prompt já tem o do Gemma
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": num_predict,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, data=data,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return body.get("response", "")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_err = e
            wait = min(2 ** attempt, 30)
            print(f"    [retry {attempt}/{retries}] erro: {e}; aguardando {wait}s",
                  file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"falha após {retries} tentativas: {last_err}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="JSONL de prompts")
    ap.add_argument("--output", required=True, help="JSONL de saída (com 'completion')")
    ap.add_argument("--model", default="gemma3:4b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--limit", type=int, default=0, help="processa no máximo N (0 = todos)")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Temperatura de geração (padrão: 0 — determinístico, "
                         "conforme metodologia do paper ASE'24)")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--num-predict", type=int, default=1024,
                    help="máx. de tokens gerados por resposta")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"ERRO: input não encontrado: {args.input}", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    done = load_done(args.output)
    print(f"Já respondidos: {len(done)}")

    with open(args.input, "r", encoding="utf-8") as fh:
        prompts = [json.loads(l) for l in fh if l.strip()]
    total = len(prompts)
    print(f"Total de prompts: {total} | modelo: {args.model}")

    processed = 0
    t0 = time.time()
    with open(args.output, "a", encoding="utf-8") as out:
        for i, rec in enumerate(prompts, 1):
            if _key(rec) in done:
                continue
            if args.limit and processed >= args.limit:
                break
            prompt = rec.get("prompt", "")
            if not prompt:
                continue
            t = time.time()
            try:
                completion = ollama_generate(
                    args.host, args.model, prompt,
                    temperature=args.temperature, top_p=args.top_p,
                    num_predict=args.num_predict, timeout=args.timeout,
                    retries=args.retries,
                )
            except Exception as e:  # noqa: BLE001
                print(f"[{i}/{total}] FALHA {rec.get('id')} :: {e}", file=sys.stderr)
                continue
            rec_out = dict(rec)
            rec_out["completion"] = completion
            rec_out["model"] = args.model
            out.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            out.flush()
            processed += 1
            dt = time.time() - t
            print(f"[{i}/{total}] {rec.get('id')} :: {dt:.1f}s "
                  f"({len(completion)} chars)")

    elapsed = time.time() - t0
    print(f"\n=== concluído: {processed} novos em {elapsed:.0f}s "
          f"(total no arquivo: {len(load_done(args.output))}) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
