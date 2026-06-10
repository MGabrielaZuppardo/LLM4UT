"""
Análise estatística comparativa entre modelos — RQ1.

Para cada métrica (CSR, CovL, CovB, NDD) compara todos os pares de modelos
usando o Wilcoxon signed-rank test (nível de método/bug, conforme Yang et al.)
e calcula o rank-biserial correlation como medida de effect size.

Saída:
  data/rq1/statistical_analysis_<data>.csv  — tabela completa de pares
  data/rq1/statistical_analysis_<data>.txt  — sumário legível

Uso:
  # a partir da raiz do projeto
  python rq1/analyze/statistical_analysis.py

Dependências extras além de requirements.txt:
  pip install scipy
"""

import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from itertools import combinations

sys.path.extend([".", ".."])

import pandas as pd
from scipy.stats import wilcoxon

from data.configuration import (
    ablations,
    code_base,
    formats,
    output_base_dir,
    projects,
    strategies,
    target_models,
)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

ALPHA = 0.05  # nível de significância

# ---------------------------------------------------------------------------
# Leitura dos resultados
# ---------------------------------------------------------------------------


def _load_records(model: str, project: str, fmt: str, strategy: str, ablation: str) -> list[dict]:
    path = os.path.join(output_base_dir, model, f"{project}_{fmt}_{strategy}_{ablation}.jsonl")
    if not os.path.exists(path):
        return []
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _is_valid(rec: dict) -> bool:
    c = rec.get("completion", "")
    return c not in ("", "out of max_tokens", "TOO_LARGE", None)


# ---------------------------------------------------------------------------
# Vetores de métricas por método (granularidade de comparação)
# ---------------------------------------------------------------------------


def collect_vectors(model: str, fmt: str, strategy: str, ablation: str) -> dict[str, list[float]]:
    """
    Retorna {metric: [valor_por_método]} agregando todos os projetos.

    CSR  — 1 se second_compile_res == 'success', 0 caso contrário
    CovL — covered_lines / (covered_lines + missed_lines), ou 0 se não compilou
    CovB — covered_branches / (covered_branches + missed_branches), ou 0/-1
    NDD  — por bug_id: 1 se fixed_passed=True AND buggy_passed=False, 0 caso contrário
           (NDD é calculado no nível de bug, não de método — um vetor separado)
    """
    csr_vec: list[float] = []
    covl_vec: list[float] = []
    covb_vec: list[float] = []
    ndd_map: dict[str, float] = {}  # bug_id -> 1 ou 0

    for project in projects:
        for rec in _load_records(model, project, fmt, strategy, ablation):
            if not _is_valid(rec):
                continue

            # CSR
            compiled = 1.0 if rec.get("second_compile_res") == "success" else 0.0
            csr_vec.append(compiled)

            # CovL
            cl = rec.get("covered_lines", 0) or 0
            ml = rec.get("missed_lines", 0) or 0
            total_l = cl + ml
            covl_vec.append(cl / total_l if total_l > 0 else 0.0)

            # CovB
            cb = rec.get("covered_branches", -1)
            mb = rec.get("missed_branches", -1)
            if cb != -1 and mb != -1:
                total_b = cb + mb
                covb_vec.append(cb / total_b if total_b > 0 else 0.0)

            # NDD — nível de bug (fixed_execution_result / buggy_execution_result)
            bug_id = rec.get("bug_id")
            if bug_id and rec.get("fixed_execution_result") and not rec.get("is_empty_test"):
                if rec.get("fixed_execution_error_info", [""])[0] != "not compiled":
                    detected = 1.0 if not rec.get("buggy_execution_result") else 0.0
                    ndd_map[bug_id] = max(ndd_map.get(bug_id, 0.0), detected)

    ndd_vec = list(ndd_map.values())
    return {"CSR": csr_vec, "CovL": covl_vec, "CovB": covb_vec, "NDD": ndd_vec}


# ---------------------------------------------------------------------------
# Estatística
# ---------------------------------------------------------------------------


def rank_biserial(x: list[float], y: list[float]) -> float:
    """
    Rank-biserial correlation r = 1 - (2 * W) / (n1 * n2)
    onde W é a estatística do Wilcoxon e n1=n2=len(differences).

    Interpretação:
      |r| < 0.10  negligível
      |r| < 0.30  pequeno
      |r| < 0.50  médio
      |r| >= 0.50 grande
    """
    diffs = [xi - yi for xi, yi in zip(x, y) if xi != yi]
    if not diffs:
        return 0.0
    n = len(diffs)
    stat, _ = wilcoxon(diffs, zero_method="wilcox")
    r = 1.0 - (2.0 * stat) / (n * (n + 1) / 2)
    return round(float(r), 4)


def effect_label(r: float) -> str:
    a = abs(r)
    if a < 0.10:
        return "negligível"
    if a < 0.30:
        return "pequeno"
    if a < 0.50:
        return "médio"
    return "grande"


def compare_pair(
    vec_a: list[float], vec_b: list[float], metric: str
) -> dict:
    """
    Compara dois vetores do mesmo tamanho com o teste de Wilcoxon.
    Retorna um dicionário com os resultados.
    """
    n = min(len(vec_a), len(vec_b))
    if n < 10:
        return {
            "n": n,
            "mean_a": round(sum(vec_a[:n]) / n, 4) if n else None,
            "mean_b": round(sum(vec_b[:n]) / n, 4) if n else None,
            "statistic": None,
            "p_value": None,
            "significant": None,
            "r": None,
            "effect": "insuficiente (n<10)",
        }

    a = vec_a[:n]
    b = vec_b[:n]
    diffs = [x - y for x, y in zip(a, b)]
    if all(d == 0 for d in diffs):
        stat, p = 0.0, 1.0
    else:
        stat, p = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")

    r = rank_biserial(a, b)
    return {
        "n": n,
        "mean_a": round(sum(a) / n, 4),
        "mean_b": round(sum(b) / n, 4),
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "significant": bool(p < ALPHA),
        "r": r,
        "effect": effect_label(r),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    fmt = formats[0]
    strategy = strategies[0]
    ablation = ablations[0]

    print(f"Coletando vetores de métricas ({fmt}/{strategy}/{ablation}) ...")
    model_vectors: dict[str, dict[str, list[float]]] = {}
    for model in target_models:
        model_vectors[model] = collect_vectors(model, fmt, strategy, ablation)
        for metric, vec in model_vectors[model].items():
            mean = sum(vec) / len(vec) if vec else 0
            print(f"  {model:25s} {metric}: n={len(vec):4d}  mean={mean:.4f}")

    metrics = ["CSR", "CovL", "CovB", "NDD"]
    rows = []

    print("\nComparações par-a-par (Wilcoxon + rank-biserial r):\n")
    header = f"{'Metrica':<6} {'Modelo A':<25} {'Modelo B':<25} {'n':>5} {'meanA':>7} {'meanB':>7} {'W':>10} {'p':>9} {'sig':>4} {'r':>7} {'efeito'}"
    print(header)
    print("-" * len(header))

    for metric in metrics:
        for model_a, model_b in combinations(target_models, 2):
            va = model_vectors[model_a][metric]
            vb = model_vectors[model_b][metric]
            res = compare_pair(va, vb, metric)
            sig_str = "S" if res["significant"] else "N"
            print(
                f"{metric:<6} {model_a:<25} {model_b:<25} "
                f"{res['n']:>5} {str(res['mean_a']):>7} {str(res['mean_b']):>7} "
                f"{str(res['statistic']):>10} {str(res['p_value']):>9} "
                f"{sig_str:>4} {str(res['r']):>7} {res['effect']}"
            )
            rows.append(
                {
                    "metric": metric,
                    "model_a": model_a,
                    "model_b": model_b,
                    "n": res["n"],
                    "mean_a": res["mean_a"],
                    "mean_b": res["mean_b"],
                    "wilcoxon_W": res["statistic"],
                    "p_value": res["p_value"],
                    "significant_p005": res["significant"],
                    "rank_biserial_r": res["r"],
                    "effect_size": res["effect"],
                }
            )

    # Salva CSV
    date_str = datetime.now().strftime("%Y-%m-%d")
    csv_path = os.path.join(code_base, f"data/rq1/statistical_analysis_{date_str}.csv")
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"\nTabela salva em: {csv_path}")

    # Salva TXT legível
    txt_path = csv_path.replace(".csv", ".txt")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(f"Análise Estatística — LLM4UT\n")
        fh.write(f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.write(f"Configuração: format={fmt}, strategy={strategy}, ablation={ablation}\n")
        fh.write(f"Modelos: {', '.join(target_models)}\n")
        fh.write(f"Projetos: {', '.join(projects)}\n")
        fh.write(f"Alpha: {ALPHA}\n")
        fh.write(f"Teste: Wilcoxon signed-rank (two-sided), zero_method='wilcox'\n")
        fh.write(f"Effect size: rank-biserial r  (|r|<0.10 negligivel, <0.30 pequeno, <0.50 medio, >=0.50 grande)\n\n")
        fh.write(header + "\n")
        fh.write("-" * len(header) + "\n")
        for row in rows:
            sig_str = "S" if row["significant_p005"] else "N"
            fh.write(
                f"{row['metric']:<6} {row['model_a']:<25} {row['model_b']:<25} "
                f"{str(row['n']):>5} {str(row['mean_a']):>7} {str(row['mean_b']):>7} "
                f"{str(row['wilcoxon_W']):>10} {str(row['p_value']):>9} "
                f"{sig_str:>4} {str(row['rank_biserial_r']):>7} {row['effect_size']}\n"
            )
    print(f"Sumário salvo em: {txt_path}")


if __name__ == "__main__":
    main()
