"""
Resumo RQ1 + RQ2 para todos os modelos avaliados.

Métricas do paper ASE'24:
  CSR   = test classes que compilaram (após remoção recursiva, não-vazias) / total
  CovL  = covered_lines / (covered_lines + missed_lines)  — média sobre métodos com cobertura
  CovB  = covered_branches / (covered_branches + missed_branches)
  NDD   = número de defeitos detectados (passa no fixed, falha no buggy)
"""
import json, os, glob, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

rq1_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data/rq1')
t1_base  = os.path.join(rq1_base, 't1', 'results_0128')

# Cada entrada: (label, pasta_de_resultados, subdir_dentro_da_pasta)
# Para t0 os jsonls ficam direto em results_<model>_t0/
# Para t1 ficam em t1/results_0128/<model>/
model_configs = []
for m in ['gemma3_4b', 'llama4_scout_17b', 'deepseek_r1_1.5b', 'deepseek_r1_8b', 'mistral_7b', 'llama3.1_8b']:
    t0_dir = os.path.join(rq1_base, f'results_{m}_t0')
    t1_dir = os.path.join(t1_base, m)
    if os.path.isdir(t0_dir):
        model_configs.append((f'{m}_t0', t0_dir))
    if os.path.isdir(t1_dir):
        model_configs.append((f'{m}_t1', t1_dir))

# Compatibilidade: 'models' lista labels; helper retorna glob pattern
def _result_files(result_dir):
    return glob.glob(os.path.join(result_dir, '*.jsonl'))

models = [label for label, _ in model_configs]
model_dirs = {label: d for label, d in model_configs}

SEP = '=' * 72

print(SEP)
print('RQ1 — CSR / CovL / CovB  (métricas do paper ASE\'24)')
print(SEP)

summary = {}
for model in models:
    files = _result_files(model_dirs[model])
    total = csr_ok = total_uts = comp_uts = exec_uts = pass_uts = 0
    cov_l_num = cov_l_den = cov_b_num = cov_b_den = 0

    for f in files:
        for line in open(f, encoding='utf-8'):
            d = json.loads(line)
            if not d.get('completion') or d['completion'] in ('', 'TOO_LARGE'):
                continue
            total += 1
            total_uts += d.get('num_total_uts', 0)

            # CSR: compilou na 2ª rodada e não é classe vazia
            if d.get('second_compile_res') == 'success' and not d.get('is_empty_test'):
                csr_ok += 1

            if not d.get('is_empty_test'):
                comp_uts += d.get('num_compilable_uts', 0)
                exec_uts += d.get('num_executed_uts', 0)
                pass_uts += d.get('num_passed_uts', 0)

            # CovL / CovB — apenas itens onde a cobertura foi coletada
            cl = d.get('covered_lines', -1)
            ml = d.get('missed_lines', -1)
            if cl >= 0 and ml >= 0:
                cov_l_num += cl
                cov_l_den += cl + ml

            cb = d.get('covered_branches', -1)
            mb = d.get('missed_branches', -1)
            if cb >= 0 and mb >= 0:
                cov_b_num += cb
                cov_b_den += cb + mb

    csr  = csr_ok  / total    * 100 if total    else 0
    covl = cov_l_num / cov_l_den * 100 if cov_l_den else 0
    covb = cov_b_num / cov_b_den * 100 if cov_b_den else 0
    mcr  = comp_uts / total_uts * 100 if total_uts else 0
    pr   = pass_uts / exec_uts  * 100 if exec_uts  else 0

    summary[model] = dict(total=total, csr=csr, covl=covl, covb=covb,
                          mcr=mcr, comp_uts=comp_uts, pass_uts=pass_uts)

    print(f'\n{model}  ({total} métodos)')
    print(f'  CSR   (classe compilou, não-vazia): {csr_ok}/{total} = {csr:.1f}%')
    print(f'  CovL  (line coverage):              {cov_l_num}/{cov_l_den} = {covl:.1f}%')
    print(f'  CovB  (branch coverage):            {cov_b_num}/{cov_b_den} = {covb:.1f}%')
    print(f'  MCR   (UTs compiláveis/total):      {comp_uts}/{total_uts} = {mcr:.1f}%')
    print(f'  Pass  (UTs passando/executando):    {pass_uts}/{exec_uts} = {pr:.1f}%')

# Referência do paper (melhores modelos originais)
print('\n' + '-'*72)
print('Referencia paper ASE\'24 (melhores modelos originais):')
print('  GPT-4:    CSR=52.96%  CovL=40.43%  CovB=31.78%')
print('  PD-34B:   CSR=49.07%  CovL=37.45%  CovB=32.35%')
print('  DC-33B:   CSR=45.65%  CovL=32.72%  CovB=29.26%')
print('  Evosuite: CSR=85.71%  CovL=78.91%  CovB=76.59%')

print()
print(SEP)
print('RQ2 — Detecção de Defeitos (NDD)')
print(SEP)

for model in models:
    files = _result_files(model_dirs[model])
    attempts = set()
    found = set()
    for f in files:
        for line in open(f, encoding='utf-8'):
            d = json.loads(line)
            if d.get('is_empty_test'):
                continue
            err_info = d.get('fixed_execution_error_info', ['not compiled'])
            if d.get('fixed_execution_result') and err_info[0] != 'not compiled':
                attempts.add(d['bug_id'])
                if not d.get('buggy_execution_result'):
                    found.add(d['bug_id'])
    ntd = len(attempts)
    ndd = len(found)
    rate = ndd / ntd * 100 if ntd else 0
    print(f'\n{model}')
    print(f'  NTD (defeitos testáveis):   {ntd}')
    print(f'  NDD (defeitos detectados):  {ndd}  ({rate:.1f}% dos testáveis)')
    top = sorted(found)[:8]
    suffix = f' ... +{ndd-8} mais' if ndd > 8 else ''
    if found:
        print(f'  Bugs: {top}{suffix}')

print('\n' + '-'*72)
print('Referencia paper ASE\'24:')
print('  GPT-4:  NTD=65  NDD=39  (60.0%)')
print('  DC-33B: NTD=62  NDD=33  (53.2%)')
print('  PD-34B: NTD=63  NDD=30  (47.6%)')
print('  DC-7B:  NTD=60  NDD=24  (40.0%)')
