#!/usr/bin/env python3
"""Monitor the rq1 evaluation progress across all projects and models."""
import json, os, sys, time, subprocess

sys.path.extend(['/mnt/d/LLM4UT/rq1', '/mnt/d/LLM4UT'])
from data import configuration

PROJECTS  = configuration.projects
MODELS    = configuration.target_models
FORMATS   = configuration.formats
STRATEGIES = configuration.strategies
ABLATIONS  = configuration.ablations
CODE_BASE = configuration.code_base
RESULTS   = os.path.join(CODE_BASE, 'data/rq1/results_0128')
SUFFIX    = f'{FORMATS[0]}_{STRATEGIES[0]}_{ABLATIONS[0]}.jsonl'


def expected(model, project):
    path = os.path.join(CODE_BASE, f'data/outputs/{model}_{SUFFIX}')
    if not os.path.exists(path):
        return 0
    count = 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            try:
                d = json.loads(line)
                if d.get('project') == project and str(d.get('is_public', '0')) == '1':
                    count += 1
            except Exception:
                pass
    return count


def got(model, project):
    path = os.path.join(RESULTS, model, f'{project}_{SUFFIX}')
    if not os.path.exists(path):
        return 0
    try:
        return sum(1 for l in open(path, encoding='utf-8') if l.strip())
    except Exception:
        return 0


def running_projects():
    try:
        out = subprocess.check_output(['ps', 'aux'], text=True)
        return [l.split()[-1] for l in out.splitlines()
                if 'rq1.py' in l and 'grep' not in l and l.split()[-1] in PROJECTS]
    except Exception:
        return []


def quick_stats(model, project):
    path = os.path.join(RESULTS, model, f'{project}_{SUFFIX}')
    if not os.path.exists(path):
        return None
    try:
        results = [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]
        if not results:
            return None
        csr   = sum(1 for r in results if r.get('second_compile_res') == 'success')
        uts_c = sum(r.get('num_compilable_uts', 0) for r in results)
        return {'csr': csr, 'n': len(results), 'uts_c': uts_c}
    except Exception:
        return None


def main():
    # Build expected counts once
    exp = {}
    for m in MODELS:
        for p in PROJECTS:
            exp[(m, p)] = expected(m, p)

    while True:
        os.system('clear')
        now = time.strftime('%H:%M:%S')
        running = running_projects()

        print(f'╔══ LLM4UT Evaluation Monitor  {now} ══╗')
        print()

        # Header
        col = 18
        print(f'  {"Project":<{col}}', end='')
        for m in MODELS:
            short = m.replace('llama4_scout_17b', 'llama4').replace('gemma3_4b', 'gemma3')
            print(f'  {short:<22}', end='')
        print()
        print('  ' + '-' * (col + 24 * len(MODELS)))

        all_done = True
        for p in PROJECTS:
            running_mark = ' ⟳' if p in running else '  '
            print(f'{running_mark}{p:<{col}}', end='')
            for m in MODELS:
                g = got(m, p)
                e = exp[(m, p)]
                if e == 0:
                    cell = f'{"n/a":>6}'
                elif g >= e:
                    st = quick_stats(m, p)
                    if st:
                        cell = f'✓ {g:>3}  CSR={100*st["csr"]/st["n"]:>4.0f}%'
                    else:
                        cell = f'✓ {g:>3}/{e:<3}'
                else:
                    pct = 100 * g / e if e else 0
                    cell = f'{g:>3}/{e:<3} {pct:>4.0f}%'
                    all_done = False
                print(f'  {cell:<22}', end='')
            print()

        # Summary totals
        print()
        for m in MODELS:
            total_g = sum(got(m, p) for p in PROJECTS)
            total_e = sum(exp[(m, p)] for p in PROJECTS)
            results_all = []
            for p in PROJECTS:
                path = os.path.join(RESULTS, m, f'{p}_{SUFFIX}')
                if os.path.exists(path):
                    try:
                        results_all += [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]
                    except Exception:
                        pass
            csr = sum(1 for r in results_all if r.get('second_compile_res') == 'success')
            uts_c = sum(r.get('num_compilable_uts', 0) for r in results_all)
            uts_p = sum(r.get('num_passed_uts', 0) for r in results_all)
            n = len(results_all)
            short = m.replace('llama4_scout_17b', 'llama4_scout_17b').replace('gemma3_4b', 'gemma3_4b')
            print(f'  {short:<22}  Total: {total_g}/{total_e}  '
                  f'CSR={100*csr/max(n,1):.1f}%  '
                  f'comp_UTs={uts_c}  passed={uts_p}')

        print()
        if running:
            print(f'  Running: {", ".join(running)}')
            print(f'\n  Refreshing every 60s  (Ctrl+C to exit)')
            time.sleep(60)
        else:
            print('  All processes finished.')
            if all_done:
                print('  Evaluation complete!')
            else:
                print('  Some projects still incomplete — re-run rq1_starter.py')
            break


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n  Monitor stopped.')
