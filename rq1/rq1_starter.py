import sys
sys.path.extend(['.', '..'])
import subprocess
import time
import json
import os
from data import configuration

projects = configuration.projects
code_base = configuration.code_base
python_bin = configuration.python_bin
target_models = configuration.target_models
formats = configuration.formats
strategies = configuration.strategies
ablations = configuration.ablations

report_dir = f"{code_base}/rq1/coverage_reports"
if os.path.exists(report_dir):
    os.system(f"rm -rf {report_dir}")

results_base = configuration.output_base_dir  # lê de configuration.output_dir


def expected_count(model, project):
    """Count how many public items exist for this project in the model's output file."""
    output_file = os.path.join(code_base, f"data/outputs/{model}_{formats[0]}_{strategies[0]}_{ablations[0]}.jsonl")
    if not os.path.exists(output_file):
        return 0
    count = 0
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                if d.get("project") == project and str(d.get("is_public", "0")) == "1":
                    count += 1
            except Exception:
                pass
    return count


def result_line_count(model, project):
    path = os.path.join(results_base, model, f"{project}_{formats[0]}_{strategies[0]}_{ablations[0]}.jsonl")
    if not os.path.exists(path):
        return 0
    try:
        return sum(1 for l in open(path, encoding="utf-8") if l.strip())
    except Exception:
        return 0


def is_complete(project):
    """Return True only if ALL models have result files with >= expected item count."""
    for model in target_models:
        exp = expected_count(model, project)
        got = result_line_count(model, project)
        if got < exp:
            return False
    return True


lst = []
for project in projects:
    if is_complete(project):
        print(f"Skipping {project} (all models complete)")
        continue
    # Show what's missing
    for model in target_models:
        exp = expected_count(model, project)
        got = result_line_count(model, project)
        if got < exp:
            print(f"  {model}/{project}: {got}/{exp} — will rerun")
    p = subprocess.Popen([f"{python_bin} {code_base}/rq1/rq1.py {project}"], shell=True)
    time.sleep(1)
    lst.append(p)

print(f"\nWaiting for {len(lst)} project(s) to finish...")
for x in lst:
    x.wait()

print("All evaluations finished.")
