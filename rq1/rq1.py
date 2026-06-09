import sys
import os
import json
from tqdm import tqdm

sys.path.extend([".", ".."])

from assistant_methods import run, filter_data_according_to_project
from utils.d4j_utils import load_assistant_data_with_shell
from data import configuration
from data.configuration import output_base_dir, code_base, d4j_proj_base

target_models = configuration.target_models
formats = configuration.formats
strategies = configuration.strategies
ablations = configuration.ablations


def main(target_project="Chart"):
    # 加载辅助数据
    assistant_datas = load_assistant_data_with_shell()
    # 遍历所有目标模型
    for tgt_model in target_models:
        # 遍历所有prompt格式
        for strategy in strategies:
            # 遍历所有变体
            for ablation in ablations:
                for format in formats:
                    # 定义输出结果的路径
                    output_base = os.path.join(output_base_dir, tgt_model)
                    if not os.path.exists(output_base):
                        os.makedirs(output_base)

                    input_file = os.path.join(
                        code_base,
                        f"data/outputs/{tgt_model}_{format}_{strategy}_{ablation}.jsonl",
                    )

                    if not os.path.exists(input_file):
                        print(
                            "ERROR: Generation file %s does not exist, please check."
                            % input_file
                        )
                        continue
                    if not os.path.exists(d4j_proj_base):
                        print(
                            "ERROR: Defects4j folder %s does not exist, please check"
                            % d4j_proj_base
                        )
                        continue

                    # Skip if this model/project combination already has a complete result file.
                    result_path = os.path.join(
                        output_base_dir, tgt_model,
                        f"{target_project}_{format}_{strategy}_{ablation}.jsonl"
                    )
                    if os.path.exists(result_path) and os.path.getsize(result_path) > 0:
                        # Count expected items for this model/project
                        expected = 0
                        with open(input_file, "r", encoding="utf-8") as _f:
                            for _line in _f:
                                try:
                                    _d = json.loads(_line)
                                    if _d.get("project") == target_project and str(_d.get("is_public", "0")) == "1":
                                        expected += 1
                                except Exception:
                                    pass
                        actual = sum(1 for _ in open(result_path, encoding="utf-8") if _.strip())
                        if actual >= expected:
                            print(f"Skipping {tgt_model}/{target_project}: already complete ({actual}/{expected})")
                            continue

                    datas = filter_data_according_to_project(
                        input_file, assistant_datas, target_project
                    )

                    print("Load %d generations from %s" % (len(datas), input_file))

                    # Carrega bug_ids já processados para retomar de onde parou
                    result_path = os.path.join(
                        output_base,
                        f"{target_project}_{format}_{strategy}_{ablation}.jsonl"
                    )
                    done_bug_ids = set()
                    if os.path.exists(result_path):
                        with open(result_path, "r", encoding="utf-8") as _rf:
                            for _line in _rf:
                                try:
                                    _d = json.loads(_line)
                                    _bid = _d.get("bug_id") or _d.get("id")
                                    if _bid:
                                        done_bug_ids.add(_bid)
                                except Exception:
                                    pass

                    analyze_res_writer = open(
                        result_path,
                        "a" if done_bug_ids else "w",
                        encoding="utf-8",
                    )
                    pending = [d for d in datas if (d.get("id", "") not in done_bug_ids)]
                    print(f"Resuming {tgt_model}/{target_project}: "
                          f"{len(done_bug_ids)} done, {len(pending)} remaining. "
                          f"Next: {pending[0].get('id') if pending else 'none'}")
                    # 开始遍历模型输出结果，进行编译&测试，统计收集指标
                    for index, data in tqdm(
                            enumerate(datas),
                            total=len(datas),
                            desc=f"Evaluating {tgt_model} on {target_project}",
                    ):
                        _bid = data.get("id", "")
                        if _bid in done_bug_ids:
                            continue
                        # Bugs que travam o compile no WSL2 — registra como falha e pula
                        _HANG_PREFIXES = ("Closure_164", "Closure_167", "Closure_169", "Closure_22", "Closure_30")
                        if any(_bid.startswith(p) for p in _HANG_PREFIXES):
                            print(f"[skip hang] {_bid}")
                            _skip_rec = dict(data)
                            _skip_rec.update({
                                "index": index, "bug_id": _bid,
                                "exception": "compile_hang_skipped",
                                "first_compile_res": "error", "second_compile_res": "error",
                                "is_empty_test": True, "num_total_uts": 0,
                                "num_compilable_uts": 0, "num_executed_uts": 0,
                                "num_passed_uts": 0, "covered_lines": 0,
                                "missed_lines": 0, "covered_branches": 0,
                                "missed_branches": 0,
                            })
                            analyze_res_writer.write(json.dumps(_skip_rec) + "\n")
                            analyze_res_writer.flush()
                            continue
                        try:
                            res_dict = run(
                                model=tgt_model,
                                strategy=strategy,
                                data=data,
                                index=index,
                                ablation=ablation,
                                format=format,
                            )
                        except Exception as e:
                            import traceback
                            print(f"ERROR item {index} ({data.get('id','?')}): {type(e).__name__}: {e}")
                            traceback.print_exc()
                            res_dict = dict(data)
                            res_dict.update({
                                "index": index, "bug_id": data.get("id", "?"),
                                "exception": f"{type(e).__name__}: {e}",
                                "first_compile_res": "error", "second_compile_res": "error",
                                "is_empty_test": True, "num_total_uts": 0,
                                "num_compilable_uts": 0, "num_executed_uts": 0,
                                "num_passed_uts": 0, "covered_lines": -1,
                                "covered_branches": -1,
                            })
                        analyze_res_writer.write(json.dumps(res_dict) + "\n")
                        analyze_res_writer.flush()
                    analyze_res_writer.close()


if __name__ == "__main__":
    proj = sys.argv[1]
    # proj = "Chart"
    main(proj)

    # from data.configuration import projects
    # for proj in projects:
    #     main(proj)
    #     # break
