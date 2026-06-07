import os
import sys

# garante que a raiz do projeto esteja no path para 'utils' resolver
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

d4j_home = "/home/gabriela_zuppardo/defects4j"
d4j_proj_base = f"{d4j_home}/d4j_projects"
# output_dir = "data/rq1/results_debug"
# t1 (temperatura padrão): output_dir = "data/rq1/t1/results_0128"
output_dir = "data/rq1/results_gemma4_12b_t0"
code_base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
output_base_dir = os.path.join(code_base, output_dir)
d4j_command = f"{d4j_home}/framework/bin/defects4j"
python_bin = sys.executable

# tree-sitter: API moderna (0.22+) via camada de compatibilidade.
# O antigo Language.build_library() foi removido; ver utils/ts_compat.py.
from utils.ts_compat import JAVA_LANGUAGE  # noqa: E402

RES_FILE="/Users/yanglin/Documents/Projects/code-bot/data/outputs/deepseek-coder-6.7b-instruct_comment_extend_full.jsonl"
TMP_FOLDER="/Users/yanglin/Documents/Projects/code-bot/data/tmp"
GRANULARITY = 'method'
OUTPUT_FILE = '/Users/yanglin/Documents/Projects/example.jsonl'


proxy_host = None
proxy_port = None
proxy_username = None
proxy_password = None

projects = [
    "JxPath",
    "Cli",
    "Math",
    "Csv",
    "Compress",
    "JacksonDatabind",
    "Time",
    "Collections",
    "JacksonXml",
    # "Mockito",
    "JacksonCore",
    "Lang",
    "Jsoup",
    "Chart",
    "Gson",
    "Closure",
    "Codec",
]

target_models = [
    # "gemma3_4b",
    # "llama4_scout_17b",
    # "deepseek_r1_1.5b",
    # "deepseek_r1_8b",
    # "mistral_7b",
    # "llama3.1_8b",
    "gemma4_12b",
    # "gemma3_12b",
    # "gemma3_27b",
    # "deepseek_r1_32b",
    # "qwen3_32b",
]

formats = [
    "comment",
    # 'natural'
]

strategies = [
    # 'generation',
    "extend"
]

ablations = [
    "full",
    # 'no_param',
    # 'no_param_constructor',
    # 'no_class_constructor',
    # 'no_class_fields',
    # 'no_class_other_methods'
]
