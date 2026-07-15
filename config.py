import os
from dotenv import load_dotenv

load_dotenv()

DEPTHS = [0.00, 0.05, 0.25, 0.50, 0.75, 0.95, 1.00]

EXCLUDED = set()

# GQA KV cache at 128K fills a single T4 (16 GB); safe ceiling is ~50% depth.
# MLA (DeepSeek) compresses KV and fits full 128K on one T4 or P100.
SINGLE_GPU_SAFE_DEPTH_LIMIT = 0.50

# gpu_requirement: "dual_t4" = GQA (needs 2×16 GB), "single_gpu" = MLA (fits one)
MODELS = {
    "llama-3.1-8b-instruct": {
        "hf_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "max_new_tokens": 4096,
        "temperature": 0.0,
        "gpu_requirement": "dual_t4",
        "trust_remote_code": False,
    },
    "yi-coder-9b-chat": {
        "hf_id": "01-ai/Yi-Coder-9B-Chat",
        "max_new_tokens": 4096,
        "temperature": 0.0,
        "gpu_requirement": "dual_t4",
        "trust_remote_code": False,
    },
    "deepseek-coder-v2-lite-instruct": {
        "hf_id": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        "max_new_tokens": 4096,
        "temperature": 0.0,
        "gpu_requirement": "single_gpu",
        "trust_remote_code": True,
    },
}

RATE_LIMITS = {
    "llama-3.1-8b-instruct": {"sleep": 0},
    "yi-coder-9b-chat": {"sleep": 0},
    "deepseek-coder-v2-lite-instruct": {"sleep": 0},
}

REPOS = [
    {
        "name": "isort",
        "bug_type": "A",
        "target_file": "isort/output.py",
        "target_fn": "sorted_imports"
    },
    {
        "name": "httpx",
        "bug_type": "A",
        "target_file": "httpx/_decoders.py",
        "target_fn": "decode"
    },
    {
        "name": "arrow",
        "bug_type": "A",
        "target_file": "arrow/arrow.py",
        "target_fn": "interval"
    },
    {
        "name": "loguru",
        "bug_type": "A",
        "target_file": "loguru/_get_frame.py",
        "target_fn": "get_frame_fallback"
    },
    {
        "name": "more-itertools",
        "bug_type": "A",
        "target_file": "more_itertools/more.py",
        "target_fn": "divide"
    },
    {
        "name": "rich",
        "bug_type": "B",
        "target_file": "rich/spinner.py",
        "target_fn": "update"
    },
    {
        "name": "cerberus",
        "bug_type": "B",
        "target_file": "cerberus/validator.py",
        "target_fn": "_validate_regex"
    },
    {
        "name": "cattrs",
        "bug_type": "B",
        "target_file": "src/cattrs/converters.py",
        "target_fn": "_get_dis_func"
    },
    {
        "name": "attrs",
        "bug_type": "B",
        "target_file": "src/attr/_make.py",
        "target_fn": "validator"
    },
    {
        "name": "click",
        "bug_type": "B",
        "target_file": "src/click/core.py",
        "target_fn": "_format_deprecated_suffix"
    },
    {
        "name": "returns",
        "bug_type": "C",
        "target_file": "returns/contrib/pytest/plugin.py",
        "target_fn": "_trace_function"
    },
    {
        "name": "marshmallow",
        "bug_type": "C",
        "target_file": "src/marshmallow/schema.py",
        "target_fn": "_run_validator"
    },
    {
        "name": "deepdiff",
        "bug_type": "C",
        "target_file": "deepdiff/diff.py",
        "target_fn": "_diff_uuids"
    },
    {
        "name": "sortedcontainers",
        "bug_type": "C",
        "target_file": "src/sortedcontainers/sortedlist.py",
        "target_fn": "bisect_left"
    },
    {
        "name": "glom",
        "bug_type": "C",
        "target_file": "glom/core.py",
        "target_fn": "_unpack_stack"
    },
    {
        "name": "typer",
        "bug_type": "D",
        "target_file": "typer/main.py",
        "target_fn": "solve_typer_info_defaults"
    },
    {
        "name": "funcy",
        "bug_type": "D",
        "target_file": "funcy/colls.py",
        "target_fn": "zipdict"
    },
    {
        "name": "python-dateutil",
        "bug_type": "D",
        "target_file": "src/dateutil/relativedelta.py",
        "target_fn": "__init__"
    },
    {
        "name": "structlog",
        "bug_type": "D",
        "target_file": "src/structlog/stdlib.py",
        "target_fn": "filter_by_level"
    },
    {
        "name": "apscheduler",
        "bug_type": "D",
        "target_file": "src/apscheduler/triggers/interval.py",
        "target_fn": "__setstate__"
    }
]