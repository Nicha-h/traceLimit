# config.py
import os
from dotenv import load_dotenv

# Load environment variables from the root .env file
load_dotenv()

# Depth points to be tested across the context window
DEPTHS = [0.00, 0.05, 0.25, 0.50, 0.75, 0.95, 1.00]

# Global runtime tracking set for (repo_name, model_name) pairs 
# that fail Control A (capability isolation check)
EXCLUDED = set()

# Model configurations mapped to their specific API backends
MODELS = {
    "llama-3.1-8b-instruct": {
        "hf_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "max_new_tokens": 4096,
        "temperature": 0.0,
    },
    "qwen-2.5-7b-instruct": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "max_new_tokens": 4096,
        "temperature": 0.0,
    },
    "phi-3.5-mini-instruct": {
        "hf_id": "microsoft/Phi-3.5-mini-instruct",
        "max_new_tokens": 4096,
        "temperature": 0.0,
    },
}

RATE_LIMITS = {
    "llama-3.1-8b-instruct": {"sleep": 0},
    "qwen-2.5-7b-instruct": {"sleep": 0},
    "phi-3.5-mini-instruct": {"sleep": 0},
}

# Full 20-repository dataset meticulously mapped by type balance (5 repos per type)
REPOS = [
    # --- TYPE A: Off-by-one ---
    {
        "name": "isort",
        "bug_type": "A",
        "target_file": "isort/place.py",
        "target_fn": "_output"
    },
    {
        "name": "httpx",
        "bug_type": "A",
        "target_file": "httpx/_models.py",
        "target_fn": "redirect_counter" # Tracks redirect counter boundaries
    },
    {
        "name": "arrow",
        "bug_type": "A",
        "target_file": "arrow/arrow.py",
        "target_fn": "shift"
    },
    {
        "name": "loguru",
        "bug_type": "A",
        "target_file": "loguru/_logger.py",
        "target_fn": "_log"
    },
    {
        "name": "returns",
        "bug_type": "A",
        "target_file": "returns/pipeline/pipe.py",
        "target_fn": "pipe" # Index boundary tracking
    },

    # --- TYPE B: Boolean Flip ---
    {
        "name": "rich",
        "bug_type": "B",
        "target_file": "rich/style.py",
        "target_fn": "render_text"
    },
    {
        "name": "cerberus",
        "bug_type": "B",
        "target_file": "cerberus/validator.py",
        "target_fn": "_validate_type"
    },
    {
        "name": "cattrs",
        "bug_type": "B",
        "target_file": "cattrs/converters.py",
        "target_fn": "structure"
    },
    {
        "name": "attrs",
        "bug_type": "B",
        "target_file": "attr/_make.py",
        "target_fn": "validator" # Targets the 'on=' flag
    },
    {
        "name": "click",
        "bug_type": "B",
        "target_file": "click/core.py",
        "target_fn": "make_context"
    },

    # --- TYPE C: Operator Swap ---
    {
        "name": "marshmallow",
        "bug_type": "C",
        "target_file": "marshmallow/schema.py",
        "target_fn": "handle_error"
    },
    {
        "name": "deepdiff",
        "bug_type": "C",
        "target_file": "deepdiff/diff.py",
        "target_fn": "compare" # Targets comparison branches
    },
    {
        "name": "sortedcontainers",
        "bug_type": "C",
        "target_file": "sortedcontainers/sortedlist.py",
        "target_fn": "bisect" # Targets comparison logic
    },
    {
        "name": "more-itertools",
        "bug_type": "C",
        "target_file": "more_itertools/more.py",
        "target_fn": "windowed"
    },
    {
        "name": "glom",
        "bug_type": "C",
        "target_file": "glom/core.py",
        "target_fn": "resolve" # Path resolution operator swapping
    },

    # --- TYPE D: Wrong Variable ---
    {
        "name": "typer",
        "bug_type": "D",
        "target_file": "typer/main.py",
        "target_fn": "param_decls" # Swaps decl variables
    },
    {
        "name": "funcy",
        "bug_type": "D",
        "target_file": "funcy/calc.py",
        "target_fn": "accumulate"
    },
    {
        "name": "python-dateutil",
        "bug_type": "D",
        "target_file": "dateutil/relativedelta.py",
        "target_fn": "relativedelta" # Swaps years/months variables
    },
    {
        "name": "structlog",
        "bug_type": "D",
        "target_file": "structlog/processors.py",
        "target_fn": "exc_info" # Swaps info/exc variables
    },
    {
        "name": "apscheduler",
        "bug_type": "D",
        "target_file": "apscheduler/triggers/interval.py",
        "target_fn": "__init__" # Swaps start_date/end_date variables
    }
]