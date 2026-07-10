from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results" / "tracelimit_experiment.ipynb"


def _source(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    return lines or [""]


def _cell(cell_type: str, language: str, text: str, *, include_id: bool = True) -> dict:
    metadata = {"language": language}
    if include_id:
        metadata["id"] = uuid.uuid4().hex[:8]

    cell = {
        "cell_type": cell_type,
        "metadata": metadata,
        "source": _source(text),
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def markdown(text: str) -> dict:
    return _cell("markdown", "markdown", text)


def code(text: str) -> dict:
    return _cell("code", "python", text)


def build_notebook() -> dict:
    cells = [
        markdown(
            """# Tracelimit Experiment Notebook

This notebook mirrors the repo workflow in four sections: setup, injectors, executions, and evaluations.
It also includes a long-context note for local model runs where Flash Attention 2 and KV-cache savings matter."""
        ),
        markdown("## Setup"),
        code(
            """from pathlib import Path
import os
import time
import pandas as pd
import matplotlib.pyplot as plt

from config import DEPTHS, EXCLUDED, MODELS, RATE_LIMITS, REPOS
from injector import build_context, inject_at_depth, repo_has_native_extensions, validate
from call_model import call_model
from baseline import tests_pass
from helpers import build_prompt, count_tokens, extract_code_block

ROOT = Path.cwd()
print(f"Working from {ROOT}")
print("Loaded models:", ", ".join(MODELS))"""
        ),
        markdown(
            """### Long-context setup

When you test lost-in-the-middle behavior with local inference, the prompt for the 20 repos can dominate VRAM.
Use Flash Attention 2 and a quantized load path so the KV cache has room to grow.

- Install `flash-attn` inside the notebook environment before loading the model.
- Prefer 4-bit or 8-bit loading with `bitsandbytes` when the runtime is memory constrained.
- If you use vLLM, keep the model quantized and lower the KV-cache pressure before running long prompts."""
        ),
        code(
            """# Optional local inference template for notebook-based long-context experiments.
# Uncomment the lines you need in an environment that already has the required packages.

# !pip install flash-attn --no-build-isolation
# !pip install bitsandbytes transformers accelerate vllm

# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype="bfloat16",
# )
# model = AutoModelForCausalLM.from_pretrained(
#     "meta-llama/Meta-Llama-3.1-8B-Instruct",
#     attn_implementation="flash_attention_2",
#     quantization_config=quantization_config,
#     device_map="auto",
# )"""
        ),
        markdown("## Injectors"),
        code(
            """def inspect_injection(repo_name: str, depth: float = 0.5):
    repo = next(item for item in REPOS if item["name"] == repo_name)
    repo_path = f"repos/{repo['name']}"
    target_filepath = os.path.join(repo_path, repo["target_file"])
    context, mutated = inject_at_depth(
        repo_path,
        repo["target_file"],
        repo["target_fn"],
        repo["bug_type"],
        depth,
    )
    failures = validate(repo_path, target_filepath, mutated)
    return {
        "context_tokens": count_tokens(context),
        "failures": failures,
        "prompt": build_prompt(context, failures, os.path.basename(repo["target_file"])),
        "mutated": mutated,
    }


inspect_injection(REPOS[0]["name"])"""
        ),
        markdown("## Executions"),
        code(
            """def run_experiment():
    results = []

    control_b_repo = next(
        (
            repo
            for repo in REPOS
            if os.path.exists(f"repos/{repo['name']}") and not repo_has_native_extensions(f"repos/{repo['name']}")
        ),
        None,
    )
    if control_b_repo is not None:
        control_model_name = next(iter(MODELS))
        control_b_repo_path = f"repos/{control_b_repo['name']}"
        control_b_context = build_context(
            control_b_repo_path,
            control_b_repo["target_file"],
            control_b_repo["target_fn"],
            control_b_repo["bug_type"],
            0.50,
            inject_bug=False,
        )
        control_b_prompt = build_prompt(control_b_context, [], os.path.basename(control_b_repo["target_file"]))
        control_b_response = call_model(MODELS[control_model_name], control_b_prompt)
        control_b_fixed_code = extract_code_block(control_b_response)
        control_b_success = tests_pass(control_b_repo, control_b_fixed_code)
        results.append(
            {
                "repo": control_b_repo["name"],
                "model": control_model_name,
                "depth": 0.50,
                "bug_type": control_b_repo["bug_type"],
                "success": int(control_b_success),
                "context_tokens": count_tokens(control_b_context),
                "control": "B",
            }
        )

    for repo in REPOS:
        repo_path = f"repos/{repo['name']}"
        target_filepath = os.path.join(repo_path, repo["target_file"])

        if not os.path.exists(repo_path) or not os.path.exists(target_filepath):
            continue

        if repo_has_native_extensions(repo_path):
            continue

        for depth in DEPTHS:
            context, mutated = inject_at_depth(repo_path, repo["target_file"], repo["target_fn"], repo["bug_type"], depth)

            try:
                failures = validate(repo_path, target_filepath, mutated)
            except AssertionError:
                continue

            for model_name, model_cfg in MODELS.items():
                if (repo["name"], model_name) in EXCLUDED:
                    continue

                prompt_payload = build_prompt(context, failures, os.path.basename(repo["target_file"]))
                response = call_model(model_cfg, prompt_payload)
                fixed_code = extract_code_block(response)
                success = tests_pass(repo, fixed_code)

                results.append(
                    {
                        "repo": repo["name"],
                        "model": model_name,
                        "depth": depth,
                        "bug_type": repo["bug_type"],
                        "success": int(success),
                        "context_tokens": count_tokens(context),
                    }
                )

                time.sleep(RATE_LIMITS[model_name]["sleep"])

    return results


results = run_experiment()
print(f"Collected {len(results)} rows")"""
        ),
        markdown("## Evaluations"),
        code(
            """os.makedirs("results", exist_ok=True)
pd.DataFrame(results).to_csv("results/raw_results.csv", index=False)

df = pd.DataFrame(results)
if not df.empty:
    pivot = df.groupby(["model", "depth"])["success"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, group in pivot.groupby("model"):
        ax.plot(group["depth"] * 100, group["success"] * 100, marker="o", linewidth=2, label=model_name)

    ax.axvspan(40, 60, alpha=0.08, color="red", label="Predicted trough zone")
    ax.set_xlabel("Bug depth in context window (%)")
    ax.set_ylabel("Fix success rate (%)")
    ax.set_xticks([0, 5, 25, 50, 75, 95, 100])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()"""
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Tracelimit experiment notebook.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output .ipynb path")
    args = parser.parse_args()

    output_path: Path = args.output
    if output_path.suffix != ".ipynb":
        output_path = output_path.with_suffix(".ipynb")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(build_notebook(), handle, indent=2)
        handle.write("\n")

    print(f"Wrote notebook to {output_path}")


if __name__ == "__main__":
    main()
