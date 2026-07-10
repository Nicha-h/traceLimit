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
            """\
from pathlib import Path
import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt

DATASET_ROOT = Path('/kaggle/input/Repository')
if DATASET_ROOT.exists():
    sys.path.append(str(DATASET_ROOT))
    os.environ.setdefault('TRACELIMIT_DATASET_ROOT', str(DATASET_ROOT))
REPO_ROOT = DATASET_ROOT / 'repos' if DATASET_ROOT.exists() else Path('repos')

from config import DEPTHS, MODELS, RATE_LIMITS, REPOS
from injector import build_context, extract_target_function, inject_at_depth, repo_has_native_extensions, validate
from call_model import call_model, get_local_runtime
from baseline import tests_pass
from helpers import build_prompt, count_tokens, extract_code_block

ROOT = Path.cwd()
print(f"Working from {ROOT}")
print("Dataset root:", DATASET_ROOT if DATASET_ROOT.exists() else 'not mounted')
print("Repo root:", REPO_ROOT)
print("Loaded models:", ", ".join(MODELS))"""
        ),
        markdown(
            """\
### Long-context setup

When you test lost-in-the-middle behavior with local inference, the prompt for the 20 repos can dominate VRAM.
Use Flash Attention 2 and a quantized load path so the KV cache has room to grow.

- Install `flash-attn` inside the notebook environment before loading the model.
- Prefer 4-bit or 8-bit loading with `bitsandbytes` when the runtime is memory constrained.
- If you use vLLM, keep the model quantized and lower the KV-cache pressure before running long prompts."""
        ),
        code(
            """\
# Kaggle local inference setup for long-context experiments.

%pip install flash-attn --no-build-isolation
%pip install bitsandbytes transformers accelerate vllm sentencepiece

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_LOADERS = {
    'llama-3.1-8b-instruct': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'qwen-2.5-7b-instruct': 'Qwen/Qwen2.5-7B-Instruct',
    'phi-3.5-mini-instruct': 'microsoft/Phi-3.5-mini-instruct',
}

def load_local_model(model_key: str):
    model_id = MODEL_LOADERS[model_key]
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype='bfloat16',
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation='flash_attention_2',
        quantization_config=quantization_config,
        device_map='auto',
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return {'model': model, 'tokenizer': tokenizer, 'max_new_tokens': 4096}

LOCAL_RUNTIME = {}

def get_local_runtime(model_key: str):
    runtime = LOCAL_RUNTIME.get(model_key)
    if runtime is None:
        runtime = load_local_model(model_key)
        LOCAL_RUNTIME[model_key] = runtime
    return runtime"""
        ),
        markdown("## Injectors"),
        code(
            """\
def inspect_injection(repo_name: str, depth: float = 0.5):
    repo = next(item for item in REPOS if item["name"] == repo_name)
    repo_path = str(REPO_ROOT / repo['name'])
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
        markdown("## Control A — Baseline Gate"),
        code(
            """\
def run_baseline_gate():
    excluded = set()
    for repo in REPOS:
        repo_path = str(REPO_ROOT / repo["name"])
        if not os.path.exists(repo_path):
            continue
        buggy_fn = extract_target_function(repo_path, repo["target_file"], repo["target_fn"])
        for model_name in MODELS:
            prompt = f"Fix the bug in this function. Return only the corrected function.\\n\\n{buggy_fn}"
            response = call_model(get_local_runtime(model_name), prompt, temperature=0.0)
            fixed = extract_code_block(response)
            if not tests_pass(repo, fixed):
                excluded.add((repo["name"], model_name))
                print(f"EXCLUDED: {repo['name']} x {model_name} — capability failure")
    return excluded

EXCLUDED = run_baseline_gate()
print(f"Total excluded pairs: {len(EXCLUDED)}")"""
        ),
        markdown("## Executions"),
        code(
            """\
def run_experiment():
    results = []
    completed_runs = set()

    os.makedirs("results", exist_ok=True)
    working_results_file = "results/raw_results.csv"
    previous_results_file = os.environ.get(
        "TRACELIMIT_PREVIOUS_RESULTS_FILE",
        "/kaggle/input/Repository/results/raw_results.csv",
    )

    if os.path.exists(previous_results_file) and not os.path.exists(working_results_file):
        import shutil

        shutil.copy(previous_results_file, working_results_file)

    if os.path.exists(working_results_file):
        print("Found existing progress. Loading...")
        existing_df = pd.read_csv(working_results_file)
        results = existing_df.to_dict("records")

        for row in results:
            depth = row.get("depth", 0.50)
            completed_runs.add((row["repo"], row["model"], depth))

        print(f"Skipping {len(completed_runs)} previously completed runs.")

    control_b_repo = next(
        (
            repo
            for repo in REPOS
            if os.path.exists(str(REPO_ROOT / repo['name'])) and not repo_has_native_extensions(str(REPO_ROOT / repo['name']))
        ),
        None,
    )
    if control_b_repo is not None:
        control_model_name = next(iter(MODELS))
        control_b_repo_path = str(REPO_ROOT / control_b_repo['name'])
        control_start = time.perf_counter()
        control_run_key = (control_b_repo["name"], control_model_name, 0.50)
        control_b_context = build_context(
            control_b_repo_path,
            control_b_repo["target_file"],
            control_b_repo["target_fn"],
            control_b_repo["bug_type"],
            0.50,
            inject_bug=False,
        )
        if control_run_key not in completed_runs:
            control_b_prompt = build_prompt(control_b_context, [], os.path.basename(control_b_repo["target_file"]))
            control_b_response = call_model(get_local_runtime(control_model_name), control_b_prompt)
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
                    "repo_runtime_seconds": round(time.perf_counter() - control_start, 6),
                }
            )
            pd.DataFrame(results).to_csv(working_results_file, index=False)
            completed_runs.add(control_run_key)

    for repo in REPOS:
        repo_path = str(REPO_ROOT / repo['name'])
        target_filepath = os.path.join(repo_path, repo["target_file"])

        if not os.path.exists(repo_path) or not os.path.exists(target_filepath):
            continue

        if repo_has_native_extensions(repo_path):
            continue

        repo_start = time.perf_counter()
        repo_rows = []

        for depth in DEPTHS:
            context, mutated = inject_at_depth(repo_path, repo["target_file"], repo["target_fn"], repo["bug_type"], depth)

            try:
                failures = validate(repo_path, target_filepath, mutated)
            except AssertionError:
                continue

            for model_name, model_cfg in MODELS.items():
                if (repo["name"], model_name) in EXCLUDED:
                    continue

                if (repo["name"], model_name, depth) in completed_runs:
                    print(f"Skipping {repo['name']} at depth {depth} for {model_name}")
                    continue

                prompt_payload = build_prompt(context, failures, os.path.basename(repo["target_file"]))
                response = call_model(get_local_runtime(model_name), prompt_payload)
                fixed_code = extract_code_block(response)
                success = tests_pass(repo, fixed_code)

                repo_rows.append(
                    {
                        "repo": repo["name"],
                        "model": model_name,
                        "depth": depth,
                        "bug_type": repo["bug_type"],
                        "success": int(success),
                        "context_tokens": count_tokens(context),
                    }
                )

                completed_runs.add((repo["name"], model_name, depth))
                pd.DataFrame(results + repo_rows).to_csv(working_results_file, index=False)

                time.sleep(RATE_LIMITS[model_name]["sleep"])

        repo_runtime_seconds = round(time.perf_counter() - repo_start, 6)
        for row in repo_rows:
            row["repo_runtime_seconds"] = repo_runtime_seconds
        results.extend(repo_rows)
        pd.DataFrame(results).to_csv(working_results_file, index=False)

    return results


results = run_experiment()
print(f"Collected {len(results)} rows")"""
        ),
        markdown("## Evaluations"),
        code(
            """\
os.makedirs("results", exist_ok=True)
pd.DataFrame(results).to_csv("results/raw_results.csv", index=False)

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

df = pd.DataFrame(results)

# separate control-B rows from experimental rows
if "control" in df.columns:
    df_ctrl = df[df["control"] == "B"].copy()
    df_exp  = df[df["control"].isna()].copy()
else:
    df_ctrl = pd.DataFrame()
    df_exp  = df.copy()

df_exp["failure"] = 1 - df_exp["success"]

print(f"Total experiment rows : {len(df_exp)}")
print(f"Models                : {sorted(df_exp['model'].unique())}")
print(f"Depth values          : {sorted(df_exp['depth'].unique())}")
print(f"Bug types             : {sorted(df_exp['bug_type'].unique())}")
print()

summary = (
    df_exp.groupby("model")["success"]
    .agg(
        runs="count",
        fix_rate="mean",
        failure_rate=lambda x: 1 - x.mean(),
    )
    .round(3)
    .reset_index()
)
print(summary.to_string(index=False))"""
        ),
        markdown(
            """\
### Summary Bar Chart

Overall fix and failure rates per model across all depths and repos."""
        ),
        code(
            """\
# Overall fix / failure rate per model
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Overall Model Performance Summary", fontsize=14, fontweight="bold")

models  = summary["model"].tolist()
x       = np.arange(len(models))
width   = 0.5
colors  = plt.cm.tab10.colors

ax = axes[0]
bars = ax.bar(x, summary["fix_rate"] * 100, width,
              color=colors[:len(models)], edgecolor="white", linewidth=0.8)
ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9)
ax.set_title("Fix Rate (%)")
ax.set_ylabel("Fix Rate (%)")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
ax.set_ylim(0, 115)
ax.grid(axis="y", alpha=0.3)

ax = axes[1]
bars2 = ax.bar(x, summary["failure_rate"] * 100, width,
               color=colors[:len(models)], edgecolor="white", linewidth=0.8)
ax.bar_label(bars2, fmt="%.1f%%", padding=4, fontsize=9)
ax.set_title("Failure Rate (%)")
ax.set_ylabel("Failure Rate (%)")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
ax.set_ylim(0, 115)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("results/summary_bar.png", dpi=150, bbox_inches="tight")
plt.show()"""
        ),
        markdown(
            """\
### Positional Bias Curve

Shows how bug position within the context window affects fix rate — the *lost-in-the-middle* effect appears as a dip near 50%."""
        ),
        code(
            """\
# Positional Bias Curve (lost-in-the-middle)
pivot = df_exp.groupby(["model", "depth"])["success"].mean().reset_index()
pivot["failure_rate"] = 1 - pivot["success"]

fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
fig.suptitle(
    "Positional Bias Curve\\n(bug position in context window vs fix rate)",
    fontsize=14, fontweight="bold",
)

palette = {
    m: plt.cm.tab10.colors[i]
    for i, m in enumerate(sorted(pivot["model"].unique()))
}

for model_name, group in pivot.groupby("model"):
    g   = group.sort_values("depth")
    col = palette[model_name]
    axes[0].plot(g["depth"] * 100, g["success"] * 100,
                 marker="o", linewidth=2.2, color=col, label=model_name, zorder=3)
    axes[1].plot(g["depth"] * 100, g["failure_rate"] * 100,
                 marker="o", linewidth=2.2, color=col, label=model_name, zorder=3)

for ax in axes:
    ax.axvspan(40, 60, alpha=0.09, color="red", label="Mid-context trough zone")
    ax.axvline(50, color="red", linestyle="--", linewidth=0.9, alpha=0.5)
    ax.set_xticks([0, 5, 10, 25, 50, 75, 90, 95, 100])
    ax.set_xlim(-2, 102)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")

axes[0].set_ylabel("Fix Success Rate (%)")
axes[0].set_title("Fix Rate vs Bug Position")
axes[1].set_ylabel("Failure Rate (%)")
axes[1].set_title("Failure Rate vs Bug Position")
axes[1].set_xlabel("Bug Depth in Context Window (%)")

plt.tight_layout()
plt.savefig("results/positional_bias_curve.png", dpi=150, bbox_inches="tight")
plt.show()

print("\\nFailure rates by model x depth:")
fail_tbl = pivot.pivot(index="depth", columns="model", values="failure_rate").round(3)
fail_tbl.index = (fail_tbl.index * 100).astype(int).astype(str) + "%"
fail_tbl.index.name = "depth"
print(fail_tbl.to_string())"""
        ),
        markdown(
            """\
### Failure Rate Heatmap

Model x depth failure rates — dark red = high failure, dark green = low failure."""
        ),
        code(
            """\
# Failure rate heatmap: model x depth
fail_matrix = (
    df_exp.groupby(["depth", "model"])["failure"]
    .mean()
    .unstack("model")
    .sort_index()
)

fig, ax = plt.subplots(figsize=(max(6, len(fail_matrix.columns) * 1.6), 5))
im   = ax.imshow(fail_matrix.values.T, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Failure Rate", fontsize=10)

ax.set_xticks(range(len(fail_matrix.index)))
ax.set_xticklabels([f"{int(d*100)}%" for d in fail_matrix.index], fontsize=9)
ax.set_yticks(range(len(fail_matrix.columns)))
ax.set_yticklabels(fail_matrix.columns, fontsize=9)
ax.set_xlabel("Bug Depth in Context Window")
ax.set_ylabel("Model")
ax.set_title("Failure Rate Heatmap (Model x Depth)", fontsize=13, fontweight="bold")

for i, col in enumerate(fail_matrix.columns):
    for j, depth in enumerate(fail_matrix.index):
        val = fail_matrix.loc[depth, col]
        if not np.isnan(val):
            txt_col = "black" if 0.25 < val < 0.75 else "white"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    fontsize=8, color=txt_col)

plt.tight_layout()
plt.savefig("results/failure_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()"""
        ),
        markdown(
            """\
### Bug-Type Breakdown

Fix rate segmented by bug category and model."""
        ),
        code(
            """\
# Fix rate by bug type
if "bug_type" in df_exp.columns and df_exp["bug_type"].nunique() > 1:
    bug_summary  = (
        df_exp.groupby(["bug_type", "model"])["success"]
        .agg(fix_rate="mean", n="count")
        .reset_index()
    )
    bug_types   = sorted(bug_summary["bug_type"].unique())
    models_list = sorted(bug_summary["model"].unique())
    x           = np.arange(len(bug_types))
    width       = 0.8 / max(len(models_list), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(bug_types) * 2), 5))
    for i, model_name in enumerate(models_list):
        subset = bug_summary[bug_summary["model"] == model_name]
        vals   = [
            subset[subset["bug_type"] == bt]["fix_rate"].values[0]
            if bt in subset["bug_type"].values else np.nan
            for bt in bug_types
        ]
        offset = (i - len(models_list) / 2 + 0.5) * width
        bars   = ax.bar(
            x + offset,
            [v * 100 if not np.isnan(v) else 0 for v in vals],
            width * 0.9, label=model_name, edgecolor="white", linewidth=0.6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(bug_types, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Fix Rate (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Fix Rate by Bug Type and Model", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/bug_type_breakdown.png", dpi=150, bbox_inches="tight")
    plt.show()
else:
    print("Single bug type in dataset — skipping bug-type chart.")"""
        ),
        markdown(
            """\
### Context Size vs Fix Outcome

Scatter of individual runs with a rolling fix-rate trend line."""
        ),
        code(
            """\
# Context size vs fix outcome
if "context_tokens" in df_exp.columns:
    fig, ax = plt.subplots(figsize=(10, 5))
    for model_name, group in df_exp.groupby("model"):
        jitter = np.random.uniform(-0.025, 0.025, len(group))
        ax.scatter(
            group["context_tokens"],
            group["success"] + jitter,
            alpha=0.35, s=18, label=model_name,
        )
    ax.set_xlabel("Context Size (tokens)")
    ax.set_ylabel("Fix Success (jittered)")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Fail", "Pass"])
    ax.set_title("Context Size vs Fix Outcome", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    ax2 = ax.twinx()
    for model_name, group in df_exp.groupby("model"):
        g = group.sort_values("context_tokens")
        if len(g) >= 5:
            rolling = g["success"].rolling(window=5, center=True).mean()
            ax2.plot(g["context_tokens"], rolling * 100,
                     linewidth=2, alpha=0.75, label=f"{model_name} (trend)")
    ax2.set_ylabel("Rolling Fix Rate (%)")
    ax2.set_ylim(-5, 115)
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig("results/context_tokens_scatter.png", dpi=150, bbox_inches="tight")
    plt.show()
else:
    print("No context_tokens column — skipping scatter chart.")"""
        ),
        markdown(
            """\
### Per-Repo Fix Rate

Heatmap of fix rates broken down by individual repository."""
        ),
        code(
            """\
# Per-repo fix rate heatmap
repo_pivot = (
    df_exp.groupby(["repo", "model"])["success"]
    .mean()
    .round(3)
    .unstack("model")
)
print("Fix rate by repo x model:")
print(repo_pivot.to_string())

if len(repo_pivot) > 1:
    fig, ax = plt.subplots(
        figsize=(max(6, len(repo_pivot.columns) * 1.8),
                 max(4, len(repo_pivot) * 0.55 + 1))
    )
    im   = ax.imshow(repo_pivot.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Fix Rate", fontsize=10)
    ax.set_xticks(range(len(repo_pivot.columns)))
    ax.set_xticklabels(repo_pivot.columns, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(repo_pivot.index)))
    ax.set_yticklabels(repo_pivot.index, fontsize=8)
    ax.set_title("Per-Repo Fix Rate (Model x Repo)", fontsize=13, fontweight="bold")
    for i, repo in enumerate(repo_pivot.index):
        for j, model in enumerate(repo_pivot.columns):
            val = repo_pivot.loc[repo, model]
            if not np.isnan(val):
                txt_col = "black" if 0.25 < val < 0.75 else "white"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=8, color=txt_col)
    plt.tight_layout()
    plt.savefig("results/per_repo_heatmap.png", dpi=150, bbox_inches="tight")
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
