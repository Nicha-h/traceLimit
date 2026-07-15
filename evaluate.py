import csv
import json
import os
import time
from pathlib import Path

import config
from config import REPOS, DEPTHS, MODELS, EXCLUDED, RATE_LIMITS
from injector import inject_at_depth, validate, build_context, repo_has_native_extensions, InjectionGateError
from call_model import call_model, get_local_runtime, is_depth_safe
from baseline import tests_pass
from helpers import extract_code_block, count_tokens, build_prompt


# ---------------------------------------------------------------------------
# Checkpoint helpers (Fix 3)
# ---------------------------------------------------------------------------

results_path = Path("results/raw_results.csv")

_FIELDNAMES = [
    "repo", "model", "depth", "bug_type", "success",
    "context_tokens", "control", "repo_runtime_seconds",
]


def _append_row(row: dict):
    write_header = not results_path.exists()
    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Fix 3: load already-completed rows so we can resume after a crash
    # ------------------------------------------------------------------
    # Key is (repo, model, depth_or_None, control_str) to avoid collisions
    # between Control B rows and experimental rows at the same depth.
    completed = set()
    all_results = []
    if results_path.exists():
        with open(results_path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    depth_val = float(row["depth"]) if row["depth"] else None
                except ValueError:
                    depth_val = None
                completed.add((row["repo"], row["model"], depth_val, row["control"]))
        print(f"Resuming: {len(completed)} trials already completed")

    # ------------------------------------------------------------------
    # Fix 1: load exclusions written by baseline.py
    # ------------------------------------------------------------------
    exclusions_path = Path("results/exclusions.json")
    if exclusions_path.exists():
        with open(exclusions_path) as f:
            for pair in json.load(f):
                config.EXCLUDED.add(tuple(pair))
        print(f"Loaded {len(config.EXCLUDED)} exclusions from {exclusions_path}")

    # Ensure results directory exists before any writes
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Fix 2: Control B — hallucination checks across all 20 repos
    # ------------------------------------------------------------------
    print("\n=== Control B: Hallucination checks ===")
    first_model = next(iter(MODELS))
    for repo in REPOS:
        repo_path = Path("repos") / repo["name"]
        if not repo_path.exists() or repo_has_native_extensions(str(repo_path)):
            continue
        if (repo["name"], first_model, 0.50, "B") in completed:
            print(f"  Control B {repo['name']}: skipped (already recorded)")
            continue
        context = build_context(
            str(repo_path),
            repo["target_file"],
            repo["target_fn"],
            repo["bug_type"],
            0.50,
            inject_bug=False,
        )
        runtime = get_local_runtime(first_model)
        prompt = build_prompt(context, [], os.path.basename(repo["target_file"]))
        response = call_model(runtime, prompt)
        fixed_code = extract_code_block(response)
        passed = tests_pass(repo, fixed_code)
        token_count = count_tokens(context)
        row = {
            "repo": repo["name"],
            "model": first_model,
            "depth": 0.50,
            "bug_type": repo["bug_type"],
            "success": int(passed),
            "context_tokens": token_count,
            "control": "B",
            "repo_runtime_seconds": None,
        }
        all_results.append(row)
        _append_row(row)
        completed.add((repo["name"], first_model, 0.50, "B"))
        print(f"  Control B {repo['name']}: {'PASS' if passed else 'FAIL'}")

    # ------------------------------------------------------------------
    # Main experiment loop
    # ------------------------------------------------------------------
    print("\n=== Starting Main Experimentation Loop ===")

    for repo in REPOS:
        repo_path = f"repos/{repo['name']}"
        target_filepath = os.path.join(repo_path, repo["target_file"])

        if not os.path.exists(repo_path):
            print(f"[-] Repository directory not found: {repo_path}. Skipping.")
            continue

        if not os.path.exists(target_filepath):
            print(f"[-] Target file missing: {target_filepath}. Skipping.")
            continue

        if repo_has_native_extensions(repo_path):
            print(f"[-] Native extensions found in {repo_path}. Skipping.")
            continue

        repo_start = time.perf_counter()
        repo_rows = []

        for depth in DEPTHS:
            # Fix 3: skip already-completed (repo, model, depth) combinations
            # We check per-model below, but skip building context if ALL models
            # for this depth are already done.
            all_models_done = all(
                (repo["name"], model_name, depth, "") in completed
                for model_name in MODELS
            )
            if all_models_done:
                continue

            # Build the physical multi-file padding context stack
            context, mutated = inject_at_depth(repo_path, repo["target_file"], repo["target_fn"], repo["bug_type"], depth)

            try:
                # Execution validation gate check
                failures = validate(repo_path, target_filepath, mutated)
            except (AssertionError, InjectionGateError) as gate_err:
                # Abort if the test fail count doesn't land in the 1-10 range
                print(f"[SKIP] {repo['name']} depth={depth:.2f}: {gate_err}")
                continue

            for model_name, model_cfg in MODELS.items():
                # Fix 3: skip this specific (repo, model, depth) if already done
                if (repo["name"], model_name, depth, "") in completed:
                    continue

                if (repo['name'], model_name) in EXCLUDED:
                    continue

                if not is_depth_safe(model_name, depth):
                    continue

                # Format payload text structure dynamically
                prompt_payload = build_prompt(context, failures, os.path.basename(repo["target_file"]))

                # Execute API completion
                response = call_model(get_local_runtime(model_name), prompt_payload)

                # Process outputs via your helpers module
                fixed_code = extract_code_block(response)

                # Check patch correctness state
                success = tests_pass(repo, fixed_code)

                row = {
                    "repo": repo["name"],
                    "model": model_name,
                    "depth": depth,
                    "bug_type": repo["bug_type"],
                    "success": int(success),
                    "context_tokens": count_tokens(context),
                    "control": None,
                    "repo_runtime_seconds": None,
                }
                repo_rows.append(row)
                _append_row(row)  # Fix 3: persist immediately after each call
                completed.add((repo["name"], model_name, depth, ""))

                # Strict rate limit compliance handling
                time.sleep(RATE_LIMITS[model_name]["sleep"])

        repo_runtime_seconds = round(time.perf_counter() - repo_start, 6)
        for row in repo_rows:
            row["repo_runtime_seconds"] = repo_runtime_seconds
        all_results.extend(repo_rows)

    print("Evaluation loop complete. Data saved to results/raw_results.csv")
