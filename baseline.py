
import csv
import json
import os
from pathlib import Path

from call_model import call_model, get_local_runtime
from config import EXCLUDED, MODELS, REPOS
from helpers import build_prompt
from injector import read, run_pytest, apply_mutation, repo_has_native_extensions

def tests_pass(repo_config: dict, fixed_code: str) -> bool:
    """
    Overwrites the target file with the model's fix, runs pytest, 
    and checks if the codebase is completely healthy again.
    """
    repo_path = f"repos/{repo_config['name']}"
    target_filepath = os.path.join(repo_path, repo_config["target_file"])

    if repo_has_native_extensions(repo_path):
        raise AssertionError(f"Native extensions found in {repo_path}; repo must be pure Python")
    
    # Backup original code state before overwriting
    original_backup = read(target_filepath)
        
    try:
        with open(target_filepath, "w", encoding="utf-8") as f:
            f.write(fixed_code)
            
        failures = run_pytest(repo_path)
        return len(failures) == 0  # True if all tests pass
    finally:
        # Always restore codebase to original healthy state
        with open(target_filepath, "w", encoding="utf-8") as f:
            f.write(original_backup)
_RESULTS_PATH = Path("results/raw_results.csv")
_FIELDNAMES = ["repo", "model", "depth", "bug_type", "success", "context_tokens", "control", "repo_runtime_seconds"]


def _append_control_a_row(row: dict):
    _RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not _RESULTS_PATH.exists()
    with open(_RESULTS_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    print("=== Starting Control A: Capability Baseline Check ===")

    # Resume: skip (repo, model) pairs that already have a Control A row in CSV.
    control_a_done = set()
    if _RESULTS_PATH.exists():
        with open(_RESULTS_PATH, newline="") as f:
            for row in csv.DictReader(f):
                if row["control"] == "A":
                    control_a_done.add((row["repo"], row["model"]))
    if control_a_done:
        print(f"Resuming Control A: {len(control_a_done)} pairs already recorded")

    for repo in REPOS:
        repo_path = f"repos/{repo['name']}"
        target_filepath = os.path.join(repo_path, repo["target_file"])

        if not os.path.exists(target_filepath):
            continue

        original_content = read(target_filepath)
        buggy_fn = apply_mutation(original_content, repo["target_fn"], repo["bug_type"])

        for model_name, model_cfg in MODELS.items():
            if (repo["name"], model_name) in control_a_done:
                print(f"  Skipping {repo['name']} x {model_name} — already recorded")
                continue

            prompt = build_prompt(buggy_fn, [], os.path.basename(target_filepath))
            result = call_model(get_local_runtime(model_name), prompt)

            passed = tests_pass(repo, result)
            if not passed:
                EXCLUDED.add((repo['name'], model_name))
                print(f"EXCLUDED: {repo['name']} x {model_name} — capability failure")

            _append_control_a_row({
                "repo": repo["name"], "model": model_name, "depth": None,
                "bug_type": repo["bug_type"], "success": int(passed),
                "context_tokens": None, "control": "A", "repo_runtime_seconds": None,
            })
            control_a_done.add((repo["name"], model_name))

    exclusions_path = Path("results/exclusions.json")
    exclusions_path.parent.mkdir(parents=True, exist_ok=True)
    exclusions_path.write_text(
        json.dumps(sorted([list(pair) for pair in EXCLUDED]), indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(EXCLUDED)} exclusions to results/exclusions.json")