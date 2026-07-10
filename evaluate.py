import os
import time
import pandas as pd
from config import REPOS, DEPTHS, MODELS, EXCLUDED, RATE_LIMITS
from injector import inject_at_depth, validate, build_context, repo_has_native_extensions
from call_model import call_model, get_local_runtime
from baseline import tests_pass
from helpers import extract_code_block, count_tokens, build_prompt


if __name__ == "__main__":
    results = []

    print("=== Starting Main Experimentation Loop ===")

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
        control_start = time.perf_counter()
        control_b_context = build_context(
            control_b_repo_path,
            control_b_repo["target_file"],
            control_b_repo["target_fn"],
            control_b_repo["bug_type"],
            0.50,
            inject_bug=False,
        )
        control_b_prompt = build_prompt(control_b_context, [], os.path.basename(control_b_repo["target_file"]))
        control_b_response = call_model(get_local_runtime(control_model_name), control_b_prompt)
        control_b_fixed_code = extract_code_block(control_b_response)
        control_b_success = tests_pass(control_b_repo, control_b_fixed_code)
        results.append({
            "repo": control_b_repo["name"],
            "model": control_model_name,
            "depth": 0.50,
            "bug_type": control_b_repo["bug_type"],
            "success": int(control_b_success),
            "context_tokens": count_tokens(control_b_context),
            "control": "B",
            "repo_runtime_seconds": round(time.perf_counter() - control_start, 6),
        })

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
            # Build the physical multi-file padding context stack
            context, mutated = inject_at_depth(repo_path, repo["target_file"], repo["target_fn"], repo["bug_type"], depth)
            
            try:
                # Execution validation gate check
                failures = validate(repo_path, target_filepath, mutated)
            except AssertionError:
                # Abort if the test fail count doesn't land perfectly in the 1-10 range
                continue

            for model_name, model_cfg in MODELS.items():
                if (repo['name'], model_name) in EXCLUDED:
                    continue

                # Format payload text structure dynamically
                prompt_payload = build_prompt(context, failures, os.path.basename(repo["target_file"]))
                
                # Execute API completion
                response = call_model(get_local_runtime(model_name), prompt_payload)
                
                # Process outputs via your helpers module
                fixed_code = extract_code_block(response)
                
                # Check patch correctness state
                success = tests_pass(repo, fixed_code)

                repo_rows.append({
                    "repo": repo["name"], 
                    "model": model_name,
                    "depth": depth, 
                    "bug_type": repo["bug_type"],
                    "success": int(success),
                    "context_tokens": count_tokens(context),
                })

                # Strict rate limit compliance handling
                time.sleep(RATE_LIMITS[model_name]["sleep"])

        repo_runtime_seconds = round(time.perf_counter() - repo_start, 6)
        for row in repo_rows:
            row["repo_runtime_seconds"] = repo_runtime_seconds
        results.extend(repo_rows)

    # Persistent storage serialization loop
    os.makedirs("results", exist_ok=True)
    pd.DataFrame(results).to_csv("results/raw_results.csv", index=False)
    print("Evaluation loop complete. Data saved to results/raw_results.csv")