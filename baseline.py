
import os

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
if __name__ == "__main__":
    print("=== Starting Control A: Capability Baseline Check ===")
    for repo in REPOS:
        repo_path = f"repos/{repo['name']}"
        target_filepath = os.path.join(repo_path, repo["target_file"])
        
        if not os.path.exists(target_filepath):
            continue
            
        original_content = read(target_filepath)
        buggy_fn = apply_mutation(original_content, repo["target_fn"], repo["bug_type"])
        
        for model_name, model_cfg in MODELS.items():
            prompt = build_prompt(buggy_fn, [], os.path.basename(target_filepath))
            result = call_model(get_local_runtime(model_name), prompt)
            
            if not tests_pass(repo, result):
                EXCLUDED.add((repo['name'], model_name))
                print(f"EXCLUDED: {repo['name']} x {model_name} — capability failure")