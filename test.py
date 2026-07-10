import os
from injector import run_pytest

def apply_and_test(repo_config: dict, fixed_code: str) -> bool:
    """
    Overwrites the targeted file with the LLM's fix, runs pytest, and checks if all tests pass.
    """
    repo_path = f"repos/{repo_config['name']}"
    target_filepath = os.path.join(repo_path, repo_config["target_file"])
    
    # Backup original code state before overwriting
    with open(target_filepath, "r") as f:
        original_backup = f.read()
        
    try:
        with open(target_filepath, "w") as f:
            f.write(fixed_code)
            
        failures = run_pytest(repo_path)
        # Success means 0 failing tests found
        return len(failures) == 0
    finally:
        # Always restore codebase stability state
        with open(target_filepath, "w") as f:
            f.write(original_backup)