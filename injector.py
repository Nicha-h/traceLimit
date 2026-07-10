# injector.py
import os
import sys
import glob
import json
import subprocess
import shutil
from typing import Optional
import libcst as cst
from mutation import apply_mutation
from config import REPOS, DEPTHS

# ==========================================
# 1. FILE SYSTEM & UTILITY HELPERS
# ==========================================

def collect_python_files(repo_path: str) -> list:
    """
    Finds and systematically sorts all target python files within the repo path
    to guarantee a completely deterministic file sequence.
    """
    pattern = os.path.join(repo_path, "**", "*.py")
    files = glob.glob(pattern, recursive=True)
    return sorted([os.path.abspath(f) for f in files])


def repo_has_native_extensions(repo_path: str) -> bool:
    """
    Verifies that a target repo is pure Python by checking for compiled extensions.
    """
    for root, _, filenames in os.walk(repo_path):
        for filename in filenames:
            if filename.endswith((".so", ".pyd")):
                return True
    return False

def read(filepath: str) -> str:
    """Reads raw contents of a file safely, ignoring decoding anomalies."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def concatenate(repo_path: str, files: list, replacements: Optional[dict] = None) -> str:
    """Merges independent source files into a unified traceable context payload."""
    delimiter = "\n\n# " + "="*40 + "\n"
    payload = []
    for file_path in files:
        relative_path = os.path.relpath(file_path, repo_path).replace("\\", "/")
        content = replacements.get(file_path, read(file_path)) if replacements else read(file_path)
        payload.append(f"# FILE: {relative_path}\n{content}")
    return delimiter.join(payload)


def _build_padding(size: int) -> str:
    if size <= 0:
        return ""

    unit = "\n# PADDING BUFFER BLOCK " + "-" * 80 + "\n"
    repeats, remainder = divmod(size, len(unit))
    return (unit * repeats) + unit[:remainder]

def pad_to_depth(full_context: str, target_anchor: int, depth: float) -> str:
    """
    Pads the context so the target anchor lands at the requested fractional depth.
    """
    total_len = len(full_context)
    if total_len == 0:
        return full_context

    depth = max(0.0, min(1.0, depth))
    current_depth = target_anchor / total_len

    if depth <= 0.0:
        return full_context + _build_padding(total_len * 50)

    if depth >= 1.0:
        return _build_padding(total_len * 50) + full_context

    if depth > current_depth:
        prefix = int(round((depth * total_len - target_anchor) / max(1e-9, 1.0 - depth)))
        return _build_padding(prefix) + full_context

    suffix = int(round((target_anchor / max(1e-9, depth)) - total_len))
    return full_context + _build_padding(suffix)

# ==========================================
# 2. CONCRETE SYNTAX TREE (CST) MUTATIONS
# ==========================================

class BugTransformer(cst.CSTTransformer):
    """
    Traverses a module's Concrete Syntax Tree to apply format-preserving mutations 
    strictly inside the target function scope.
    """
    def __init__(self, target_fn: str, bug_type: str):
        self.target_fn = target_fn
        self.bug_type = bug_type
        self.in_target = False

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        if node.name.value == self.target_fn:
            self.in_target = True
        return True

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        if original_node.name.value == self.target_fn:
            self.in_target = False
        return updated_node

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.CSTNode:
        # Type A Mutation: Off-by-one bug targeting loop bounds
        if self.in_target and self.bug_type == "A" and isinstance(updated_node.func, cst.Name) and updated_node.func.value == "range":
            if updated_node.args:
                orig_arg = updated_node.args[0].value
                new_arg = cst.BinaryOperation(
                    left=orig_arg,
                    operator=cst.Minus(),
                    right=cst.Integer(value="1")
                )
                return updated_node.with_changes(args=[cst.Arg(value=new_arg)])
        return updated_node

    def leave_Comparison(self, original_node: cst.Comparison, updated_node: cst.Comparison) -> cst.CSTNode:
        # Type C Mutation: Operator Swap (Inverting equality operators)
        if self.in_target and self.bug_type == "C":
            new_ops = []
            for op in updated_node.operators:
                if isinstance(op.operator, cst.Equal):
                    new_ops.append(op.with_changes(operator=cst.NotEqual()))
                elif isinstance(op.operator, cst.NotEqual):
                    new_ops.append(op.with_changes(operator=cst.Equal()))
                else:
                    new_ops.append(op)
            return updated_node.with_changes(operators=new_ops)
        return updated_node


def apply_mutation(file_content: str, target_fn: str, bug_type: str) -> str:
    """Parses source code into a CST structure, runs mutations, and returns format-preserved source."""
    try:
        source_tree = cst.parse_module(file_content)
        transformer = BugTransformer(target_fn, bug_type)
        modified_tree = source_tree.visit(transformer)
        return modified_tree.code
    except Exception as e:
        print(f"CST parsing compilation failure: {e}")
        return file_content


def inject_and_extract_function(repo_config: dict) -> str:
    """Loads the target file for a repo, applies the configured mutation, and returns the mutated source."""
    repo_path = os.path.join("repos", repo_config["name"])
    target_filepath = os.path.join(repo_path, repo_config["target_file"])
    original_code = read(target_filepath)
    return apply_mutation(original_code, repo_config["target_fn"], repo_config["bug_type"])


def build_context(repo_path: str, target_file: str, target_fn: str, bug_type: str, depth: float, inject_bug: bool = True) -> str:
    """Builds a padded multi-file context, optionally with the injected bug embedded."""
    files = collect_python_files(repo_path)
    actual_target_path = os.path.join(repo_path, target_file)
    replacement_code = apply_mutation(read(actual_target_path), target_fn, bug_type) if inject_bug else read(actual_target_path)
    context = concatenate(repo_path, files, replacements={os.path.abspath(actual_target_path): replacement_code})
    target_header = f"# FILE: {os.path.relpath(actual_target_path, repo_path).replace('\\', '/')}\n"
    target_anchor = context.index(target_header)
    return pad_to_depth(context, target_anchor, depth)

# ==========================================
# 3. TEST RUNNER SUBSYSTEM
# ==========================================

def run_pytest(repo_path: str, mutated_file: str = None) -> list:
    """Executes pytest dynamically inside the target directory and parses structural test outcomes."""
    if shutil.which("docker") is not None:
        project_root = os.path.abspath(os.path.dirname(__file__))
        repo_abs = os.path.abspath(repo_path)
        container_repo = os.path.join("/workspace", os.path.relpath(repo_abs, project_root)).replace("\\", "/")
        report_path = os.path.join(container_repo, ".pytest_report.json")
        docker_cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{project_root.replace('\\', '/')}:" + "/workspace",
            "-w",
            container_repo,
            os.getenv("TRACELIMIT_DOCKER_IMAGE", "python:3.11-slim"),
            "bash",
            "-lc",
            (
                "python -m pip install -q pytest pytest-json-report && "
                f"pytest --json-report --json-report-file={report_path} -q"
            ),
        ]
        try:
            result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=120)
            host_report = os.path.join(repo_abs, ".pytest_report.json")
            if os.path.exists(host_report):
                with open(host_report, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return [t for t in data.get("tests", []) if t.get("outcome") == "failed"]
            stderr = (result.stderr or "").strip()
            return [{"outcome": "error", "message": stderr or "pytest report missing"}]
        except subprocess.TimeoutExpired:
            return [{"outcome": "error", "message": "Test execution timed out"}]
        except Exception as e:
            return [{"outcome": "error", "message": str(e)}]

    report_path = os.path.join(repo_path, ".pytest_report.json")
    
    # Clean up stale structural tracking reports if they exist
    if os.path.exists(report_path):
        os.remove(report_path)
        
    cmd = ["pytest", f"--json-report", f"--json-report-file={report_path}", "-q"]
    
    try:
        # Strict 60s execution bounds budget per repository standard run loop
        result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True, timeout=60)
        
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                data = json.load(f)
            failures = [t for t in data.get("tests", []) if t.get("outcome") == "failed"]
            return failures
        stderr = (result.stderr or "").strip()
        return [{"outcome": "error", "message": stderr or "pytest report missing"}]
    except subprocess.TimeoutExpired:
        return [{"outcome": "error", "message": "Test execution timed out"}]
    except Exception as e:
        return [{"outcome": "error", "message": str(e)}]

# ==========================================
# 4. CORE METHODOLOGY FUNCTIONS
# ==========================================

def inject_at_depth(repo_path, target_file, target_fn, bug_type, depth):
    """Constructs the full padded codebase context window payload and injects mutations."""
    actual_target_path = os.path.join(repo_path, target_file)
    mutated = apply_mutation(read(actual_target_path), target_fn, bug_type)  # libcst
    context = build_context(repo_path, target_file, target_fn, bug_type, depth, inject_bug=True)
    return context, mutated

def validate(repo_path, target_filepath, mutated_file):
    """Mandatory evaluation gate checking for valid mutation footprint signal limits."""
    if repo_has_native_extensions(repo_path):
        raise AssertionError(f"Native extensions found in {repo_path}; repo must be pure Python")

    original_backup = read(target_filepath)
    with open(target_filepath, "w", encoding="utf-8") as f:
        f.write(mutated_file)

    failures = run_pytest(repo_path, mutated_file)
    try:
        failing_tests = [f for f in failures if f.get("outcome", "failed") == "failed"]
        assert 1 <= len(failing_tests) <= 10, (
            f"Mutation invalid: {len(failures)} test failures "
            f"(need 1-10 for clean signal)"
        )
        return failing_tests
    finally:
        with open(target_filepath, "w", encoding="utf-8") as f:
            f.write(original_backup)

# ==========================================
# 5. PRE-FLIGHT DRY-RUN TERMINAL TRIGGER
# ==========================================

if __name__ == "__main__":
    if "--dry-run" in sys.argv:
        print("="*60)
        print("TRACELIMIT PRE-FLIGHT INJECTION RUNTIME AUDIT")
        print("="*60)
        
        for repo in REPOS:
            name = repo["name"]
            r_path = f"repos/{name}"
            t_file = repo["target_file"]
            t_fn = repo["target_fn"]
            b_type = repo["bug_type"]
            
            full_target_filepath = os.path.join(r_path, t_file)
            
            if not os.path.exists(full_target_filepath):
                print(f"[-] {name.upper()}: Target file missing at {full_target_filepath}. Did you run git clone?")
                continue
                
            print(f"[*] Testing Mutation on {name.upper()} (Type {b_type})...")
            
            # Read backup state 
            original_code = read(full_target_filepath)
            
            # Apply script transformation loop
            mutated_code = apply_mutation(original_code, t_fn, b_type)
            
            if original_code == mutated_code:
                print(f"    [!] Warning: CST didn't modify any functions. Check your target_fn name matching.")
                continue
                
            try:
                # Target gate verification check
                test_failures = validate(r_path, full_target_filepath, mutated_code)
                print(f"    [+] PASS: Gated validation succeeded. Found {len(test_failures)} test failures.")
            except AssertionError as error:
                print(f"    [-] FAIL: {error}")
            finally:
                # validate() restores the codebase state after each trial.
                pass
        print("\nPre-flight checks complete.")