"""Pre-flight check: verify all repos pass their test suites with zero failures."""
import sys
from pathlib import Path
from config import REPOS
from injector import run_pytest, repo_has_native_extensions

REPOS_DIR = Path("repos")


def verify_all() -> bool:
    all_pass = True
    print(f"Checking {len(REPOS)} repos...\n")
    for repo in REPOS:
        repo_path = REPOS_DIR / repo["name"]
        if not repo_path.exists():
            print(f"  MISSING  {repo['name']} (path not found: {repo_path})")
            all_pass = False
            continue
        if repo_has_native_extensions(repo_path):
            print(f"  SKIP     {repo['name']} (native extensions detected)")
            continue
        failures = run_pytest(repo_path)
        if not failures:
            print(f"  PASS     {repo['name']}")
        else:
            print(f"  FAIL     {repo['name']} — {len(failures)} failing test(s):")
            for f in failures[:3]:
                print(f"           {f.get('nodeid', '?')}")
            if len(failures) > 3:
                print(f"           ... and {len(failures) - 3} more")
            all_pass = False
    return all_pass


if __name__ == "__main__":
    ok = verify_all()
    print(f"\n{'All repos clean.' if ok else 'Some repos have pre-existing failures — fix before running the experiment.'}")
    sys.exit(0 if ok else 1)
