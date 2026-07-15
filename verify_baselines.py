"""Verify that baseline exclusions are consistent with experiment results."""
import json
import sys
from pathlib import Path

EXCLUSIONS_PATH = Path("results/exclusions.json")
RESULTS_PATH = Path("results/raw_results.csv")


def verify() -> bool:
    if not EXCLUSIONS_PATH.exists():
        print(f"No exclusions file found at {EXCLUSIONS_PATH}. Run baseline.py first.")
        return False

    with open(EXCLUSIONS_PATH) as f:
        exclusions = [tuple(pair) for pair in json.load(f)]

    print(f"Loaded {len(exclusions)} excluded (repo, model) pairs from {EXCLUSIONS_PATH}\n")

    if not RESULTS_PATH.exists():
        print(f"No results file found at {RESULTS_PATH}. Nothing to cross-check.")
        return True

    import csv
    rows = []
    with open(RESULTS_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    all_ok = True
    for repo_name, model_key in exclusions:
        control_a_rows = [
            r for r in rows
            if r["repo"] == repo_name and r["model"] == model_key and r.get("control") == "A"
        ]
        experiment_rows = [
            r for r in rows
            if r["repo"] == repo_name and r["model"] == model_key and not r.get("control")
        ]

        if not control_a_rows:
            print(f"  WARNING  ({repo_name}, {model_key}): excluded but no Control A row found in CSV")
            all_ok = False
        else:
            successes = [r for r in control_a_rows if r.get("success") == "1"]
            if successes:
                print(f"  WARNING  ({repo_name}, {model_key}): Control A row shows success=1 but pair is excluded")
                all_ok = False

        if experiment_rows:
            print(f"  ERROR    ({repo_name}, {model_key}): excluded pair has {len(experiment_rows)} non-control rows in results — exclusion was not applied!")
            all_ok = False
        else:
            print(f"  OK       ({repo_name}, {model_key})")

    return all_ok


if __name__ == "__main__":
    ok = verify()
    print(f"\n{'Exclusion list is consistent.' if ok else 'Inconsistencies found — review above.'}")
    sys.exit(0 if ok else 1)
