"""Search Kaggle for Indian heart disease datasets and print top candidates.

Setup:
  export KAGGLE_USERNAME=your_user
  export KAGGLE_KEY=your_key

Usage:
  python data/search_kaggle_indian.py --query "heart disease india" --limit 10

This script uses Kaggle API via kaggle package.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys


def ensure_kaggle_installed():
    try:
        import kaggle  # noqa: F401
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"], check=True)


def kaggle_search(query: str, limit: int = 10):
    # Kaggle CLI search (no -x flag; parse space-delimited)
    cmd = ["kaggle", "datasets", "list", "-s", query]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr)
        raise SystemExit("Kaggle search failed. Ensure credentials are set.")
    lines = [l for l in res.stdout.splitlines() if l.strip()]
    items = []
    # Skip header line if present
    for l in lines[1:]:
        parts = l.split()
        if not parts:
            continue
        ref = parts[0]
        title = " ".join(parts[1:])
        items.append({"ref": ref, "title": title})
        if len(items) >= limit:
            break
    return items


def main():
    ap = argparse.ArgumentParser(description="Search Kaggle datasets for Indian heart disease")
    ap.add_argument("--query", default="heart disease india")
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()

    ensure_kaggle_installed()
    items = kaggle_search(args.query, args.limit)
    print(json.dumps(items, indent=2))

if __name__ == "__main__":
    main()
