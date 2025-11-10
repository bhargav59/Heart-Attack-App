"""Utility script to fetch an Indian heart disease dataset.

Supports three strategies (attempted in order unless overridden):
1. Direct URL (set INDIAN_DATASET_URL env or pass --url)
2. Kaggle dataset (set KAGGLE_DATASET env or pass --kaggle <owner/dataset>)
3. Fallback: do nothing (user may supply manually) or combine with synthetic generator.

Usage examples:
    python data/fetch_indian_dataset.py --url https://example.com/indian_heart.csv --out data/indian_heart.csv
    python data/fetch_indian_dataset.py --kaggle ankur6u/heart-disease-dataset --out data/indian_heart.csv

Kaggle requires environment variables:
    KAGGLE_USERNAME, KAGGLE_KEY  (or a kaggle.json in ~/.kaggle/)

Note: Public *India-specific* heart disease datasets are limited. Some Kaggle datasets claim Indian origin but may be mixed.
Always review licensing and provenance before clinical use.
"""
from __future__ import annotations
import argparse
import os
import sys
import subprocess
import json
from pathlib import Path
import requests

DEFAULT_OUT = "data/indian_heart.csv"

def download_url(url: str, out_path: Path):
    print(f"[fetch] Downloading from URL: {url}")
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        raise SystemExit(f"Failed to fetch URL {url}: {r.status_code}")
    out_path.write_bytes(r.content)
    print(f"[fetch] Saved to {out_path}")

def download_kaggle(dataset: str, out_path: Path):
    print(f"[fetch] Downloading Kaggle dataset: {dataset}")
    # Ensure kaggle CLI is available
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"], check=True)
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"Could not install kaggle package: {e}")

    # Create temp dir
    tmp_dir = out_path.parent / "_kaggle_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", str(tmp_dir), "--force"]
    print(f"[fetch] Running: {' '.join(cmd)}")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise SystemExit("Kaggle CLI failed. Ensure KAGGLE_USERNAME and KAGGLE_KEY are set.")

    # Unzip all archives
    for f in tmp_dir.glob("*.zip"):
        subprocess.run(["unzip", "-o", str(f), "-d", str(tmp_dir)])

    # Heuristic: choose first CSV with 'heart' in name
    candidates = list(tmp_dir.glob("*heart*\.csv")) or list(tmp_dir.glob("*.csv"))
    if not candidates:
        raise SystemExit("No CSV files found in Kaggle dataset.")
    src = candidates[0]
    out_path.write_bytes(src.read_bytes())
    print(f"[fetch] Selected {src.name} -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Fetch Indian heart disease dataset")
    ap.add_argument("--url", help="Direct URL to CSV", default=os.getenv("INDIAN_DATASET_URL"))
    ap.add_argument("--kaggle", help="Kaggle dataset spec (owner/dataset)", default=os.getenv("KAGGLE_DATASET"))
    ap.add_argument("--out", help="Output path", default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true", help="Overwrite existing file")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not args.force:
        print(f"[fetch] Output file {out_path} already exists. Use --force to overwrite.")
        return

    if args.url:
        download_url(args.url, out_path)
        return
    if args.kaggle:
        download_kaggle(args.kaggle, out_path)
        return

    print("[fetch] No source specified. Provide --url or --kaggle. For synthetic data, run generate_indian_synthetic.py.")

if __name__ == "__main__":
    main()
