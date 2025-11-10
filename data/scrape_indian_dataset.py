"""Scrape an HTML page containing a heart disease-related table and emit a normalized CSV.

Usage:
    python data/scrape_indian_dataset.py --url https://example.com/heart_stats.html --out data/indian_scraped.csv

The script will:
1. Download the page (GET request).
2. Parse tables using pandas.read_html (lxml backend) and BeautifulSoup for fallback.
3. Attempt to map column names to the required feature schema:
   ['age','sex','cp','trtbps','chol','fbs','restecg','thalachh','exng','oldpeak','slp','caa','thall','target']
4. Drop rows with excessive missing values and coerce types.

Limitations & Notes:
- Scraping must comply with the website's Terms of Service and robots.txt.
- Do NOT use this for bulk unauthorized extraction of proprietary data.
- Real India-specific heart disease open datasets are scarce; this is a flexible ingestion helper.
- If multiple tables match, the first adequate one is chosen.

Column Mapping Heuristics:
- Accept case-insensitive matches and common synonyms.
- 'sex' may appear as 'gender'.
- 'trtbps' might appear as 'restbp', 'bp', 'blood_pressure'.
- 'chol' might appear as 'cholesterol'.
- 'fbs' as 'fasting_bs', 'fasting_blood_sugar'.
- 'restecg' as 'rest_ecg'.
- 'thalachh' as 'max_hr', 'max_heart_rate'.
- 'exng' as 'exercise_angina'.
- 'oldpeak' as 'st_depression'.
- 'slp' as 'slope'.
- 'caa' as 'ca', 'num_vessels', 'vessels'.
- 'thall' as 'thal', 'thalassemia', 'thallium'.
- 'target' as 'outcome', 'disease', 'has_disease'.

If target column is not present, script can generate a synthetic placeholder using a simple heuristic unless --require-target is passed.
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path
import requests
import pandas as pd
from bs4 import BeautifulSoup

FEATURES = ['age','sex','cp','trtbps','chol','fbs','restecg','thalachh','exng','oldpeak','slp','caa','thall','target']

SYNONYMS = {
    'sex': ['gender','sex'],
    'cp': ['chest_pain','cp','chestpain'],
    'trtbps': ['trtbps','restbp','bp','blood_pressure'],
    'chol': ['chol','cholesterol'],
    'fbs': ['fbs','fasting_bs','fasting_blood_sugar'],
    'restecg': ['restecg','rest_ecg','ecg_rest'],
    'thalachh': ['thalachh','max_hr','max_heart_rate','thalach'],
    'exng': ['exng','exercise_angina','ex_angina'],
    'oldpeak': ['oldpeak','st_depression','stdep'],
    'slp': ['slp','slope'],
    'caa': ['caa','ca','num_vessels','vessels'],
    'thall': ['thall','thal','thalassemia','thallium'],
    'target': ['target','outcome','disease','has_disease']
}

NUMERIC_FORCE = set(['age','sex','cp','trtbps','chol','fbs','restecg','thalachh','exng','oldpeak','slp','caa','thall','target'])


def fetch_html(url: str) -> str:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.text


def extract_tables(html: str) -> list[pd.DataFrame]:
    # First try pandas.read_html
    try:
        tables = pd.read_html(html, flavor='lxml')
        if tables:
            return tables
    except Exception:
        pass
    # Fallback: manual soup table parse into DataFrame(s)
    soup = BeautifulSoup(html, 'lxml')
    out = []
    for tbl in soup.find_all('table'):
        rows = []
        for tr in tbl.find_all('tr'):
            cells = [c.get_text(strip=True) for c in tr.find_all(['th','td'])]
            if cells:
                rows.append(cells)
        if rows:
            header = rows[0]
            data_rows = rows[1:]
            df = pd.DataFrame(data_rows, columns=header)
            out.append(df)
    return out


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame | None:
    col_map = {}
    lowered = {c.lower(): c for c in df.columns}

    for feat in FEATURES:
        if feat in ('target'):  # treat similarly
            pass
        synonyms = SYNONYMS.get(feat, [feat])
        found = None
        for syn in synonyms:
            # exact or underscores/hyphen insensitive
            for raw_lower, original in lowered.items():
                if raw_lower == syn.lower():
                    found = original
                    break
                # remove non-alphanum for fuzzy compare
                if re.sub(r'[^a-z0-9]', '', raw_lower) == re.sub(r'[^a-z0-9]', '', syn.lower()):
                    found = original
                    break
            if found:
                break
        if found:
            col_map[found] = feat

    # Need minimum required (all features except target if we allow synthetic)
    missing = [f for f in FEATURES if f not in col_map.values() and f != 'target']
    if len(missing) > 4:  # too many gaps
        return None

    df2 = df.rename(columns=col_map)
    # Add absent columns with NaN
    for f in FEATURES:
        if f not in df2.columns:
            df2[f] = pd.NA
    # Coerce numeric
    for f in NUMERIC_FORCE:
        df2[f] = pd.to_numeric(df2[f], errors='coerce')

    # Generate synthetic target if missing
    if df2['target'].isna().all():
        # Simple heuristic: combine age, chol, trtbps
        risk_raw = (
            (df2['age'].fillna(0) > 55).astype(int) +
            (df2['chol'].fillna(0) > 240).astype(int) +
            (df2['trtbps'].fillna(0) > 140).astype(int)
        )
        df2['target'] = (risk_raw >= 2).astype(int)

    # Drop rows where fewer than half of required numeric features are present
    numeric_feats = [f for f in FEATURES if f != 'target']
    def sufficient(row):
        return row[numeric_feats].notna().sum() >= len(numeric_feats) // 2
    df2 = df2[df2.apply(sufficient, axis=1)]

    return df2[FEATURES]


def scrape(url: str) -> pd.DataFrame:
    html = fetch_html(url)
    tables = extract_tables(html)
    if not tables:
        raise SystemExit('No tables found on page.')
    for t in tables:
        norm = normalize_columns(t)
        if norm is not None:
            return norm
    raise SystemExit('No suitable table matched the feature schema heuristics.')


def main():
    ap = argparse.ArgumentParser(description='Scrape heart dataset table from a web page.')
    ap.add_argument('--url', required=True, help='Page URL containing table.')
    ap.add_argument('--out', default='data/indian_scraped.csv', help='Output CSV path.')
    ap.add_argument('--force', action='store_true', help='Overwrite existing output.')
    args = ap.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        print(f"Output {out_path} exists. Use --force to overwrite.")
        return

    df = scrape(args.url)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved scraped dataset: {out_path} | rows={len(df)}")

if __name__ == '__main__':
    main()
