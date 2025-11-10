"""Generate a synthetic Indian heart disease dataset approximating known prevalence patterns.

DISCLAIMER: Synthetic data; not real clinical records. For development & demonstration only.

Approximate prevalence adjustments (non-authoritative):
- Higher early onset: more cases in 40-55 age bracket
- Elevated diabetes (fbs=1) proportion
- Hypertension (trtbps>140) moderately common among >45
- Cholesterol distribution slightly right-skewed
- Chest pain types distributed; typical angina lower share

Feature schema:
 age,sex,cp,trtbps,chol,fbs,restecg,thalachh,exng,oldpeak,slp,caa,thall,target

Run:
    python data/generate_indian_synthetic.py --rows 1500 --out data/indian_heart_synthetic.csv
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

FEATURES = ["age","sex","cp","trtbps","chol","fbs","restecg","thalachh","exng","oldpeak","slp","caa","thall","target"]

rng = np.random.default_rng(42)

def sample_row() -> dict:
    age = rng.integers(30, 80)  # broader younger distribution
    sex = rng.choice([0,1], p=[0.35,0.65])  # male higher proportion in many datasets
    cp = rng.choice([0,1,2,3], p=[0.20,0.30,0.30,0.20])
    # Blood pressure: base + age effect
    trtbps = int(rng.normal(120 + (age-40)*0.9, 15))
    trtbps = max(80, min(trtbps, 200))
    # Cholesterol: skewed distribution
    chol = int(rng.gamma(5, 40) + 140)  # shape*scale + shift
    chol = max(120, min(chol, 520))
    fbs = rng.choice([0,1], p=[0.78,0.22])  # elevated
    restecg = rng.choice([0,1,2], p=[0.55,0.35,0.10])
    thalachh = int(rng.normal(160 - (age-30)*0.6, 12))
    thalachh = max(90, min(thalachh, 200))
    exng = rng.choice([0,1], p=[0.75,0.25])
    oldpeak = round(max(0.0, rng.normal(0.8 + (exng*0.5), 0.6)), 1)
    slp = rng.choice([0,1,2], p=[0.50,0.35,0.15])
    caa = rng.choice([0,1,2,3], p=[0.60,0.20,0.15,0.05])
    thall = rng.choice([1,2,3], p=[0.60,0.20,0.20])  # 1=Normal,2=Fixed,3=Reversible

    # Risk model heuristic for target (binary); not medically validated
    risk_score = 0
    risk_score += (age > 55) * 1.2
    risk_score += (trtbps > 140) * 1.0
    risk_score += (chol > 240) * 1.0
    risk_score += (fbs == 1) * 0.8
    risk_score += (exng == 1) * 1.1
    risk_score += (oldpeak > 1.5) * 1.3
    risk_score += (caa >= 1) * 1.4
    risk_score += (thall == 3) * 1.0

    # Convert risk_score to probability via logistic-like transform
    prob = 1/(1 + np.exp(- (risk_score - 2.0)))  # shift center
    target = int(rng.random() < prob)

    return {"age":age,"sex":sex,"cp":cp,"trtbps":trtbps,"chol":chol,"fbs":fbs,"restecg":restecg,
            "thalachh":thalachh,"exng":exng,"oldpeak":oldpeak,"slp":slp,"caa":caa,"thall":thall,"target":target}


def generate(rows: int) -> pd.DataFrame:
    data = [sample_row() for _ in range(rows)]
    return pd.DataFrame(data)[FEATURES]


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic Indian heart dataset")
    ap.add_argument("--rows", type=int, default=1000)
    ap.add_argument("--out", type=str, default="data/indian_heart_synthetic.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    global rng
    rng = np.random.default_rng(args.seed)

    df = generate(args.rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote synthetic dataset: {out_path} ({len(df)} rows)")

if __name__ == "__main__":
    main()
