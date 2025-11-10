from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from backend.config import MODEL_DIR, MODEL_PATH, SCALER_PATH

FEATURES = [
    "age", "sex", "cp", "trtbps", "chol", "fbs", "restecg",
    "thalachh", "exng", "oldpeak", "slp", "caa", "thall"
]

TARGET = "target"


def ensure_model_dir():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def prepare_dataframe(df: pd.DataFrame, target_col: str = TARGET) -> Tuple[pd.DataFrame, pd.Series]:
    missing = [c for c in FEATURES + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    X = df[FEATURES].copy()
    y = df[target_col].astype(int)
    return X, y


def train_on_csv(csv_path: str, target_col: str = TARGET) -> Tuple[Dict[str, float], List[str], str]:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(p)

    # Map common Indian dataset categorical labels to numeric if necessary
    # Expecting that upstream UI or ETL maps them; here we enforce types
    for col in ["sex", "cp", "fbs", "restecg", "exng", "slp", "caa", "thall"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    X, y = prepare_dataframe(df, target_col)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    # Extract scaler and model to save separately for compatibility with app
    scaler: StandardScaler = pipeline.named_steps["scaler"]
    model: LogisticRegression = pipeline.named_steps["clf"]

    ensure_model_dir()
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    # Evaluate
    proba = model.predict_proba(scaler.transform(X_test))
    preds = model.predict(scaler.transform(X_test))

    # Note: If label orientation differs (0=high risk), ROC may need flipping externally
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, proba[:, 1]))
    }

    model_version = "v" + pd.Timestamp.utcnow().strftime("%Y%m%d%H%M%S")
    return metrics, FEATURES, model_version
