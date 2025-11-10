from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
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


def map_indian_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map the Indian Kaggle dataset columns to the expected schema if detected.

    The Indian dataset uses columns like:
      Age, Gender, Diabetes, Hypertension, Cholesterol_Level, Systolic_BP, Diastolic_BP, Heart_Attack_Risk
    We'll derive/approximate the 13 features:
      age -> Age
      sex -> 1 if Male else 0
      cp -> proxy from Chest pain not available: use Stress_Level bucket (0-10) mapped to 0-3
      trtbps -> Systolic_BP
      chol -> Cholesterol_Level
      fbs -> Diabetes (already 0/1)
      restecg -> Family_History (treat as proxy categorical 0/1 -> 0/1/2 using simple expansion)
      thalachh -> derive as (220 - Age) - (Stress_Level*2) clipped
      exng -> Physical_Activity inverse: if Physical_Activity score <5 -> 1 else 0
      oldpeak -> Stress_Level / 3.0 (approx scaling)
      slp -> map Alcohol_Consumption (0/1) and Smoking to slope categories
      caa -> Hypertension (0/1) + Obesity (0/1) combined (0-2)
      thall -> HDL level bucketed (<=40:3, 41-55:2, >55:1)
    This mapping is heuristic for demo; not clinically validated.
    """
    cols = {c.lower(): c for c in df.columns}
    if 'heart_attack_risk' not in cols and 'Heart_Attack_Risk' not in df.columns:
        return df  # Not the expected Indian dataset; skip mapping

    def get(col):
        return df[cols[col.lower()]] if col.lower() in cols else pd.Series([pd.NA]*len(df))

    age = get('Age')
    gender = get('Gender')
    diabetes = get('Diabetes')
    hypertension = get('Hypertension')
    obesity = get('Obesity')
    smoking = get('Smoking')
    alcohol = get('Alcohol_Consumption')
    physical = get('Physical_Activity')
    stress = get('Stress_Level')
    chol_level = get('Cholesterol_Level')
    hdl = get('HDL_Level')
    systolic = get('Systolic_BP')
    family_hist = get('Family_History')
    heart_risk = get('Heart_Attack_Risk')

    # Derived features
    sex = (gender.str.lower() == 'male').astype(int)
    cp = pd.cut(stress.fillna(0), bins=[-1,3,6,8,11], labels=[0,1,2,3]).astype(int)
    trtbps = systolic.fillna(systolic.median())
    chol = chol_level.fillna(chol_level.median())
    fbs = diabetes.fillna(0).astype(int)
    # restecg proxy: 0 none family history, 1 family history, 2 family history + hypertension
    restecg = (family_hist.fillna(0).astype(int) + hypertension.fillna(0).astype(int)).clip(0,2)
    thalachh = (220 - age.fillna(age.median()) - stress.fillna(0)*2).clip(90, 200)
    exng = (physical.fillna(5) < 5).astype(int)
    oldpeak = (stress.fillna(0) / 3.0).round(1)
    slp = (alcohol.fillna(0)*2 + smoking.fillna(0)).clip(0,2)
    caa = (hypertension.fillna(0) + obesity.fillna(0)).clip(0,3)
    thall = pd.cut(hdl.fillna(hdl.median()), bins=[-1,40,55,500], labels=[3,2,1]).astype(int)

    mapped = pd.DataFrame({
        'age': age,
        'sex': sex,
        'cp': cp,
        'trtbps': trtbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalachh': thalachh,
        'exng': exng,
        'oldpeak': oldpeak,
        'slp': slp,
        'caa': caa,
        'thall': thall,
        'target': heart_risk.fillna(0).astype(int)
    })
    return mapped

def prepare_dataframe(df: pd.DataFrame, target_col: str = TARGET) -> Tuple[pd.DataFrame, pd.Series]:
    # Attempt Indian dataset mapping if signature matches
    if 'Heart_Attack_Risk' in df.columns or 'heart_attack_risk' in [c.lower() for c in df.columns]:
        df = map_indian_columns(df)
    # After mapping we expect canonical columns present
    missing = [c for c in FEATURES + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after mapping: {missing}")
    X = df[FEATURES].copy()
    y = df[target_col].astype(int)
    return X, y


def train_on_csv(csv_path: str, target_col: str = TARGET) -> Tuple[
    Dict[str, Optional[float]],
    List[str],
    str,
    Dict[str, int],
    List[List[int]],
]:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(p)

    # Attempt heuristic mapping for Indian Kaggle dataset BEFORE coercing expected numeric columns
    if 'Heart_Attack_Risk' in df.columns or 'heart_attack_risk' in [c.lower() for c in df.columns]:
        df = map_indian_columns(df)
    else:
        # Standard dataset path: enforce numeric types only if columns are already named
        for col in ["sex", "cp", "fbs", "restecg", "exng", "slp", "caa", "thall"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    X, y = prepare_dataframe(df, target_col)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
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
    roc_val: Optional[float]
    try:
        roc_val = float(roc_auc_score(y_test, proba[:, 1]))
    except ValueError:
        # roc_auc undefined when only one class present in y_test
        roc_val = None

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "roc_auc": roc_val,
    }

    class_distribution = {str(int(cls)): int(count) for cls, count in y.value_counts().items()}
    cm = confusion_matrix(y_test, preds, labels=[0, 1]).tolist()

    model_version = "v" + pd.Timestamp.utcnow().strftime("%Y%m%d%H%M%S")
    return metrics, FEATURES, model_version, class_distribution, cm
