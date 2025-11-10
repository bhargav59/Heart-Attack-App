from pathlib import Path
from typing import List, Tuple
import joblib
import numpy as np
import pandas as pd

from .config import MODEL_PATH, SCALER_PATH, ROOT_MODEL_PATH, ROOT_SCALER_PATH

MODEL_VERSION = "v1"

class ModelNotFound(Exception):
    pass

class MLService:
    def __init__(self):
        self.model, self.scaler = self._load_artifacts()

    def _load_artifacts(self):
        model_path = MODEL_PATH if MODEL_PATH.exists() else ROOT_MODEL_PATH
        scaler_path = SCALER_PATH if SCALER_PATH.exists() else ROOT_SCALER_PATH
        if not model_path.exists() or not scaler_path.exists():
            raise ModelNotFound("Model or scaler file not found. Place 'heart_attack_model.pkl' and 'scaler.pkl' under models/ or repo root.")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler

    def predict(self, X: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
        X_df = pd.DataFrame(X, columns=[
            "age", "sex", "cp", "trtbps", "chol", "fbs", "restecg",
            "thalachh", "exng", "oldpeak", "slp", "caa", "thall"
        ])
        X_scaled = self.scaler.transform(X_df)
        probs = self.model.predict_proba(X_scaled)
        preds = self.model.predict(X_scaled)
        return preds, probs

    @staticmethod
    def to_risk(prob_high: float) -> Tuple[str, float]:
        risk_percent = float(prob_high * 100.0)
        if risk_percent >= 70:
            return "HIGH RISK", risk_percent
        if risk_percent >= 40:
            return "MODERATE RISK", risk_percent
        return "LOW RISK", risk_percent
