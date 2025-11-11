from pathlib import Path
from typing import List, Tuple, Optional, Callable
import joblib
import numpy as np
import pandas as pd

from .config import MODEL_PATH, SCALER_PATH, ROOT_MODEL_PATH, ROOT_SCALER_PATH, MODEL_DIR, ROOT_DIR

MODEL_VERSION = "v2_advanced"

class ModelNotFound(Exception):
    pass

class MLService:
    def __init__(self):
        self.model, self.scaler, self.feature_engineer = self._load_artifacts()

    def _load_artifacts(self):
        model_path = MODEL_PATH if MODEL_PATH.exists() else ROOT_MODEL_PATH
        scaler_path = SCALER_PATH if SCALER_PATH.exists() else ROOT_SCALER_PATH
        
        # Try to import feature engineering from ml module (advanced training)
        feature_engineer: Optional[Callable] = None
        feature_eng_path = MODEL_DIR / "feature_engineer.pkl" if MODEL_DIR.exists() else ROOT_DIR / "feature_engineer.pkl"
        
        # If advanced model artifacts exist, use feature engineering
        if feature_eng_path.exists() or (model_path.exists() and self._is_advanced_model(model_path)):
            try:
                from ml.feature_engineering import create_advanced_features
                feature_engineer = create_advanced_features
            except ImportError:
                feature_engineer = None
        
        if not model_path.exists() or not scaler_path.exists():
            raise ModelNotFound("Model or scaler file not found. Place 'heart_attack_model.pkl' and 'scaler.pkl' under models/ or repo root.")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler, feature_engineer
    
    def _is_advanced_model(self, model_path: Path) -> bool:
        """Check if model expects advanced features (22 vs 13)."""
        try:
            model = joblib.load(model_path)
            # LightGBM, XGBoost models have n_features_in_ attribute
            if hasattr(model, 'n_features_in_'):
                return model.n_features_in_ > 13
            return False
        except Exception:
            return False

    def predict(self, X: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
        X_df = pd.DataFrame(X, columns=[
            "age", "sex", "cp", "trtbps", "chol", "fbs", "restecg",
            "thalachh", "exng", "oldpeak", "slp", "caa", "thall"
        ])
        
        # Apply feature engineering if available
        if self.feature_engineer is not None:
            X_df = self.feature_engineer(X_df)
        
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
