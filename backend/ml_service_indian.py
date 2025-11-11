"""
ML Service for Indian Heart Attack Dataset (34 features)
"""
from pathlib import Path
from typing import List, Tuple, Dict
import joblib
import numpy as np
import pandas as pd

from .config import MODEL_PATH, SCALER_PATH, ROOT_MODEL_PATH, ROOT_SCALER_PATH, MODEL_DIR, ROOT_DIR

MODEL_VERSION = "v3_indian_34features"

class ModelNotFound(Exception):
    pass

class MLServiceIndian:
    def __init__(self):
        self.model, self.scaler, self.feature_names = self._load_artifacts()

    def _load_artifacts(self):
        model_path = MODEL_PATH if MODEL_PATH.exists() else ROOT_MODEL_PATH
        scaler_path = SCALER_PATH if SCALER_PATH.exists() else ROOT_SCALER_PATH
        feature_names_path = MODEL_DIR / "feature_names.pkl" if MODEL_DIR.exists() else ROOT_DIR / "feature_names.pkl"
        
        if not model_path.exists() or not scaler_path.exists():
            raise ModelNotFound("Model or scaler file not found. Place 'heart_attack_model.pkl' and 'scaler.pkl' under models/ or repo root.")
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Load feature names if available
        feature_names = None
        if feature_names_path.exists():
            feature_names = joblib.load(feature_names_path)
        
        return model, scaler, feature_names

    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced features from base Indian features.
        Matches the features created in train_best_indian_model.py
        """
        df = df.copy()
        
        # Cardiovascular Risk Score (weighted composite)
        df['cv_risk_score'] = (
            df['Age'] * 0.15 +
            df['Diabetes'] * 20 +
            df['Hypertension'] * 25 +
            df['Obesity'] * 15 +
            df['Smoking'] * 30 +
            df['Family_History'] * 20
        )
        
        # Metabolic Syndrome (count of risk factors)
        df['metabolic_syndrome'] = (
            (df['Cholesterol_Level'] > 240).astype(int) +
            (df['Triglyceride_Level'] > 200).astype(int) +
            (df['HDL_Level'] < 40).astype(int) +
            (df['Systolic_BP'] > 140).astype(int) +
            df['Obesity']
        )
        
        # Lifestyle Risk (sum of behavioral factors)
        df['lifestyle_risk'] = (
            df['Smoking'] +
            df['Alcohol_Consumption'] +
            (1 - df['Physical_Activity']) +  # Sedentary lifestyle
            (10 - df['Diet_Score']) / 2  # Poor diet
        )
        
        # Blood Pressure Categories
        df['bp_category'] = pd.cut(
            df['Systolic_BP'],
            bins=[0, 120, 140, 160, 300],
            labels=[0, 1, 2, 3]
        ).astype(int)
        
        df['bp_diastolic_cat'] = pd.cut(
            df['Diastolic_BP'],
            bins=[0, 80, 90, 100, 200],
            labels=[0, 1, 2, 3]
        ).astype(int)
        
        # Age Risk Groups
        df['age_risk'] = pd.cut(
            df['Age'],
            bins=[0, 40, 55, 65, 120],
            labels=[0, 1, 2, 3]
        ).astype(int)
        
        # Cholesterol Ratios
        df['total_hdl_ratio'] = df['Cholesterol_Level'] / (df['HDL_Level'] + 1)
        df['ldl_hdl_ratio'] = df['LDL_Level'] / (df['HDL_Level'] + 1)
        
        # Important Interactions
        df['age_x_smoking'] = df['Age'] * df['Smoking']
        df['age_x_bp'] = df['Age'] * (df['Systolic_BP'] / 100)
        df['obesity_x_diabetes'] = df['Obesity'] * df['Diabetes']
        
        return df

    def predict(self, patient_data: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict heart attack risk from Indian dataset features.
        
        Args:
            patient_data: Dictionary with 23 base Indian features
            
        Returns:
            predictions: Binary predictions (0=Low Risk, 1=High Risk)
            probabilities: Probability matrix [prob_low, prob_high]
        """
        # Convert to DataFrame with base features
        base_features = [
            'Age', 'Gender', 'Diabetes', 'Hypertension', 'Obesity',
            'Smoking', 'Alcohol_Consumption', 'Physical_Activity', 'Diet_Score',
            'Cholesterol_Level', 'Triglyceride_Level', 'LDL_Level', 'HDL_Level',
            'Systolic_BP', 'Diastolic_BP', 'Air_Pollution_Exposure',
            'Family_History', 'Stress_Level', 'Healthcare_Access',
            'Heart_Attack_History', 'Emergency_Response_Time', 'Annual_Income',
            'Health_Insurance'
        ]
        
        X_df = pd.DataFrame([patient_data], columns=base_features)
        
        # Create advanced features
        X_df = self.create_advanced_features(X_df)
        
        # Ensure feature order matches training
        if self.feature_names:
            X_df = X_df[self.feature_names]
        
        # Scale and predict
        X_scaled = self.scaler.transform(X_df)
        probs = self.model.predict_proba(X_scaled)
        preds = self.model.predict(X_scaled)
        
        return preds, probs

    @staticmethod
    def to_risk(prob_high: float) -> Tuple[str, float]:
        """Convert probability to risk level and percentage"""
        risk_percent = float(prob_high * 100.0)
        if risk_percent >= 70:
            return "HIGH RISK", risk_percent
        if risk_percent >= 40:
            return "MODERATE RISK", risk_percent
        return "LOW RISK", risk_percent
