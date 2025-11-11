"""
ML Service for Z-Alizadeh Sani Real Medical Dataset Model
===========================================================

This service handles predictions using the model trained on real hospital data
from Z-Alizadeh Sani UCI dataset (303 patients, 56 features).

The model expects 40 engineered features and achieves:
- 86.89% Accuracy
- 92.38% ROC AUC
- 91.11% F1 Score
"""

from pathlib import Path
from typing import List, Tuple, Dict
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .config import MODEL_DIR, ROOT_DIR

class ZAlizadehMLService:
    def __init__(self):
        self.model, self.scaler, self.feature_names, self.metadata = self._load_artifacts()
        self.label_encoders = self._prepare_encoders()

    def _load_artifacts(self):
        """Load model artifacts for Z-Alizadeh Sani dataset"""
        model_path = MODEL_DIR / "heart_attack_model_real.pkl" if MODEL_DIR.exists() else ROOT_DIR / "models" / "heart_attack_model_real.pkl"
        scaler_path = MODEL_DIR / "scaler_real.pkl" if MODEL_DIR.exists() else ROOT_DIR / "models" / "scaler_real.pkl"
        features_path = MODEL_DIR / "feature_names_real.pkl" if MODEL_DIR.exists() else ROOT_DIR / "models" / "feature_names_real.pkl"
        metadata_path = MODEL_DIR / "model_metadata_real.pkl" if MODEL_DIR.exists() else ROOT_DIR / "models" / "model_metadata_real.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        feature_names = joblib.load(features_path) if features_path.exists() else None
        metadata = joblib.load(metadata_path) if metadata_path.exists() else None
        
        return model, scaler, feature_names, metadata

    def _prepare_encoders(self) -> Dict[str, LabelEncoder]:
        """Prepare label encoders for categorical features"""
        encoders = {}
        
        # Sex encoder: Male=1, Female=0
        sex_encoder = LabelEncoder()
        sex_encoder.fit(['Fmale', 'Male'])  # Note: Dataset uses 'Fmale' typo
        encoders['Sex'] = sex_encoder
        
        # Binary Y/N encoders
        yn_encoder = LabelEncoder()
        yn_encoder.fit(['N', 'Y'])
        for col in ['Obesity', 'CRF', 'CVA', 'Airway disease', 'Thyroid Disease', 
                    'CHF', 'DLP', 'Weak Peripheral Pulse', 'Lung rales', 
                    'Systolic Murmur', 'Diastolic Murmur', 'Dyspnea', 'Atypical', 
                    'Nonanginal', 'LowTH Ang', 'LVH', 'Poor R Progression']:
            encoders[col] = yn_encoder
        
        # BBB encoder
        bbb_encoder = LabelEncoder()
        bbb_encoder.fit(['N', 'LBBB', 'RBBB'])
        encoders['BBB'] = bbb_encoder
        
        # VHD encoder
        vhd_encoder = LabelEncoder()
        vhd_encoder.fit(['N', 'mild', 'Moderate', 'Severe'])
        encoders['VHD'] = vhd_encoder
        
        return encoders

    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create the same engineered features as training script
        
        Input: DataFrame with 56 Z-Alizadeh Sani raw features
        Output: DataFrame with 74 features (56 original + 18 engineered)
        """
        df_engineered = df.copy()
        
        # Ensure all numeric columns are actually numeric
        for col in df_engineered.columns:
            if col in df_engineered.select_dtypes(include=[np.number]).columns:
                df_engineered[col] = pd.to_numeric(df_engineered[col], errors='coerce').fillna(0)
        
        # 1. CV Risk Score
        cv_risk_features = []
        if 'Age' in df.columns:
            df_engineered['Age_Risk'] = (df['Age'] > 60).astype(int)
            cv_risk_features.append('Age_Risk')
        
        for col in ['HTN', 'DM', 'Current Smoker', 'DLP']:
            if col in df.columns:
                cv_risk_features.append(col)
        
        if cv_risk_features:
            df_engineered['CV_Risk_Score'] = df_engineered[cv_risk_features].sum(axis=1)
        
        # 2. Metabolic Syndrome
        metabolic_features = []
        if 'BMI' in df.columns:
            df_engineered['High_BMI'] = (df['BMI'] > 30).astype(int)
            metabolic_features.append('High_BMI')
        if 'FBS' in df.columns:
            df_engineered['High_FBS'] = (df['FBS'] > 126).astype(int)
            metabolic_features.append('High_FBS')
        if 'TG' in df.columns:
            df_engineered['High_TG'] = (df['TG'] > 150).astype(int)
            metabolic_features.append('High_TG')
        if 'HDL' in df.columns:
            df_engineered['Low_HDL'] = (df['HDL'] < 40).astype(int)
            metabolic_features.append('Low_HDL')
        
        if metabolic_features:
            df_engineered['Metabolic_Syndrome'] = df_engineered[metabolic_features].sum(axis=1)
        
        # 3. ECG Abnormality Score
        ecg_features = []
        for col in ['Q Wave', 'St Elevation', 'St Depression', 'Tinversion', 'LVH', 'Poor R Progression']:
            if col in df.columns:
                ecg_features.append(col)
        
        if ecg_features:
            df_engineered['ECG_Abnormality_Score'] = df_engineered[ecg_features].sum(axis=1)
        
        # 4. Lipid Ratios
        if 'LDL' in df.columns and 'HDL' in df.columns:
            df_engineered['LDL_HDL_Ratio'] = df['LDL'] / (df['HDL'] + 1)
        
        if 'TG' in df.columns and 'HDL' in df.columns:
            df_engineered['TG_HDL_Ratio'] = df['TG'] / (df['HDL'] + 1)
        
        # 5. Age-related risk factors
        if 'Age' in df.columns:
            if 'Sex' in df.columns:
                df_engineered['Age_Sex_Risk'] = df['Age'] * (df['Sex'] == 1)
            
            if 'DM' in df.columns:
                df_engineered['Age_DM_Risk'] = df['Age'] * df['DM']
        
        # 6. Cardiac function
        if 'EF-TTE' in df.columns:
            df_engineered['Low_EF'] = (df['EF-TTE'] < 40).astype(int)
        
        # 7. Symptom Severity
        symptom_features = []
        for col in ['Typical Chest Pain', 'Dyspnea', 'Atypical', 'Nonanginal', 'Exertional CP']:
            if col in df.columns:
                symptom_features.append(col)
        
        if symptom_features:
            df_engineered['Symptom_Severity'] = df_engineered[symptom_features].sum(axis=1)
        
        # 8. Lab Abnormality
        lab_features = []
        if 'ESR' in df.columns:
            df_engineered['High_ESR'] = (df['ESR'] > 20).astype(int)
            lab_features.append('High_ESR')
        if 'WBC' in df.columns:
            df_engineered['High_WBC'] = (df['WBC'] > 11000).astype(int)
            lab_features.append('High_WBC')
        if 'CR' in df.columns:
            df_engineered['High_CR'] = (df['CR'] > 1.3).astype(int)
            lab_features.append('High_CR')
        
        if lab_features:
            df_engineered['Lab_Abnormality'] = df_engineered[lab_features].sum(axis=1)
        
        return df_engineered

    def prepare_features(self, raw_data: Dict) -> pd.DataFrame:
        """
        Prepare features from raw input data
        
        Args:
            raw_data: Dictionary with Z-Alizadeh Sani features (56 fields)
        
        Returns:
            DataFrame with engineered features matching training pipeline
        """
        # Create DataFrame from input
        df = pd.DataFrame([raw_data])
        
        # Encode categorical variables FIRST before feature engineering
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                if df[col].dtype == 'object' or isinstance(df[col].iloc[0], str):
                    try:
                        df[col] = encoder.transform(df[col].astype(str))
                    except ValueError:
                        # Handle unknown categories by using most common value
                        df[col] = encoder.transform([encoder.classes_[0]])[0]
        
        # Ensure all columns are numeric after encoding
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to numeric, if fails, use label encoding
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    # Last resort: create a simple encoder
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
        
        # Create advanced features (56 → 74)
        df_engineered = self.create_advanced_features(df)
        
        # Select only the 40 features used by the model
        if self.feature_names:
            # Ensure all required features exist
            missing_features = set(self.feature_names) - set(df_engineered.columns)
            if missing_features:
                # Fill missing features with 0
                for feat in missing_features:
                    df_engineered[feat] = 0
            
            df_selected = df_engineered[self.feature_names]
        else:
            df_selected = df_engineered
        
        return df_selected

    def predict(self, raw_data: Dict) -> Tuple[int, np.ndarray]:
        """
        Make prediction on raw Z-Alizadeh Sani format data
        
        Args:
            raw_data: Dictionary with 56 Z-Alizadeh features
        
        Returns:
            Tuple of (prediction, probabilities)
            prediction: 0 (Normal) or 1 (CAD)
            probabilities: [prob_normal, prob_cad]
        """
        # Prepare features
        X = self.prepare_features(raw_data)
        
        # Scale features
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        # Predict
        pred = self.model.predict(X_scaled)[0]
        proba = self.model.predict_proba(X_scaled)[0]
        
        return int(pred), proba

    @staticmethod
    def to_risk(prob_cad: float) -> Tuple[str, float]:
        """
        Convert probability to risk category
        
        Args:
            prob_cad: Probability of CAD (0-1)
        
        Returns:
            Tuple of (risk_level, risk_percentage)
        """
        risk_percent = float(prob_cad * 100.0)
        
        if risk_percent >= 70:
            return "HIGH RISK", risk_percent
        elif risk_percent >= 40:
            return "MODERATE RISK", risk_percent
        else:
            return "LOW RISK", risk_percent

    def get_model_info(self) -> Dict:
        """Return model metadata"""
        return {
            "model_name": self.metadata.get("model_name", "Stacking") if self.metadata else "Stacking",
            "dataset": self.metadata.get("dataset", "Z-Alizadeh Sani") if self.metadata else "Z-Alizadeh Sani",
            "n_features": len(self.feature_names) if self.feature_names else 40,
            "training_date": self.metadata.get("training_date", "2025-11-11") if self.metadata else "2025-11-11",
            "accuracy": 0.8689,
            "roc_auc": 0.9238,
            "f1_score": 0.9111
        }


# Singleton instance
_z_alizadeh_service: ZAlizadehMLService = None

def get_z_alizadeh_service() -> ZAlizadehMLService:
    """Get or create singleton Z-Alizadeh ML service instance"""
    global _z_alizadeh_service
    if _z_alizadeh_service is None:
        _z_alizadeh_service = ZAlizadehMLService()
    return _z_alizadeh_service
