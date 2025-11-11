"""
Pydantic Schemas for Z-Alizadeh Sani Real Medical Dataset API
==============================================================

Input schema for the model trained on real hospital coronary angiography data.
56 clinical features from Z-Alizadeh Sani UCI dataset.
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal


class ZAlizadehInput(BaseModel):
    """
    Input schema for Z-Alizadeh Sani heart attack prediction
    
    56 features from real hospital data:
    - Demographics: Age, Sex, BMI, Weight, Length
    - Risk Factors: DM, HTN, Smoking, Family History, etc.
    - ECG: Q Wave, ST changes, T inversion, LVH, etc.
    - Echo: EF-TTE, RWMA, VHD
    - Symptoms: Chest pain types, Dyspnea
    - Labs: FBS, Lipids, CBC, Chemistry
    - Physical Exam: BP, PR, Edema, Murmurs, etc.
    """
    
    # Demographics
    Age: int = Field(..., ge=18, le=120, description="Age in years")
    Sex: Literal["Male", "Fmale"] = Field(..., description="Sex (Male or Fmale)")
    Weight: float = Field(..., ge=30, le=200, description="Weight in kg")
    Length: float = Field(..., ge=100, le=250, description="Height in cm")
    BMI: float = Field(..., ge=10, le=60, description="Body Mass Index")
    
    # Risk Factors
    DM: int = Field(..., ge=0, le=1, description="Diabetes Mellitus (0=No, 1=Yes)")
    HTN: int = Field(..., ge=0, le=1, description="Hypertension (0=No, 1=Yes)")
    Current_Smoker: int = Field(..., alias="Current Smoker", ge=0, le=1, description="Current smoker (0=No, 1=Yes)")
    EX_Smoker: int = Field(..., alias="EX-Smoker", ge=0, le=1, description="Ex-smoker (0=No, 1=Yes)")
    FH: Literal["N", "Y"] = Field(..., description="Family History (N=No, Y=Yes)")
    Obesity: Literal["N", "Y"] = Field(..., description="Obesity (N=No, Y=Yes)")
    
    # Medical History
    CRF: Literal["N", "Y"] = Field(..., description="Chronic Renal Failure")
    CVA: Literal["N", "Y"] = Field(..., description="Cerebrovascular Accident")
    Airway_disease: Literal["N", "Y"] = Field(..., alias="Airway disease", description="Airway disease")
    Thyroid_Disease: Literal["N", "Y"] = Field(..., alias="Thyroid Disease", description="Thyroid Disease")
    CHF: Literal["N", "Y"] = Field(..., description="Congestive Heart Failure")
    DLP: Literal["N", "Y"] = Field(..., description="Dyslipidemia")
    
    # Vital Signs & Physical Exam
    BP: int = Field(..., ge=60, le=250, description="Blood Pressure (systolic)")
    PR: int = Field(..., ge=40, le=200, description="Pulse Rate (bpm)")
    Edema: int = Field(..., ge=0, le=1, description="Edema (0=No, 1=Yes)")
    Weak_Peripheral_Pulse: Literal["N", "Y"] = Field(..., alias="Weak Peripheral Pulse", description="Weak Peripheral Pulse")
    Lung_rales: Literal["N", "Y"] = Field(..., alias="Lung rales", description="Lung rales")
    Systolic_Murmur: Literal["N", "Y"] = Field(..., alias="Systolic Murmur", description="Systolic Murmur")
    Diastolic_Murmur: Literal["N", "Y"] = Field(..., alias="Diastolic Murmur", description="Diastolic Murmur")
    
    # Symptoms
    Typical_Chest_Pain: int = Field(..., alias="Typical Chest Pain", ge=0, le=1, description="Typical Chest Pain")
    Dyspnea: Literal["N", "Y"] = Field(..., description="Dyspnea (shortness of breath)")
    Function_Class: int = Field(..., alias="Function Class", ge=0, le=4, description="Functional Class (0-4)")
    Atypical: Literal["N", "Y"] = Field(..., description="Atypical symptoms")
    Nonanginal: Literal["N", "Y"] = Field(..., description="Nonanginal")
    Exertional_CP: Literal["N", "Y"] = Field(..., alias="Exertional CP", description="Exertional Chest Pain")
    LowTH_Ang: Literal["N", "Y"] = Field(..., alias="LowTH Ang", description="Low Threshold Angina")
    
    # ECG Findings
    Q_Wave: int = Field(..., alias="Q Wave", ge=0, le=1, description="Q Wave present")
    St_Elevation: int = Field(..., alias="St Elevation", ge=0, le=1, description="ST Elevation")
    St_Depression: int = Field(..., alias="St Depression", ge=0, le=1, description="ST Depression")
    Tinversion: int = Field(..., ge=0, le=1, description="T wave inversion")
    LVH: Literal["N", "Y"] = Field(..., description="Left Ventricular Hypertrophy")
    Poor_R_Progression: Literal["N", "Y"] = Field(..., alias="Poor R Progression", description="Poor R Progression")
    BBB: Literal["N", "LBBB", "RBBB"] = Field(..., description="Bundle Branch Block (N=None, LBBB=Left, RBBB=Right)")
    
    # Laboratory Tests
    FBS: float = Field(..., ge=50, le=500, description="Fasting Blood Sugar (mg/dL)")
    CR: float = Field(..., ge=0.3, le=15, description="Creatinine (mg/dL)")
    TG: float = Field(..., ge=30, le=1000, description="Triglycerides (mg/dL)")
    LDL: float = Field(..., ge=20, le=400, description="LDL Cholesterol (mg/dL)")
    HDL: float = Field(..., ge=10, le=150, description="HDL Cholesterol (mg/dL)")
    BUN: float = Field(..., ge=5, le=200, description="Blood Urea Nitrogen (mg/dL)")
    ESR: float = Field(..., ge=0, le=200, description="Erythrocyte Sedimentation Rate (mm/hr)")
    HB: float = Field(..., ge=5, le=20, description="Hemoglobin (g/dL)")
    K: float = Field(..., ge=2, le=8, description="Potassium (mEq/L)")
    Na: float = Field(..., ge=120, le=160, description="Sodium (mEq/L)")
    WBC: float = Field(..., ge=1000, le=50000, description="White Blood Cell count (cells/μL)")
    Lymph: float = Field(..., ge=0, le=100, description="Lymphocyte percentage")
    Neut: float = Field(..., ge=0, le=100, description="Neutrophil percentage")
    PLT: float = Field(..., ge=50, le=1000, description="Platelet count (×10³/μL)")
    
    # Echocardiography
    EF_TTE: float = Field(..., alias="EF-TTE", ge=10, le=80, description="Ejection Fraction by TTE (%)")
    Region_RWMA: int = Field(..., alias="Region RWMA", ge=0, le=5, description="Regional Wall Motion Abnormality (0-5 regions)")
    VHD: Literal["N", "mild", "Moderate", "Severe"] = Field(..., description="Valvular Heart Disease severity")
    
    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "Age": 60,
                "Sex": "Male",
                "Weight": 75,
                "Length": 170,
                "BMI": 25.95,
                "DM": 1,
                "HTN": 1,
                "Current Smoker": 0,
                "EX-Smoker": 1,
                "FH": "Y",
                "Obesity": "N",
                "CRF": "N",
                "CVA": "N",
                "Airway disease": "N",
                "Thyroid Disease": "N",
                "CHF": "N",
                "DLP": "Y",
                "BP": 140,
                "PR": 80,
                "Edema": 0,
                "Weak Peripheral Pulse": "N",
                "Lung rales": "N",
                "Systolic Murmur": "N",
                "Diastolic Murmur": "N",
                "Typical Chest Pain": 1,
                "Dyspnea": "Y",
                "Function Class": 2,
                "Atypical": "N",
                "Nonanginal": "N",
                "Exertional CP": "N",
                "LowTH Ang": "N",
                "Q Wave": 0,
                "St Elevation": 0,
                "St Depression": 1,
                "Tinversion": 1,
                "LVH": "N",
                "Poor R Progression": "N",
                "BBB": "N",
                "FBS": 150,
                "CR": 1.1,
                "TG": 180,
                "LDL": 130,
                "HDL": 40,
                "BUN": 18,
                "ESR": 25,
                "HB": 14.5,
                "K": 4.2,
                "Na": 140,
                "WBC": 8000,
                "Lymph": 30,
                "Neut": 65,
                "PLT": 250,
                "EF-TTE": 45,
                "Region RWMA": 2,
                "VHD": "mild"
            }
        }


class ZAlizadehPrediction(BaseModel):
    """Prediction response for Z-Alizadeh model"""
    prediction: int = Field(..., description="Prediction: 0 (Normal) or 1 (CAD)")
    probability: float = Field(..., ge=0, le=1, description="Probability of CAD (0-1)")
    risk_level: str = Field(..., description="Risk category: LOW RISK, MODERATE RISK, or HIGH RISK")
    risk_percentage: float = Field(..., ge=0, le=100, description="Risk percentage (0-100)")
    model_info: dict = Field(..., description="Model metadata and performance")
    
    model_config = {
        "protected_namespaces": (),  # Allow model_info field
        "json_schema_extra": {
            "example": {
                "prediction": 1,
                "probability": 0.85,
                "risk_level": "HIGH RISK",
                "risk_percentage": 85.0,
                "model_info": {
                    "model_name": "Stacking",
                    "dataset": "Z-Alizadeh Sani (UCI)",
                    "n_features": 40,
                    "accuracy": 0.8689,
                    "roc_auc": 0.9238,
                    "f1_score": 0.9111
                }
            }
        }
    }
