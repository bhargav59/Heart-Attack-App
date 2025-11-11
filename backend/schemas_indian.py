"""
Pydantic schemas for Indian Heart Attack Dataset
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class IndianHeartInput(BaseModel):
    """Input schema for Indian heart attack risk prediction with all 26 features"""
    
    # Demographics
    age: int = Field(ge=20, le=100, description="Age in years")
    gender: str = Field(description="Gender: Male or Female")
    
    # Medical Conditions (Binary: 0=No, 1=Yes)
    diabetes: int = Field(ge=0, le=1, description="Has diabetes")
    hypertension: int = Field(ge=0, le=1, description="Has hypertension")
    obesity: int = Field(ge=0, le=1, description="Is obese")
    
    # Lifestyle Factors (Binary: 0=No, 1=Yes)
    smoking: int = Field(ge=0, le=1, description="Smoker")
    alcohol_consumption: int = Field(ge=0, le=1, description="Consumes alcohol")
    physical_activity: int = Field(ge=0, le=1, description="Physically active")
    diet_score: int = Field(ge=0, le=10, description="Diet quality score (0-10)")
    
    # Blood Work (Continuous values)
    cholesterol_level: int = Field(ge=100, le=400, description="Total cholesterol (mg/dL)")
    triglyceride_level: int = Field(ge=30, le=400, description="Triglycerides (mg/dL)")
    ldl_level: int = Field(ge=30, le=300, description="LDL cholesterol (mg/dL)")
    hdl_level: int = Field(ge=15, le=100, description="HDL cholesterol (mg/dL)")
    
    # Blood Pressure
    systolic_bp: int = Field(ge=80, le=200, description="Systolic BP (mmHg)")
    diastolic_bp: int = Field(ge=50, le=130, description="Diastolic BP (mmHg)")
    
    # Environmental & Social Factors
    air_pollution_exposure: int = Field(ge=0, le=1, description="Exposed to air pollution")
    family_history: int = Field(ge=0, le=1, description="Family history of heart disease")
    stress_level: int = Field(ge=1, le=10, description="Stress level (1-10)")
    
    # Healthcare Access
    healthcare_access: int = Field(ge=0, le=1, description="Has healthcare access")
    heart_attack_history: int = Field(ge=0, le=1, description="Previous heart attack")
    
    # Socioeconomic
    emergency_response_time: int = Field(ge=5, le=500, description="Emergency response time (minutes)")
    annual_income: int = Field(ge=10000, le=3000000, description="Annual income (INR)")
    health_insurance: int = Field(ge=0, le=1, description="Has health insurance")
    
    def to_feature_dict(self) -> Dict[str, float]:
        """Convert to dictionary with proper feature names"""
        return {
            'Age': self.age,
            'Gender': 1 if self.gender.lower() == 'male' else 0,  # Male=1, Female=0
            'Diabetes': self.diabetes,
            'Hypertension': self.hypertension,
            'Obesity': self.obesity,
            'Smoking': self.smoking,
            'Alcohol_Consumption': self.alcohol_consumption,
            'Physical_Activity': self.physical_activity,
            'Diet_Score': self.diet_score,
            'Cholesterol_Level': self.cholesterol_level,
            'Triglyceride_Level': self.triglyceride_level,
            'LDL_Level': self.ldl_level,
            'HDL_Level': self.hdl_level,
            'Systolic_BP': self.systolic_bp,
            'Diastolic_BP': self.diastolic_bp,
            'Air_Pollution_Exposure': self.air_pollution_exposure,
            'Family_History': self.family_history,
            'Stress_Level': self.stress_level,
            'Healthcare_Access': self.healthcare_access,
            'Heart_Attack_History': self.heart_attack_history,
            'Emergency_Response_Time': self.emergency_response_time,
            'Annual_Income': self.annual_income,
            'Health_Insurance': self.health_insurance
        }

class PredictRequest(BaseModel):
    """Request for heart attack risk prediction"""
    patient_data: IndianHeartInput

class PredictResponse(BaseModel):
    """Response with risk prediction"""
    model_config = {"protected_namespaces": ()}
    
    risk_percent: float = Field(description="Risk percentage (0-100)")
    risk_level: str = Field(description="Risk level: LOW RISK, MODERATE RISK, or HIGH RISK")
    probabilities: Dict[str, float] = Field(description="Class probabilities")
    model_version: str = Field(description="Model version used")
    
class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_type: Optional[str] = None
    features_count: Optional[int] = None

class TrainRequest(BaseModel):
    """Request to retrain model"""
    dataset_path: str = Field(default="data/_kaggle_tmp/heart_attack_prediction_india.csv")
    use_smote: bool = Field(default=True)
    target_accuracy: float = Field(default=0.85, ge=0.5, le=1.0)

class TrainResponse(BaseModel):
    """Response after training"""
    model_config = {"protected_namespaces": ()}
    
    success: bool
    model_version: str
    accuracy: float
    roc_auc: float
    message: str
