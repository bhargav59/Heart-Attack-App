from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List
from pathlib import Path

from .config import CORS_ORIGINS
from .database import Base, engine, get_db
from .models import PredictionLog
from .schemas import PredictRequest, PredictResponse, PredictResponseItem, TrainRequest, TrainResponse
from .schemas_indian import IndianHeartInput, PredictRequest as IndianPredictRequest, PredictResponse as IndianPredictResponse
from .schemas_z_alizadeh import ZAlizadehInput, ZAlizadehPrediction
from .ml_service import MLService, MODEL_VERSION, ModelNotFound
from .ml_service_indian import MLServiceIndian, MODEL_VERSION as INDIAN_MODEL_VERSION
from .ml_service_z_alizadeh import get_z_alizadeh_service, ZAlizadehMLService

app = FastAPI(title="Heart Attack Risk API", version="1.0.0")

# CORS
origins = [o.strip() for o in CORS_ORIGINS.split(",")] if CORS_ORIGINS else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create DB tables
Base.metadata.create_all(bind=engine)

# Initialize ML services once
_ml = None
_ml_indian = None

def get_ml() -> MLService:
    global _ml
    if _ml is None:
        _ml = MLService()
    return _ml

def get_ml_indian() -> MLServiceIndian:
    global _ml_indian
    if _ml_indian is None:
        _ml_indian = MLServiceIndian()
    return _ml_indian

# Serve the HTML frontend
@app.get("/app")
async def serve_frontend():
    """Serve the HTML frontend at /app"""
    frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
    return FileResponse(frontend_path)

@app.get("/")
async def root():
    return {
        "message": "Heart Attack Risk Prediction API",
        "version": "1.0.0",
        "frontend": "http://localhost:8000/app",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST) - Standard 13 features (UCI Heart Disease)",
            "predict_indian": "/predict_indian (POST) - Indian 23 features (synthetic data - deprecated)",
            "predict_real": "/predict_real (POST) - Z-Alizadeh Sani 56 features (REAL medical data - RECOMMENDED)",
            "train": "/train (POST)",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "model_versions": {
            "standard": MODEL_VERSION,
            "indian": INDIAN_MODEL_VERSION,
            "real": "v1.0_z_alizadeh_sani"
        },
        "recommended_endpoint": "/predict_real (86.89% accuracy, 92.38% ROC AUC on real hospital data)"
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest, db: Session = Depends(get_db)):
    try:
        ml = get_ml()
    except ModelNotFound as e:
        raise HTTPException(status_code=500, detail=str(e))

    X = [item.to_feature_list() for item in req.data]
    preds, probs = ml.predict(X)

    results: List[PredictResponseItem] = []
    for i in range(len(X)):
        # Advanced model (Indian dataset): class 0 = LOW RISK, class 1 = HIGH RISK
        prob_low = float(probs[i][0])
        prob_high = float(probs[i][1])
        risk_level, risk_percent = ml.to_risk(prob_high)

        # Persist log
        log = PredictionLog(
            inputs={
                "features": X[i]
            },
            risk_percent=risk_percent,
            risk_level=risk_level,
            model_version=MODEL_VERSION,
            client=req.client,
        )
        db.add(log)
        db.commit()

        results.append(PredictResponseItem(
            risk_percent=risk_percent,
            risk_level=risk_level,
            probabilities={"high": prob_high, "low": prob_low},
        ))

    return PredictResponse(results=results, model_version=MODEL_VERSION)

@app.post("/predict_indian", response_model=IndianPredictResponse)
async def predict_indian(req: IndianPredictRequest, db: Session = Depends(get_db)):
    """
    Predict heart attack risk using Indian dataset with 23 native features + 11 engineered = 34 total.
    Achieves 69.05% accuracy with calibrated Gradient Boosting ensemble.
    
    NOTE: Dataset appears to be synthetic (max feature correlation < 0.03).
    Real medical data would show stronger predictive relationships.
    """
    try:
        ml = get_ml_indian()
    except ModelNotFound as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Convert input to feature dictionary
    patient_features = req.patient_data.to_feature_dict()
    
    # Get prediction
    preds, probs = ml.predict(patient_features)
    
    # Extract probabilities (class 0 = LOW RISK, class 1 = HIGH RISK)
    prob_low = float(probs[0][0])
    prob_high = float(probs[0][1])
    risk_level, risk_percent = ml.to_risk(prob_high)
    
    # Log prediction
    log = PredictionLog(
        inputs=patient_features,
        risk_percent=risk_percent,
        risk_level=risk_level,
        model_version=INDIAN_MODEL_VERSION,
        client="indian_api",
    )
    db.add(log)
    db.commit()
    
    return IndianPredictResponse(
        risk_percent=risk_percent,
        risk_level=risk_level,
        probabilities={"high": prob_high, "low": prob_low},
        model_version=INDIAN_MODEL_VERSION
    )

@app.post("/predict_real", response_model=ZAlizadehPrediction)
async def predict_real(patient_data: ZAlizadehInput, db: Session = Depends(get_db)):
    """
    Predict heart attack risk using Z-Alizadeh Sani REAL medical dataset.
    
    **RECOMMENDED ENDPOINT** - Trained on authentic hospital coronary angiography data.
    
    Dataset: Z-Alizadeh Sani et al. (2013), UCI Machine Learning Repository
    - 303 Asian patients from real hospital data
    - 56 clinical features (ECG, Echo, Labs, Symptoms, Risk Factors)
    - Peer-reviewed medical research
    
    Performance:
    - Accuracy: 86.89% (+17.84% vs synthetic)
    - ROC AUC: 92.38% (+44% vs synthetic)
    - F1 Score: 91.11% (+22% vs synthetic)
    
    Model: Stacking Ensemble (RF + ET + GB + XGBoost + LightGBM + CatBoost)
    """
    try:
        ml = get_z_alizadeh_service()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model loading error: {str(e)}")
    
    # Convert Pydantic model to dict
    patient_dict = patient_data.model_dump(by_alias=True)
    
    # Get prediction
    try:
        pred, proba = ml.predict(patient_dict)
    except Exception as e:
        import traceback
        error_details = {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "patient_data_keys": list(patient_dict.keys())[:10]
        }
        raise HTTPException(status_code=500, detail=f"Prediction error: {error_details}")
    
    # Extract probabilities (class 0 = Normal, class 1 = CAD)
    prob_cad = float(proba[1])
    risk_level, risk_percent = ml.to_risk(prob_cad)
    
    # Log prediction
    log = PredictionLog(
        inputs=patient_dict,
        risk_percent=risk_percent,
        risk_level=risk_level,
        model_version="v1.0_z_alizadeh_sani",
        client="real_api",
    )
    db.add(log)
    db.commit()
    
    return ZAlizadehPrediction(
        prediction=int(pred),
        probability=prob_cad,
        risk_level=risk_level,
        risk_percentage=risk_percent,
        model_info=ml.get_model_info()
    )

@app.post("/train", response_model=TrainResponse)
async def train(req: TrainRequest):
    # Lazy import to avoid heavy deps on boot
    import pandas as pd
    from ..ml.train import train_on_csv

    metrics, features, model_version, class_distribution, confusion = train_on_csv(
        req.dataset_path,
        target_col=req.target_column,
    )
    # refresh ML singleton
    global _ml
    _ml = None
    return TrainResponse(
        model_version=model_version,
        metrics=metrics,
        features=features,
        class_distribution=class_distribution,
        confusion_matrix=confusion,
    )
