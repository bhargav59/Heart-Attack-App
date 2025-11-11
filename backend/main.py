from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List

from .config import CORS_ORIGINS
from .database import Base, engine, get_db
from .models import PredictionLog
from .schemas import PredictRequest, PredictResponse, PredictResponseItem, TrainRequest, TrainResponse
from .schemas_indian import IndianHeartInput, PredictRequest as IndianPredictRequest, PredictResponse as IndianPredictResponse
from .ml_service import MLService, MODEL_VERSION, ModelNotFound
from .ml_service_indian import MLServiceIndian, MODEL_VERSION as INDIAN_MODEL_VERSION

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

@app.get("/")
async def root():
    return {
        "message": "Heart Attack Risk Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST) - Standard 13 features",
            "predict_indian": "/predict_indian (POST) - Indian 23 features",
            "train": "/train (POST)",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "model_versions": {
            "standard": MODEL_VERSION,
            "indian": INDIAN_MODEL_VERSION
        }
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
