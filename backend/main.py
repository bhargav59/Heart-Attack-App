from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List

from .config import CORS_ORIGINS
from .database import Base, engine, get_db
from .models import PredictionLog
from .schemas import PredictRequest, PredictResponse, PredictResponseItem, TrainRequest, TrainResponse
from .ml_service import MLService, MODEL_VERSION, ModelNotFound

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

# Initialize ML service once
_ml = None

def get_ml() -> MLService:
    global _ml
    if _ml is None:
        _ml = MLService()
    return _ml

@app.get("/")
async def root():
    return {
        "message": "Heart Attack Risk Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "train": "/train (POST)",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "model_version": MODEL_VERSION
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
        # IMPORTANT: model trained with inverted labels: class 0 = HIGH RISK
        prob_high = float(probs[i][0])
        prob_low = float(probs[i][1])
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
