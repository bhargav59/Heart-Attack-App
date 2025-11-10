import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Model paths
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "heart_attack_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

# Fallback to root-level model files if not present in models/
ROOT_MODEL_PATH = BASE_DIR / "heart_attack_model.pkl"
ROOT_SCALER_PATH = BASE_DIR / "scaler.pkl"

# Database
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{(BASE_DIR / 'app.db').as_posix()}")

# CORS origins for frontend
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
