from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, JSON
from .database import Base

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    inputs = Column(JSON, nullable=False)
    risk_percent = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False)
    model_version = Column(String(50), default="v1")
    client = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
