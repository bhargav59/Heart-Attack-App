#!/usr/bin/env python3
"""
Quick demo of the finalized Heart Attack Risk Prediction API
"""
import sys
sys.path.insert(0, '/workspaces/Heart-Attack-App')

from backend.ml_service import MLService
import numpy as np

print("=" * 70)
print("Heart Attack Risk Prediction - Demo")
print("=" * 70)

# Load the model
print("\n✅ Loading model trained on 10,000 Indian patient records...")
ml = MLService()
print(f"   Model loaded successfully!")
print(f"   Model type: {type(ml.model).__name__}")
print(f"   Scaler type: {type(ml.scaler).__name__}")

# Test predictions
print("\n" + "=" * 70)
print("Sample Predictions")
print("=" * 70)

# Patient 1: Lower risk profile
patient1 = [
    35,   # age
    0,    # sex (Female)
    0,    # cp (no chest pain)
    120,  # trtbps (normal blood pressure)
    200,  # chol (normal cholesterol)
    0,    # fbs (no diabetes)
    0,    # restecg (normal ECG)
    170,  # thalachh (good max heart rate)
    0,    # exng (no exercise angina)
    0.0,  # oldpeak (no ST depression)
    0,    # slp
    0,    # caa (no major vessels)
    1     # thall
]

# Patient 2: Higher risk profile
patient2 = [
    65,   # age (older)
    1,    # sex (Male)
    3,    # cp (severe chest pain)
    160,  # trtbps (high blood pressure)
    300,  # chol (high cholesterol)
    1,    # fbs (diabetes)
    2,    # restecg (abnormal)
    110,  # thalachh (low max heart rate)
    1,    # exng (exercise angina)
    3.5,  # oldpeak (significant ST depression)
    2,    # slp
    2,    # caa (2 major vessels)
    3     # thall
]

patients = [patient1, patient2]
names = ["Patient 1 (Lower Risk Profile)", "Patient 2 (Higher Risk Profile)"]

for i, (patient, name) in enumerate(zip(patients, names)):
    print(f"\n{name}:")
    print(f"  Input: Age={patient[0]}, Sex={'Male' if patient[1] else 'Female'}, "
          f"BP={patient[3]}, Chol={patient[4]}")
    
    preds, probs = ml.predict([patient])
    prob_high = float(probs[0][0])
    risk_level, risk_percent = ml.to_risk(prob_high)
    
    print(f"  Prediction: {risk_level}")
    print(f"  Risk Score: {risk_percent:.2f}%")
    print(f"  Probabilities: High={prob_high:.3f}, Low={probs[0][1]:.3f}")

print("\n" + "=" * 70)
print("FastAPI Endpoints Ready")
print("=" * 70)
print("\nTo start the API server:")
print("  python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload")
print("\nTo run tests:")
print("  python -m pytest tests/ -v")
print("\nTo retrain model:")
print("  python retrain_model.py")
print("\n" + "=" * 70)
