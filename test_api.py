#!/usr/bin/env python3
"""
Script to test the FastAPI endpoints
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    print("🔍 Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    return response.status_code == 200

def test_predict():
    print("\n🔍 Testing /predict endpoint...")
    
    # Sample data for a patient
    payload = {
        "data": [
            {
                "age": 45,
                "sex": 1,  # Male
                "cp": 2,   # Chest pain type
                "trtbps": 130,  # Resting blood pressure
                "chol": 250,    # Cholesterol
                "fbs": 1,       # Fasting blood sugar > 120
                "restecg": 1,   # Resting ECG
                "thalachh": 150,  # Max heart rate
                "exng": 0,      # Exercise induced angina
                "oldpeak": 1.5,  # ST depression
                "slp": 1,       # Slope
                "caa": 1,       # Number of major vessels
                "thall": 2      # Thalassemia
            }
        ],
        "client": "test_script"
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"   Model Version: {result.get('model_version')}")
        for i, res in enumerate(result.get('results', [])):
            print(f"\n   Patient {i+1}:")
            print(f"      Risk Level: {res.get('risk_level')}")
            print(f"      Risk Percent: {res.get('risk_percent'):.2f}%")
            print(f"      Probabilities: {res.get('probabilities')}")
    else:
        print(f"   Error: {response.text}")
    
    return response.status_code == 200

def main():
    print("=" * 60)
    print("FastAPI Heart Attack Prediction - API Tests")
    print("=" * 60)
    
    try:
        health_ok = test_health()
        predict_ok = test_predict()
        
        print("\n" + "=" * 60)
        print("Test Summary:")
        print(f"  Health Endpoint: {'✅ PASS' if health_ok else '❌ FAIL'}")
        print(f"  Predict Endpoint: {'✅ PASS' if predict_ok else '❌ FAIL'}")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API. Make sure the server is running on port 8000.")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
