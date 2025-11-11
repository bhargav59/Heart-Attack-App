import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_root_endpoint():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "endpoints" in data
    assert "model_version" in data

def test_health_endpoint():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_endpoint():
    """Test the prediction endpoint with sample data"""
    payload = {
        "data": [
            {
                "age": 45,
                "sex": 1,
                "cp": 2,
                "trtbps": 130,
                "chol": 250,
                "fbs": 1,
                "restecg": 1,
                "thalachh": 150,
                "exng": 0,
                "oldpeak": 1.5,
                "slp": 1,
                "caa": 1,
                "thall": 2
            }
        ],
        "client": "pytest"
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    result = response.json()
    assert "results" in result
    assert "model_version" in result
    assert len(result["results"]) == 1
    
    prediction = result["results"][0]
    assert "risk_percent" in prediction
    assert "risk_level" in prediction
    assert "probabilities" in prediction
    assert prediction["risk_level"] in ["LOW RISK", "MODERATE RISK", "HIGH RISK"]
    assert 0 <= prediction["risk_percent"] <= 100

def test_predict_multiple_patients():
    """Test prediction with multiple patients"""
    payload = {
        "data": [
            {
                "age": 45, "sex": 1, "cp": 2, "trtbps": 130, "chol": 250,
                "fbs": 1, "restecg": 1, "thalachh": 150, "exng": 0,
                "oldpeak": 1.5, "slp": 1, "caa": 1, "thall": 2
            },
            {
                "age": 60, "sex": 0, "cp": 3, "trtbps": 150, "chol": 300,
                "fbs": 1, "restecg": 2, "thalachh": 120, "exng": 1,
                "oldpeak": 2.5, "slp": 2, "caa": 2, "thall": 3
            }
        ]
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    result = response.json()
    assert len(result["results"]) == 2

def test_predict_validation_error():
    """Test that validation errors are properly handled"""
    payload = {
        "data": [
            {
                "age": 200,  # Invalid age
                "sex": 1,
                "cp": 2,
                "trtbps": 130,
                "chol": 250,
                "fbs": 1,
                "restecg": 1,
                "thalachh": 150,
                "exng": 0,
                "oldpeak": 1.5,
                "slp": 1,
                "caa": 1,
                "thall": 2
            }
        ]
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error
