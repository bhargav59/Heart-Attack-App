# API Quick Reference Guide

## ✅ Issue Fixed!

The `{"detail":"Not Found"}` error was happening because you were accessing the root URL `/` which didn't have an endpoint defined. This has now been fixed!

## 📍 Available Endpoints

### 1. Root Endpoint (NEW!)
```bash
GET http://localhost:8000/
```
Returns API information and available endpoints.

### 2. Health Check
```bash
GET http://localhost:8000/health
```
Returns: `{"status": "ok"}`

### 3. Predict Heart Attack Risk
```bash
POST http://localhost:8000/predict
Content-Type: application/json

{
  "data": [{
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
  }],
  "client": "optional_client_id"
}
```

**Field Descriptions:**
- `age`: Age (18-100)
- `sex`: Gender (1=Male, 0=Female)
- `cp`: Chest pain type (0-3)
- `trtbps`: Resting blood pressure (70-250 mmHg)
- `chol`: Cholesterol (80-700 mg/dl)
- `fbs`: Fasting blood sugar > 120 mg/dl (1=true, 0=false)
- `restecg`: Resting ECG results (0-2)
- `thalachh`: Maximum heart rate achieved (60-220)
- `exng`: Exercise induced angina (1=yes, 0=no)
- `oldpeak`: ST depression induced by exercise (0-10)
- `slp`: Slope of peak exercise ST segment (0-2)
- `caa`: Number of major vessels colored by fluoroscopy (0-3)
- `thall`: Thalassemia (0-3)

### 4. Train Model
```bash
POST http://localhost:8000/train
Content-Type: application/json

{
  "dataset_path": "data/your_dataset.csv",
  "target_column": "target"
}
```

### 5. Interactive Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🧪 Quick Test Commands

**Test all endpoints:**
```bash
./test_endpoints.sh
```

**Test root endpoint:**
```bash
curl http://localhost:8000/
```

**Test prediction:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [{
      "age": 55, "sex": 1, "cp": 0, "trtbps": 140, "chol": 260,
      "fbs": 0, "restecg": 1, "thalachh": 150, "exng": 0,
      "oldpeak": 1.2, "slp": 1, "caa": 0, "thall": 2
    }]
  }'
```

**Use Python:**
```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Prediction
payload = {
    "data": [{
        "age": 45, "sex": 1, "cp": 2, "trtbps": 130, "chol": 250,
        "fbs": 1, "restecg": 1, "thalachh": 150, "exng": 0,
        "oldpeak": 1.5, "slp": 1, "caa": 1, "thall": 2
    }]
}
response = requests.post("http://localhost:8000/predict", json=payload)
print(response.json())
```

## 🚀 Server Commands

**Start server:**
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**Check if server is running:**
```bash
curl http://localhost:8000/health
```

**Stop server:**
```bash
pkill -f uvicorn
```

## 🐛 Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| `{"detail":"Not Found"}` | Accessing undefined endpoint | Use one of the defined endpoints above |
| Connection refused | Server not running | Start server with uvicorn command |
| 422 Validation Error | Invalid input data | Check field types and ranges |
| 500 Internal Error | Model not loaded | Ensure model files exist in `models/` directory |

## 📊 Response Format

**Successful prediction response:**
```json
{
  "results": [{
    "risk_percent": 53.07,
    "risk_level": "MODERATE RISK",
    "probabilities": {
      "high": 0.531,
      "low": 0.469
    }
  }],
  "model_version": "v1"
}
```

**Risk Levels:**
- `LOW RISK`: < 40%
- `MODERATE RISK`: 40-70%
- `HIGH RISK`: > 70%
