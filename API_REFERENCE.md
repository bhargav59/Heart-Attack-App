# API Reference Guide

Complete API documentation for the Heart Attack Risk Predictor backend.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently no authentication required. All endpoints are publicly accessible.

## Endpoints

### 1. Root Information

```http
GET /
```

Returns API information and available endpoints.

**Response:**
```json
{
  "message": "Heart Attack Risk Predictor API",
  "version": "2.0",
  "endpoints": {
    "health": "/health",
    "predict_standard": "/predict",
    "predict_real": "/predict_real",
    "predict_indian": "/predict_indian (deprecated)",
    "docs": "/docs"
  }
}
```

---

### 2. Health Check

```http
GET /health
```

Returns server health status.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

### 3. Predict (Standard Model)

```http
POST /predict
```

Predict heart attack risk using the standard 13-feature model (~85% accuracy).

**Request Body:**
```json
{
  "data": [{
    "age": 55,
    "sex": 1,
    "cp": 0,
    "trtbps": 140,
    "chol": 260,
    "fbs": 0,
    "restecg": 1,
    "thalachh": 150,
    "exng": 0,
    "oldpeak": 1.2,
    "slp": 1,
    "caa": 0,
    "thall": 2
  }],
  "client": "optional_client_id"
}
```

**Field Descriptions:**
- `age` (int): Age in years (18-100)
- `sex` (int): Gender (1=Male, 0=Female)
- `cp` (int): Chest pain type (0-3)
- `trtbps` (int): Resting blood pressure in mm Hg (70-250)
- `chol` (int): Cholesterol in mg/dl (80-700)
- `fbs` (int): Fasting blood sugar > 120 mg/dl (1=true, 0=false)
- `restecg` (int): Resting ECG results (0-2)
- `thalachh` (int): Maximum heart rate achieved (60-220)
- `exng` (int): Exercise induced angina (1=yes, 0=no)
- `oldpeak` (float): ST depression (0-10)
- `slp` (int): Slope of peak exercise ST segment (0-2)
- `caa` (int): Number of major vessels (0-3)
- `thall` (int): Thalassemia (0-3)

**Response:**
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

---

### 4. Predict Real (Recommended) ⭐

```http
POST /predict_real
```

Predict heart attack risk using the real hospital data model (56 features, **86.89% accuracy, 92.38% ROC AUC**).

**Request Body:**
```json
{
  "Age": 67,
  "Sex": "Male",
  "Weight": 80,
  "Length": 175,
  "BMI": 26.12,
  "DM": 1,
  "HTN": 1,
  "Current Smoker": 1,
  "EX-Smoker": 0,
  "FH": "Y",
  "Obesity": "Y",
  "CRF": "N",
  "CVA": "N",
  "Airway disease": "N",
  "Thyroid Disease": "N",
  "CHF": "N",
  "DLP": "Y",
  "BP": 150,
  "PR": 85,
  "Edema": 0,
  "Weak Peripheral Pulse": "N",
  "Lung rales": "N",
  "Systolic Murmur": "N",
  "Diastolic Murmur": "N",
  "Typical Chest Pain": 1,
  "Dyspnea": "Y",
  "Function Class": 3,
  "Atypical": "N",
  "Nonanginal": "N",
  "Exertional CP": "N",
  "LowTH Ang": "N",
  "Q Wave": 1,
  "St Elevation": 0,
  "St Depression": 1,
  "Tinversion": 1,
  "LVH": "Y",
  "Poor R Progression": "N",
  "BBB": "N",
  "FBS": 180,
  "CR": 1.2,
  "TG": 220,
  "LDL": 160,
  "HDL": 32,
  "BUN": 22,
  "ESR": 35,
  "HB": 14.0,
  "K": 4.3,
  "Na": 140,
  "WBC": 9500,
  "Lymph": 28,
  "Neut": 68,
  "PLT": 280,
  "EF-TTE": 38,
  "Region RWMA": 3,
  "VHD": "mild"
}
```

**Complete Field Descriptions:**

See `API_USAGE_GUIDE.md` for detailed descriptions of all 56 fields, organized by category:
- Demographics (6 fields)
- Risk Factors (11 fields)
- Physical Exam (8 fields)
- Symptoms (9 fields)
- ECG Findings (8 fields)
- Laboratory Tests (14 fields)

**Response:**
```json
{
  "risk_percent": 99.91,
  "risk_level": "HIGH RISK",
  "prediction": 1,
  "probabilities": {
    "low_risk": 0.0009,
    "high_risk": 0.9991
  },
  "model_info": {
    "version": "v2_real",
    "accuracy": 0.8689,
    "roc_auc": 0.9238,
    "features_used": 40
  }
}
```

---

### 5. Predict Indian (Deprecated)

```http
POST /predict_indian
```

⚠️ **DEPRECATED**: This endpoint uses synthetic data and is not recommended for production use. Use `/predict_real` instead.

---

## Risk Levels

| Risk Level | Probability Range | Meaning |
|-----------|------------------|---------|
| **LOW RISK** | < 40% | Low probability of heart attack |
| **MODERATE RISK** | 40% - 70% | Moderate risk, monitoring recommended |
| **HIGH RISK** | > 70% | High risk, immediate medical evaluation recommended |

---

## Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request (invalid input) |
| 422 | Validation Error (field constraints violated) |
| 500 | Internal Server Error |

---

## Error Responses

**Validation Error (422):**
```json
{
  "detail": [
    {
      "loc": ["body", "Age"],
      "msg": "ensure this value is greater than or equal to 1",
      "type": "value_error.number.not_ge"
    }
  ]
}
```

**Server Error (500):**
```json
{
  "detail": "Internal server error occurred"
}
```

---

## Example Requests

### cURL

**Standard Model:**
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

**Real Model:**
```bash
curl -X POST http://localhost:8000/predict_real \
  -H "Content-Type: application/json" \
  -d @test_data.json
```

### Python

```python
import requests

# Standard model
payload = {
    "data": [{
        "age": 55, "sex": 1, "cp": 0, "trtbps": 140, "chol": 260,
        "fbs": 0, "restecg": 1, "thalachh": 150, "exng": 0,
        "oldpeak": 1.2, "slp": 1, "caa": 0, "thall": 2
    }]
}
response = requests.post("http://localhost:8000/predict", json=payload)
print(response.json())

# Real model
real_payload = {
    "Age": 67, "Sex": "Male", "Weight": 80, "Length": 175,
    # ... (all 56 fields)
}
response = requests.post("http://localhost:8000/predict_real", json=real_payload)
print(response.json())
```

### JavaScript

```javascript
// Real model
const payload = {
  Age: 67,
  Sex: "Male",
  Weight: 80,
  // ... (all 56 fields)
};

fetch('http://localhost:8000/predict_real', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(payload)
})
.then(response => response.json())
.then(data => console.log(data));
```

---

## Interactive Documentation

Visit **http://localhost:8000/docs** for:
- Interactive API testing (Swagger UI)
- Complete schema definitions
- Request/response examples
- Try it out functionality

Alternative documentation: **http://localhost:8000/redoc**

---

## Rate Limiting

Currently no rate limiting implemented. Consider adding rate limiting in production.

---

## CORS

CORS is enabled for all origins. Configure in `backend/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Database Logging

All predictions are logged to SQLite database (`predictions.db`). Each prediction includes:
- Timestamp
- Input features
- Prediction result
- Risk level
- Client ID (optional)

---

## Model Information

| Model | Features | Accuracy | ROC AUC | Dataset |
|-------|----------|----------|---------|---------|
| Standard | 13 | ~85% | - | UCI Heart Disease |
| Real ⭐ | 56 | 86.89% | 92.38% | Z-Alizadeh Sani |
| Indian (deprecated) | 23 | - | - | Synthetic |

**Recommended**: Use `/predict_real` for production applications.

---

## Support

For issues or questions:
- Check the documentation in `API_USAGE_GUIDE.md`
- Review `REAL_DATA_TRAINING_REPORT.md` for model details
- Open an issue on GitHub
