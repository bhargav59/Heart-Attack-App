# Heart Attack Prediction API - Usage Guide

## 🎯 Recommended Endpoint: `/predict_real`

Use the **`/predict_real`** endpoint for production applications - it's trained on real hospital data with 86.89% accuracy and 92.38% ROC AUC.

---

## 📍 Available Endpoints

| Endpoint | Features | Dataset | Accuracy | ROC AUC | Status |
|----------|----------|---------|----------|---------|--------|
| **`/predict_real`** ⭐ | 56 clinical | Z-Alizadeh Sani (Real) | **86.89%** | **92.38%** | **RECOMMENDED** |
| `/predict` | 13 standard | UCI Heart Disease | ~85% | ~0.90 | Active |
| `/predict_indian` | 23 features | Synthetic (Kaggle) | 69.05% | 0.48 | Deprecated |

---

## 🚀 Quick Start

### Base URL
```
http://localhost:8000
```

### Test the API
```bash
# Check API health
curl http://localhost:8000/health

# View all endpoints
curl http://localhost:8000/
```

---

## 💊 `/predict_real` Endpoint (RECOMMENDED)

### Request Format

**POST** `/predict_real`

**Headers:**
```
Content-Type: application/json
```

**Body (56 required fields):**

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

### Response Format

```json
{
  "prediction": 1,
  "probability": 0.9991,
  "risk_level": "HIGH RISK",
  "risk_percentage": 99.91,
  "model_info": {
    "model_name": "Stacking",
    "dataset": "Z-Alizadeh Sani (UCI)",
    "n_features": 40,
    "accuracy": 0.8689,
    "roc_auc": 0.9238,
    "f1_score": 0.9111,
    "training_date": "2025-11-11"
  }
}
```

### Field Descriptions

#### Demographics
- **Age** (int): Age in years (18-120)
- **Sex** (str): "Male" or "Fmale"
- **Weight** (float): Weight in kg (30-200)
- **Length** (float): Height in cm (100-250)
- **BMI** (float): Body Mass Index (10-60)

#### Risk Factors
- **DM** (int): Diabetes Mellitus (0=No, 1=Yes)
- **HTN** (int): Hypertension (0=No, 1=Yes)
- **Current Smoker** (int): Currently smoking (0=No, 1=Yes)
- **EX-Smoker** (int): Former smoker (0=No, 1=Yes)
- **FH** (str): Family History ("N" or "Y")
- **Obesity** (str): Obese ("N" or "Y")

#### Medical History
- **CRF** (str): Chronic Renal Failure ("N" or "Y")
- **CVA** (str): Cerebrovascular Accident/Stroke ("N" or "Y")
- **Airway disease** (str): Asthma/COPD ("N" or "Y")
- **Thyroid Disease** (str): Thyroid disorder ("N" or "Y")
- **CHF** (str): Congestive Heart Failure ("N" or "Y")
- **DLP** (str): Dyslipidemia/High cholesterol ("N" or "Y")

#### Vital Signs
- **BP** (int): Blood Pressure - systolic (60-250 mmHg)
- **PR** (int): Pulse Rate (40-200 bpm)
- **Edema** (int): Swelling (0=No, 1=Yes)

#### Physical Exam
- **Weak Peripheral Pulse** (str): Weak pulse ("N" or "Y")
- **Lung rales** (str): Lung crackles ("N" or "Y")
- **Systolic Murmur** (str): Heart murmur during systole ("N" or "Y")
- **Diastolic Murmur** (str): Heart murmur during diastole ("N" or "Y")

#### Symptoms
- **Typical Chest Pain** (int): Classic angina (0=No, 1=Yes)
- **Dyspnea** (str): Shortness of breath ("N" or "Y")
- **Function Class** (int): NYHA functional class (0-4)
- **Atypical** (str): Atypical symptoms ("N" or "Y")
- **Nonanginal** (str): Non-cardiac chest pain ("N" or "Y")
- **Exertional CP** (str): Exercise-induced chest pain ("N" or "Y")
- **LowTH Ang** (str): Low threshold angina ("N" or "Y")

#### ECG Findings
- **Q Wave** (int): Pathological Q wave (0=No, 1=Yes)
- **St Elevation** (int): ST segment elevation (0=No, 1=Yes)
- **St Depression** (int): ST segment depression (0=No, 1=Yes)
- **Tinversion** (int): T wave inversion (0=No, 1=Yes)
- **LVH** (str): Left Ventricular Hypertrophy ("N" or "Y")
- **Poor R Progression** (str): Poor R wave progression ("N" or "Y")
- **BBB** (str): Bundle Branch Block ("N", "LBBB", or "RBBB")

#### Laboratory Tests
- **FBS** (float): Fasting Blood Sugar (50-500 mg/dL)
- **CR** (float): Creatinine (0.3-15 mg/dL)
- **TG** (float): Triglycerides (30-1000 mg/dL)
- **LDL** (float): LDL Cholesterol (20-400 mg/dL)
- **HDL** (float): HDL Cholesterol (10-150 mg/dL)
- **BUN** (float): Blood Urea Nitrogen (5-200 mg/dL)
- **ESR** (float): Erythrocyte Sedimentation Rate (0-200 mm/hr)
- **HB** (float): Hemoglobin (5-20 g/dL)
- **K** (float): Potassium (2-8 mEq/L)
- **Na** (float): Sodium (120-160 mEq/L)
- **WBC** (float): White Blood Cells (1000-50000 cells/μL)
- **Lymph** (float): Lymphocytes percentage (0-100%)
- **Neut** (float): Neutrophils percentage (0-100%)
- **PLT** (float): Platelets (50-1000 ×10³/μL)

#### Echocardiography
- **EF-TTE** (float): Ejection Fraction (10-80%)
- **Region RWMA** (int): Regional Wall Motion Abnormality regions (0-5)
- **VHD** (str): Valvular Heart Disease ("N", "mild", "Moderate", "Severe")

### Risk Interpretation

| Risk Level | Risk % | Interpretation |
|-----------|--------|----------------|
| **LOW RISK** | 0-39% | Low probability of CAD |
| **MODERATE RISK** | 40-69% | Moderate CAD risk, further testing recommended |
| **HIGH RISK** | 70-100% | High CAD probability, immediate medical attention |

---

## 🧪 Testing

Run the included test script:

```bash
python test_real_endpoint.py
```

This tests both HIGH RISK and LOW RISK patient profiles.

---

## 📊 Model Performance

### Z-Alizadeh Sani Model (Real Data)
- **Dataset**: 303 Asian patients from real hospital coronary angiography
- **Source**: UCI Machine Learning Repository (peer-reviewed)
- **Training**: Stacking Ensemble (6 models)
- **Accuracy**: 86.89%
- **ROC AUC**: 92.38%
- **F1 Score**: 91.11%
- **Sensitivity**: 88.4% (catches 88.4% of CAD cases)
- **Specificity**: 83.3% (correctly identifies 83.3% of normal cases)

### Feature Engineering
- **Input**: 56 clinical features
- **Engineered**: 18 additional features (CV risk score, ECG abnormality, metabolic syndrome, lipid ratios, etc.)
- **Selected**: 40 best features by mutual information
- **Preprocessing**: Label encoding, RobustScaler, SMOTE-Tomek balancing

---

## 🔧 Troubleshooting

### Common Issues

1. **500 Error: Type mismatch**
   - Ensure all categorical fields use correct values ("Y"/"N", "Male"/"Fmale", etc.)
   - Check numeric fields are numbers, not strings

2. **422 Validation Error**
   - Verify all 56 required fields are present
   - Check field names match exactly (including spaces like "Current Smoker")
   - Ensure values are within valid ranges

3. **Server not responding**
   - Check if server is running: `curl http://localhost:8000/health`
   - Start server: `uvicorn backend.main:app --host 0.0.0.0 --port 8000`

---

## 📚 Additional Resources

- **Training Report**: `REAL_DATA_TRAINING_REPORT.md`
- **Model Results**: `z_alizadeh_model_results.csv`
- **API Documentation**: http://localhost:8000/docs (Swagger UI)
- **ReDoc**: http://localhost:8000/redoc

---

## 🎯 Production Deployment

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Deployment
```bash
# Build and start services
docker-compose up -d

# Check logs
docker-compose logs -f backend
```

### Health Check
```bash
curl http://your-domain/health
```

---

**Status**: ✅ Production Ready  
**Last Updated**: November 11, 2025  
**Model Version**: v1.0_z_alizadeh_sani
