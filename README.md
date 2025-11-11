# Heart Attack Risk Predictor — Full‑Stack

End-to-end heart attack risk prediction app with:

- Modern web interface + Streamlit frontend
- FastAPI backend with multiple prediction models
- Real medical data from hospital coronary angiography (Z-Alizadeh Sani dataset)
- SQLite persistence for prediction logs (Postgres ready)
- Dockerized local deployment and CI tests
- **✨ 86.89% accuracy, 92.38% ROC AUC on real hospital data (303 patients)**

## Features

### Three Prediction Models:

1. **`/predict_real` (RECOMMENDED)** ⭐
   - 56 clinical features from Z-Alizadeh Sani dataset (UCI)
   - Real hospital data: 303 Asian patients with coronary angiography
   - **86.89% accuracy, 92.38% ROC AUC, 91.11% F1 Score**
   - Features: Demographics, Risk Factors, ECG, Labs, Echo
   - Stacking Ensemble (RF, ET, GB, XGBoost, LightGBM, CatBoost)

2. **`/predict` (Standard)**
   - 13 clinical features (UCI Heart Disease dataset)
   - Classic heart disease indicators
   - ~85% accuracy

3. **`/predict_indian` (Deprecated)**
   - 23 features from synthetic data
   - Not recommended for production use

### Web Interface
- **Modern HTML5 Interface**: `frontend/index.html` ⭐
  - Beautiful gradient design
  - 56 clinical input fields organized by category
  - Quick-fill test cases (High Risk / Low Risk patients)
  - Real-time predictions with color-coded risk levels
  - Model performance metrics display

- **Streamlit Interface**: `app.py`
  - 13 standard features
  - Simple slider-based input

### API Features
- Multiple model endpoints with different feature sets
- Probability-based risk assessment with clear risk levels (LOW/MODERATE/HIGH)
- Database logging of all predictions
- API endpoints: `/health`, `/predict`, `/predict_real`, `/docs`
- Interactive API documentation (Swagger UI)

## Project structure

```
backend/              # FastAPI service (prediction, logging)
├── main.py          # FastAPI app with 3 prediction endpoints
├── ml_service.py    # Standard 13-feature model
├── ml_service_z_alizadeh.py  # Real 56-feature model ⭐
├── ml_service_indian.py      # Deprecated Indian model
├── schemas*.py      # Pydantic models for validation
├── database.py      # SQLite database
└── models.py        # Database models

frontend/            # Web interfaces
├── index.html       # Modern HTML5 interface ⭐
└── (Streamlit: app.py in root)

ml/                  # Training utilities
├── train.py         # Training pipeline
└── feature_engineering.py  # Feature creation

models/              # Trained model artifacts
├── heart_attack_model_real.pkl  # Real model ⭐
├── scaler_real.pkl             # RobustScaler ⭐
└── feature_names_real.pkl      # 40 selected features ⭐

data/                # Datasets and processing
├── real_datasets/
│   └── z_alizadeh_sani/
│       └── z_alizadeh_sani.csv  # Real medical data ⭐
├── download_z_alizadeh_sani.py
└── process_z_alizadeh_sani.py

app.py               # Streamlit frontend (13 features)
train_z_alizadeh_model.py  # Training script for real model ⭐
test_real_endpoint.py  # Test suite for real model ⭐
requirements.txt     # Python dependencies
docker-compose.yml   # Docker setup
```

## Quick start (local)

### Super Quick Start (One Command) 🚀

```bash
./run_app.sh
```

This script will:
- ✅ Start the FastAPI backend on port 8000
- ✅ Open the HTML frontend in your browser
- ✅ Check backend health
- ✅ Display access points and logs

To stop: Press `Ctrl+C` or run `./stop_app.sh`

### Manual Setup

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start backend API

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Choose your interface:

**Option A: Modern Web Interface (Recommended)** ⭐
```bash
# Open frontend/index.html in your browser
open frontend/index.html  # macOS
# Or: start frontend/index.html  (Windows)
# Or: xdg-open frontend/index.html  (Linux)
```
- 56 clinical features (real medical data)
- Beautiful UI with quick-fill test cases
- Test with High Risk and Low Risk patient examples

**Option B: Streamlit Interface**
```bash
export BACKEND_URL=http://localhost:8000
streamlit run app.py
```
- 13 standard features
- Access at: http://localhost:8501

### 4. API Documentation

Visit http://localhost:8000/docs for interactive API documentation (Swagger UI)

## Train your own model

The app comes with pre-trained models on real medical data. To retrain:

```bash
python train_z_alizadeh_model.py
```

This will:
- Load the Z-Alizadeh Sani dataset (303 real patients from hospital)
- Perform feature engineering (56 → 74 → 40 features)
- Train a Stacking Ensemble (6 base models + meta-learner)
- Use SMOTE-Tomek for class balancing
- Save models to `models/*_real.pkl`

Expected results:
- **Accuracy**: ~86.89%
- **ROC AUC**: ~92.38%
- **F1 Score**: ~91.11%

See `REAL_DATA_TRAINING_REPORT.md` for detailed training metrics.

## Docker (quick demo)

```bash
docker compose up --build
```

Then visit:
- Frontend: http://localhost:8501 (Streamlit)
- Backend: http://localhost:8000
- API Docs: http://localhost:8000/docs

## API examples

### Predict with Real Model (Recommended) ⭐

```bash
curl -X POST http://localhost:8000/predict_real \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 67, "Sex": "Male", "Weight": 80, "Length": 175, "BMI": 26.12,
    "DM": 1, "HTN": 1, "Current Smoker": 1, "EX-Smoker": 0,
    "FH": "Y", "Obesity": "Y", "CRF": "N", "CVA": "N",
    "Airway disease": "N", "Thyroid Disease": "N", "CHF": "N", "DLP": "Y",
    "BP": 150, "PR": 85, "Edema": 0,
    "Weak Peripheral Pulse": "N", "Lung rales": "N",
    "Systolic Murmur": "N", "Diastolic Murmur": "N",
    "Typical Chest Pain": 1, "Dyspnea": "Y", "Function Class": 3,
    "Atypical": "N", "Nonanginal": "N", "Exertional CP": "N", "LowTH Ang": "N",
    "Q Wave": 1, "St Elevation": 0, "St Depression": 1, "Tinversion": 1,
    "LVH": "Y", "Poor R Progression": "N", "BBB": "N",
    "FBS": 180, "CR": 1.2, "TG": 220, "LDL": 160, "HDL": 32,
    "BUN": 22, "ESR": 35, "HB": 14.0, "K": 4.3, "Na": 140,
    "WBC": 9500, "Lymph": 28, "Neut": 68, "PLT": 280,
    "EF-TTE": 38, "Region RWMA": 3, "VHD": "mild"
  }'
```

### Predict with Standard Model

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

## Testing

Run the comprehensive test suite:

```bash
python test_real_endpoint.py
```

This tests both HIGH RISK and LOW RISK patient scenarios with the real model.

## Configuration

Copy `.env.example` to `.env` and adjust as needed:

- `BACKEND_URL` — where Streamlit calls the API (default: http://localhost:8000)
- `DATABASE_URL` — database connection (default: SQLite, supports Postgres)
- `CORS_ORIGINS` — allowed origins for API requests

## Documentation

- **README.md** (this file) — Quick start and overview
- **API_USAGE_GUIDE.md** — Complete API documentation with all 56 fields
- **API_REFERENCE.md** — API endpoint reference
- **GETTING_STARTED.md** — Detailed setup instructions
- **REAL_DATA_TRAINING_REPORT.md** — Model training report and metrics

## Development notes

- All dependencies are in `requirements.txt`
- Models stored in `models/` (gitignored, only `*_real.pkl` files are used in production)
- The `/predict_indian` endpoint uses deprecated synthetic data
- **Use `/predict_real` for production applications**

## Disclaimer

This app is for educational and research purposes only. It is NOT a medical device and should not be used for clinical diagnosis. Always consult qualified healthcare professionals for medical advice.

## License

See LICENSE file for details.
