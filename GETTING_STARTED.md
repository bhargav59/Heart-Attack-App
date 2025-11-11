# Getting Started Guide

Complete setup guide for the Heart Attack Risk Predictor application.

## Prerequisites

- Python 3.8+ installed
- pip package manager
- Git (optional, for cloning the repository)
- 4GB RAM minimum
- Modern web browser (Chrome, Firefox, Safari, or Edge)

## Quick Start (5 minutes)

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows

# Install packages
pip install -r requirements.txt
```

### 2. Start the Backend

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

You should see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 3. Test the API

Open a new terminal and run:

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "ok", "model_loaded": true}
```

### 4. Choose Your Interface

#### Option A: Modern Web Interface (Recommended) ⭐

Simply open the HTML file in your browser:

```bash
# macOS
open frontend/index.html

# Windows
start frontend/index.html

# Linux
xdg-open frontend/index.html
```

**Features:**
- 56 clinical input fields
- Beautiful gradient design
- Quick-fill test cases
- Real-time predictions
- Color-coded risk levels
- Model performance metrics

#### Option B: Streamlit Interface

In a new terminal:

```bash
export BACKEND_URL=http://localhost:8000
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

**Features:**
- 13 standard features
- Simple slider-based input
- Lightweight interface

## Detailed Setup

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8 | 3.10+ |
| RAM | 4GB | 8GB+ |
| Storage | 500MB | 1GB |
| CPU | Dual-core | Quad-core |
| Browser | Any modern | Chrome/Firefox |

### Installation Steps

1. **Clone or Download the Project**

```bash
git clone https://github.com/yourusername/heart-attack-app.git
cd heart-attack-app
```

2. **Set Up Virtual Environment**

```bash
# Create
python -m venv .venv

# Activate
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Verify
which python  # Should show path inside .venv
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

This installs:
- FastAPI (backend framework)
- scikit-learn (ML models)
- pandas, numpy (data processing)
- uvicorn (ASGI server)
- sqlalchemy (database)
- pydantic (validation)
- And all other dependencies

4. **Verify Installation**

```bash
python -c "import fastapi, sklearn, pandas; print('All imports successful!')"
```

### Configuration

1. **Environment Variables** (Optional)

Create a `.env` file:

```bash
# Backend API URL (for Streamlit)
BACKEND_URL=http://localhost:8000

# Database URL
DATABASE_URL=sqlite:///./predictions.db

# CORS Origins (comma-separated)
CORS_ORIGINS=http://localhost:3000,http://localhost:8501
```

2. **Database Setup**

The database is created automatically on first run. No manual setup needed!

SQLite database file: `predictions.db`

## Usage Guide

### Using the Modern Web Interface ⭐

1. **Open the Interface**

```bash
open frontend/index.html  # macOS
```

2. **Fill in Patient Data**

The interface has 56 fields organized by category:

- **Demographics**: Age, Sex, Weight, Height, BMI
- **Risk Factors**: Diabetes, Hypertension, Smoking, Family History
- **Physical Exam**: Blood Pressure, Heart Rate, Edema
- **Symptoms**: Chest Pain, Shortness of Breath, Function Class
- **ECG Findings**: Q Wave, ST Changes, T Wave Inversion
- **Lab Tests**: Blood Sugar, Cholesterol, Kidney Function
- **Echo**: Ejection Fraction, Wall Motion Abnormalities

3. **Use Quick-Fill Test Cases**

Click the buttons to load pre-filled data:

- **High Risk Patient**: 99.91% risk (67 yo male with multiple risk factors)
- **Low Risk Patient**: 4.08% risk (51 yo female, healthy)

4. **Get Prediction**

Click **"Predict Risk"** to see:

- Risk percentage (0-100%)
- Risk level (LOW/MODERATE/HIGH) with color coding
- Probability breakdown
- Model performance metrics

### Using the Streamlit Interface

1. **Start Streamlit**

```bash
export BACKEND_URL=http://localhost:8000
streamlit run app.py
```

2. **Enter Data**

Use sliders and dropdowns to input 13 standard features.

3. **View Results**

Risk assessment displays automatically with:
- Risk level
- Percentage
- Risk factors identified
- Recommendations

### Using the API Directly

See **API_REFERENCE.md** for complete API documentation.

**Quick Example:**

```bash
curl -X POST http://localhost:8000/predict_real \
  -H "Content-Type: application/json" \
  -d @test_data.json
```

## Testing

### Test the Real Model

Run the comprehensive test suite:

```bash
python test_real_endpoint.py
```

This tests:
- HIGH RISK patient (expected: 99.91%)
- LOW RISK patient (expected: 4.08%)

### Test All Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Standard model
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data":[{"age":55,"sex":1,"cp":0,"trtbps":140,"chol":260,"fbs":0,"restecg":1,"thalachh":150,"exng":0,"oldpeak":1.2,"slp":1,"caa":0,"thall":2}]}'

# Real model (see API_USAGE_GUIDE.md for full payload)
curl -X POST http://localhost:8000/predict_real \
  -H "Content-Type: application/json" \
  -d @test_data.json
```

## Docker Deployment

### Quick Docker Start

```bash
docker compose up --build
```

Access:
- Frontend: http://localhost:8501
- Backend: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Docker Commands

```bash
# Build and start
docker compose up --build

# Start in background
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down

# Remove volumes
docker compose down -v
```

## Model Training

### Train Your Own Model

The application comes with pre-trained models, but you can retrain:

```bash
python train_z_alizadeh_model.py
```

This will:
1. Load the Z-Alizadeh Sani dataset (303 patients)
2. Perform feature engineering (56 → 74 → 40 features)
3. Apply SMOTE-Tomek for class balancing
4. Train a Stacking Ensemble:
   - Base models: RandomForest, ExtraTrees, GradientBoosting, XGBoost, LightGBM, CatBoost
   - Meta-learner: LogisticRegression
5. Save models to `models/`

**Expected Performance:**
- Accuracy: ~86.89%
- ROC AUC: ~92.38%
- F1 Score: ~91.11%

See `REAL_DATA_TRAINING_REPORT.md` for detailed metrics.

## Troubleshooting

### Common Issues

1. **"Module not found" errors**

```bash
# Make sure virtual environment is activated
source .venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

2. **"Port already in use" (8000 or 8501)**

```bash
# Find and kill process
lsof -ti:8000 | xargs kill  # macOS/Linux
lsof -ti:8501 | xargs kill

# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

3. **"Model file not found"**

Ensure these files exist:
- `models/heart_attack_model_real.pkl`
- `models/scaler_real.pkl`
- `models/feature_names_real.pkl`

If missing, run:
```bash
python train_z_alizadeh_model.py
```

4. **Frontend can't connect to backend**

- Check backend is running: `curl http://localhost:8000/health`
- Verify CORS settings in `backend/main.py`
- Check browser console for errors (F12)

5. **Database errors**

```bash
# Delete and recreate database
rm predictions.db
# Restart backend (will recreate automatically)
```

### Debug Mode

Start backend with debug logging:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload --log-level debug
```

## Next Steps

1. **Explore the Application**
   - Try the modern web interface with quick-fill test cases
   - Test different patient profiles
   - Compare risk predictions

2. **Learn the API**
   - Visit http://localhost:8000/docs
   - Try different endpoints
   - Review `API_USAGE_GUIDE.md`

3. **Understand the Model**
   - Read `REAL_DATA_TRAINING_REPORT.md`
   - Review feature engineering in `ml/feature_engineering.py`
   - Check model code in `backend/ml_service_z_alizadeh.py`

4. **Customize**
   - Modify the frontend interface
   - Add new features
   - Integrate with your system

## Production Deployment

### Security Considerations

1. **Authentication**: Add API authentication
2. **CORS**: Restrict allowed origins
3. **HTTPS**: Use SSL/TLS certificates
4. **Rate Limiting**: Implement rate limiting
5. **Input Validation**: Already implemented via Pydantic

### Recommended Stack

- **Backend**: FastAPI with Gunicorn/Uvicorn workers
- **Database**: PostgreSQL (replace SQLite)
- **Frontend**: Nginx serving static HTML
- **Container**: Docker with Docker Compose
- **Orchestration**: Kubernetes (for scale)

### Environment Variables for Production

```bash
DATABASE_URL=postgresql://user:pass@host:5432/dbname
CORS_ORIGINS=https://yourdomain.com
SECRET_KEY=your-secret-key
LOG_LEVEL=info
```

## Support and Documentation

- **README.md**: Project overview
- **API_REFERENCE.md**: Complete API documentation
- **API_USAGE_GUIDE.md**: All 56 field descriptions
- **REAL_DATA_TRAINING_REPORT.md**: Model training details
- **GETTING_STARTED.md**: This file

## License and Disclaimer

This application is for **educational and research purposes only**. It is NOT a medical device and should not be used for clinical diagnosis. Always consult qualified healthcare professionals for medical advice.

---

**Need Help?**
- Check the documentation
- Review code comments
- Open an issue on GitHub
- Contact the maintainers

Happy predicting! 🚀
