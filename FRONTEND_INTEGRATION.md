# Frontend Integration Guide

## 🎯 Overview

The Heart Attack Risk Predictor has a complete **full-stack architecture**:

- **Backend**: FastAPI (Python) - Machine Learning & Data Processing
- **Frontend**: Streamlit (Python) - Interactive Web Interface
- **Database**: SQLite - Prediction Logs

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         User Browser                         │
│                    http://localhost:8501                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  Streamlit Frontend (app.py)                 │
│  • Patient data input forms                                  │
│  • Interactive UI with sliders/selectors                     │
│  • Risk visualization & recommendations                      │
│  • Model retraining interface                                │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP REST API
                           │ POST /predict
                           │ POST /train
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              FastAPI Backend (backend/main.py)               │
│  • GET  /          - API info                                │
│  • GET  /health    - Health check                            │
│  • POST /predict   - Heart attack risk prediction            │
│  • POST /train     - Model retraining                        │
│  • GET  /docs      - Interactive API documentation           │
└──────────────────────────┬──────────────────────────────────┘
                           │
                ┌──────────┴──────────┐
                ▼                     ▼
    ┌──────────────────┐  ┌──────────────────┐
    │  ML Service      │  │  SQLite Database │
    │  • Model Loading │  │  • Predictions   │
    │  • Predictions   │  │  • Logs          │
    │  • Training      │  └──────────────────┘
    └──────────────────┘
                │
                ▼
    ┌──────────────────┐
    │  Model Artifacts │
    │  • model.pkl     │
    │  • scaler.pkl    │
    └──────────────────┘
```

## 🚀 Quick Start

### Option 1: Use Startup Script (Recommended)
```bash
./start_services.sh
```

This will:
- ✅ Start FastAPI backend on port 8000
- ✅ Start Streamlit frontend on port 8501
- ✅ Verify both services are running
- ✅ Provide status and access URLs

**Stop services:**
```bash
./stop_services.sh
```

### Option 2: Manual Start

**Terminal 1 - Backend:**
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 - Frontend:**
```bash
export BACKEND_URL=http://localhost:8000
python -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### Option 3: Docker Compose
```bash
docker compose up --build
```

## 🌐 Access Points

Once running, you can access:

| Service | URL | Description |
|---------|-----|-------------|
| **Streamlit App** | http://localhost:8501 | Main user interface |
| **API Root** | http://localhost:8000 | API information |
| **API Health** | http://localhost:8000/health | Health check |
| **Swagger Docs** | http://localhost:8000/docs | Interactive API docs |
| **ReDoc** | http://localhost:8000/redoc | Alternative API docs |

## 📱 Frontend Features

### 1. **Patient Risk Assessment**
- Interactive form with 13 clinical parameters
- Real-time validation
- User-friendly labels (e.g., "Male/Female" instead of 1/0)
- Slider controls for numeric values
- Submit button to get predictions

### 2. **Risk Visualization**
- Color-coded risk levels:
  - 🟢 **LOW RISK** (<40%): Green success message
  - 🟡 **MODERATE RISK** (40-70%): Yellow warning
  - 🔴 **HIGH RISK** (>70%): Red error message
- Risk percentage display
- Probability breakdown (high vs low)
- Personalized recommendations

### 3. **Risk Factor Analysis**
- Automatic identification of contributing factors
- Bulleted list of detected risk factors
- Clinical interpretation for each factor

### 4. **Model Retraining Interface**
- Sidebar upload widget for CSV datasets
- Automatic backend training trigger
- Real-time training status and metrics
- Model version tracking

### 5. **Educational Content**
- Sidebar information panel
- Risk level explanations
- Key risk factor descriptions
- Medical disclaimer

## 🔌 Frontend-Backend Integration

### Communication Flow

1. **User Input** → Streamlit form
2. **Data Validation** → Client-side checks
3. **API Request** → POST to `/predict` endpoint
4. **Payload Format**:
```json
{
  "data": [{
    "age": 55,
    "sex": 1,
    "cp": 2,
    "trtbps": 145,
    "chol": 280,
    "fbs": 1,
    "restecg": 1,
    "thalachh": 140,
    "exng": 1,
    "oldpeak": 2.0,
    "slp": 1,
    "caa": 1,
    "thall": 2
  }],
  "client": "streamlit_app"
}
```

5. **API Response**:
```json
{
  "results": [{
    "risk_percent": 56.41,
    "risk_level": "MODERATE RISK",
    "probabilities": {
      "high": 0.564,
      "low": 0.436
    }
  }],
  "model_version": "v1"
}
```

6. **Result Display** → Formatted in Streamlit UI

### Error Handling

The frontend handles:
- ✅ Connection errors (backend not running)
- ✅ API errors (4xx, 5xx responses)
- ✅ Timeout errors
- ✅ Invalid responses
- ✅ File upload errors

Example error display:
```python
try:
    resp = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=20)
    if resp.status_code != 200:
        st.error(f"Prediction failed: {resp.status_code} {resp.text}")
        st.stop()
    data = resp.json()
except requests.RequestException as e:
    st.error(f"Could not reach backend at {BACKEND_URL}. Error: {e}")
    st.stop()
```

## 🎨 UI Components

### Input Section
```python
# Organized in 2 columns for better UX
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", [...])
    # ... more inputs

with col2:
    thalachh = st.slider("Max Heart Rate", 70, 200, 150)
    # ... more inputs
```

### Results Display
```python
if risk_percent >= 70:
    st.error(f"🔴 HIGH RISK: {risk_percent:.1f}%")
elif risk_percent >= 40:
    st.warning(f"🟡 MODERATE RISK: {risk_percent:.1f}%")
else:
    st.success(f"🟢 LOW RISK: {risk_percent:.1f}%")
```

### Sidebar Information
```python
with st.sidebar:
    st.header("About")
    st.write("Description...")
    
    st.divider()
    st.subheader("Train on Indian Dataset")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    # ... training logic
```

## 🔧 Configuration

### Environment Variables
```bash
# Backend URL (default: http://localhost:8000)
export BACKEND_URL=http://localhost:8000

# For production deployment:
export BACKEND_URL=https://api.yourdomain.com
```

### Streamlit Config
Create `.streamlit/config.toml`:
```toml
[server]
port = 8501
address = "0.0.0.0"
headless = true

[browser]
gatherUsageStats = false
serverAddress = "localhost"
serverPort = 8501
```

## 📊 Testing the Integration

### 1. Test Backend API
```bash
curl http://localhost:8000/health
```

### 2. Test Prediction via API
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [{"age": 55, "sex": 1, "cp": 2, "trtbps": 145, "chol": 280, "fbs": 1, "restecg": 1, "thalachh": 140, "exng": 1, "oldpeak": 2.0, "slp": 1, "caa": 1, "thall": 2}]}'
```

### 3. Test Frontend
- Open http://localhost:8501
- Fill in patient data
- Click "Predict Risk"
- Verify results display correctly

### 4. Test Model Retraining
- Use sidebar upload widget
- Upload `data/sample_indian_heart.csv`
- Verify training completes
- Check new model version

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Frontend can't connect to backend | Check BACKEND_URL environment variable |
| Port already in use | Use different ports or stop conflicting services |
| Import errors | Run `pip install -r requirements.txt` |
| Model not found | Run `python retrain_model.py` to generate model |
| CORS errors | Check CORS_ORIGINS in backend config |

## 🚀 Production Deployment

### Frontend Deployment Options

1. **Streamlit Cloud** (easiest)
   - Push to GitHub
   - Connect repository at streamlit.io
   - Set BACKEND_URL secret

2. **Docker**
   - Use provided `docker-compose.yml`
   - Deploy to AWS ECS, Azure Container Apps, etc.

3. **Custom Server**
   ```bash
   python -m streamlit run app.py \
     --server.port 8501 \
     --server.address 0.0.0.0 \
     --server.headless true
   ```

### Security Considerations

- ✅ Use HTTPS in production
- ✅ Set proper CORS origins
- ✅ Add authentication/authorization
- ✅ Rate limiting on API
- ✅ Input validation on both frontend and backend
- ✅ Secure API keys and secrets

## 📚 Additional Resources

- **Streamlit Docs**: https://docs.streamlit.io/
- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **API Reference**: See `API_REFERENCE.md`
- **Model Training**: See `MODEL_RETRAINING_SUMMARY.md`

## 🎉 Summary

The frontend integration is **complete and production-ready**:

✅ Full-stack application with clean separation of concerns  
✅ Interactive Streamlit UI with rich visualizations  
✅ Robust error handling and user feedback  
✅ RESTful API communication  
✅ Model retraining capabilities  
✅ Comprehensive documentation  
✅ Easy deployment options  
✅ Health monitoring and logging  

**Get started:** `./start_services.sh` and open http://localhost:8501
