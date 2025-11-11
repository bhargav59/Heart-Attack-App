# Model Retraining Summary

## Dataset Used
- **Source**: Indian Heart Attack Prediction Dataset from Kaggle
- **Location**: `data/_kaggle_tmp/heart_attack_prediction_india.csv`
- **Records**: 10,000 patient records
- **Features**: 26 columns including demographic, health, and lifestyle factors

## Training Process

### Dataset Mapping
The training pipeline automatically maps the Indian dataset columns to the standard 13-feature schema:
- Age → age
- Gender → sex (Male=1, Female=0)
- Diabetes → fbs (fasting blood sugar)
- Hypertension + Obesity → caa (number of major vessels)
- Cholesterol_Level → chol
- Systolic_BP → trtbps (resting blood pressure)
- HDL_Level → thall (thalassemia, bucketed)
- Stress_Level → cp (chest pain type, bucketed)
- Physical_Activity → exng (exercise induced angina, inverse)
- And other derived features...

### Model Training Results
```
Model Version: v20251110151552
Accuracy:  0.4935 (49.35%)
F1 Score:  0.3842 (38.42%)
ROC AUC:   0.5002 (50.02%)

Class Distribution:
  - Class 0 (No Heart Attack): 6,993 samples
  - Class 1 (Heart Attack):    3,007 samples

Confusion Matrix:
  [[671, 728],
   [285, 316]]
```

### Model Artifacts Saved
- `models/heart_attack_model.pkl` - Trained Logistic Regression model
- `models/scaler.pkl` - StandardScaler for feature normalization

## API Status

### Endpoints Tested ✅
- `GET /health` - Health check endpoint
- `POST /predict` - Prediction endpoint (single and multiple patients)
- Validation error handling

### Test Results
All 7 tests passed:
1. ✅ API imports correctly
2. ✅ Dataset mapping produces expected schema
3. ✅ Model loads without errors
4. ✅ Health endpoint responds
5. ✅ Single prediction works correctly
6. ✅ Multiple patient prediction works
7. ✅ Validation errors handled properly

## Quick Start

### Start the API Server
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### Test the API
```bash
python test_api.py
```

### Retrain the Model
```bash
python retrain_model.py
```

## Code Updates

### Files Modified
1. **backend/schemas.py** - Fixed Pydantic warnings by adding `model_config`
2. **README.md** - Updated with new dataset information and retraining instructions

### Files Created
1. **retrain_model.py** - Convenient script to retrain the model
2. **test_api.py** - Manual API testing script
3. **tests/test_predict_endpoint.py** - Comprehensive pytest-based API tests

## Notes

### Model Performance
The current model shows ~50% accuracy and ROC AUC, which indicates the synthetic/mapped features may need refinement. The mapping from the Indian dataset features to the standard cardiac risk features is heuristic and not clinically validated.

### Recommendations for Improvement
1. **Better Feature Engineering**: Review and refine the mapping from Indian dataset to standard features
2. **More Training Data**: Collect real clinical data with proper labels
3. **Model Selection**: Try other algorithms (Random Forest, XGBoost, Neural Networks)
4. **Hyperparameter Tuning**: Optimize model parameters
5. **Feature Selection**: Use feature importance analysis to identify key predictors

## Docker Deployment

The application is ready for Docker deployment:
```bash
docker compose up --build
```

This will start both the FastAPI backend (port 8000) and Streamlit frontend (port 8501).
