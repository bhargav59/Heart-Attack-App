# Advanced Model Training Guide

## 🎯 Goal
Achieve **85%+ accuracy** using advanced techniques including SMOTE, feature engineering, hyperparameter tuning, and ensemble methods.

## 📋 Prerequisites

### System Requirements
- **RAM**: At least 8GB (16GB recommended)
- **CPU**: Multi-core processor (4+ cores recommended)
- **Python**: 3.8+
- **Time**: ~30-60 minutes for full training

### Install Dependencies
```bash
pip install -r requirements.txt
pip install imbalanced-learn  # For SMOTE
```

## 🚀 Quick Start

### Option 1: Advanced Training (Recommended)
This script includes SMOTE, feature engineering, and extensive hyperparameter tuning:

```bash
python advanced_model_training.py
```

**What it does:**
- ✅ Loads 10,000 Indian heart attack records
- ✅ Applies SMOTE to balance classes
- ✅ Creates interaction features (age×cholesterol, BP×BMI, etc.)
- ✅ Polynomial features for non-linear relationships
- ✅ Hyperparameter tuning with GridSearchCV (5-fold CV)
- ✅ Tests 6 algorithms with optimized parameters
- ✅ Builds ensemble model (Voting Classifier)
- ✅ Saves best model automatically

**Expected Output:**
```
🏆 Best Model: [Algorithm Name]
   Accuracy: 85%+ (Target)
   ROC AUC: 0.85+
   F1 Score: 0.75+
```

### Option 2: Quick Improvement (Faster)
Basic algorithm comparison without extensive tuning:

```bash
python improve_model.py
```

## 📊 Training Steps Explained

### 1. Data Loading & Preparation
```python
# Load Indian dataset
df = pd.read_csv('data/_kaggle_tmp/heart_attack_prediction_india.csv')

# Map to standard features
mapped_df = map_indian_columns(df)
```

### 2. Handle Class Imbalance
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

### 3. Feature Engineering
```python
# Interaction features
age_chol = age * cholesterol
bp_bmi = blood_pressure * bmi

# Polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
```

### 4. Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    GradientBoostingClassifier(),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
```

### 5. Ensemble Learning
```python
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
    estimators=[
        ('gb', best_gb_model),
        ('xgb', best_xgb_model),
        ('lgbm', best_lgbm_model)
    ],
    voting='soft'
)
```

## 🔧 Configuration Options

### Modify SMOTE Parameters
In `advanced_model_training.py`, line ~80:
```python
smote = SMOTE(
    random_state=42,
    k_neighbors=5,  # Adjust neighbors
    sampling_strategy='auto'  # Or use ratio like 0.8
)
```

### Adjust Feature Engineering
In `advanced_model_training.py`, line ~120:
```python
poly = PolynomialFeatures(
    degree=2,  # Try 3 for more features (slower)
    interaction_only=False,  # Set True for interactions only
    include_bias=False
)
```

### Tune Hyperparameter Search Space
In `advanced_model_training.py`, line ~200:
```python
param_grid = {
    'n_estimators': [200, 500, 1000],  # More estimators
    'learning_rate': [0.01, 0.03, 0.05],  # Finer grid
    'max_depth': [5, 7, 10, 15],  # Deeper trees
    # Add more parameters...
}
```

## 📈 Monitoring Progress

### During Training
Watch for these outputs:
```
[INFO] Loading dataset...
[INFO] Applying SMOTE...
[INFO] Creating features...
[INFO] Training Gradient Boosting...
[INFO] Best params: {'learning_rate': 0.1, 'n_estimators': 500}
[INFO] Validation Accuracy: 87.3%
```

### Check Logs
```bash
# View training log
cat advanced_training_log.txt

# Monitor in real-time
tail -f advanced_training_log.txt
```

## 🎯 Expected Results

### Target Metrics
| Metric | Target | Current |
|--------|--------|---------|
| Accuracy | ≥85% | 69.4% |
| ROC AUC | ≥0.85 | 0.511 |
| F1 Score | ≥0.75 | 0.050 |
| Precision | ≥0.80 | 0.372 |
| Recall | ≥0.70 | 0.027 |

### Best Algorithms to Try
1. **Gradient Boosting** (currently best at 69.4%)
2. **XGBoost** with tuned parameters
3. **LightGBM** for speed + accuracy
4. **CatBoost** for categorical features
5. **Ensemble** of top 3 models

## 🐛 Troubleshooting

### Memory Issues
If you encounter memory errors:
```python
# Reduce GridSearchCV CV folds
cv=3  # Instead of cv=5

# Use RandomizedSearchCV instead
from sklearn.model_selection import RandomizedSearchCV
random_search = RandomizedSearchCV(
    model, param_distributions, n_iter=50, cv=3
)
```

### Slow Training
Speed up training:
```python
# Use fewer parameter combinations
param_grid = {
    'n_estimators': [100, 300],  # Reduce options
    'learning_rate': [0.05, 0.1]
}

# Enable parallel processing
n_jobs=-1  # Use all CPU cores
```

### Low Accuracy Still
If accuracy remains below 85%:

1. **Check data quality**:
   ```bash
   python -c "import pandas as pd; df = pd.read_csv('data/_kaggle_tmp/heart_attack_prediction_india.csv'); print(df.info()); print(df.describe())"
   ```

2. **Increase SMOTE ratio**:
   ```python
   smote = SMOTE(sampling_strategy=0.9)  # 90% minority
   ```

3. **Add more features**:
   ```python
   # Age groups
   df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 70, 100])
   
   # Risk scores
   df['risk_score'] = (df['age'] * 0.3 + df['cholesterol'] * 0.4 + df['bp'] * 0.3)
   ```

4. **Try ensemble with more models**:
   ```python
   ensemble = VotingClassifier([
       ('gb', gb_model),
       ('xgb', xgb_model),
       ('lgbm', lgbm_model),
       ('catboost', catboost_model),
       ('rf', rf_model)
   ])
   ```

## 📦 After Training

### Test the New Model
```bash
# Test with demo
python demo.py

# Run integration tests
python test_integration.py

# Test specific predictions
python -c "
from backend.ml_service import MLService
ml = MLService()
result = ml.predict({
    'age': 65, 'sex': 1, 'cp': 3, 'trestbps': 160,
    'chol': 300, 'fbs': 1, 'restecg': 1, 'thalach': 120,
    'exang': 1, 'oldpeak': 2.5, 'slope': 2, 'ca': 2, 'thal': 3
})
print(f'Risk: {result[\"risk_level\"]} - {result[\"risk_percent\"]}%')
"
```

### Deploy Updated Model
```bash
# Stop services
./stop_services.sh

# Restart with new model
./start_services.sh

# Verify
curl http://localhost:8000/health
```

### Commit Improved Model
```bash
git add models/heart_attack_model.pkl models/scaler.pkl
git commit -m "Update model with 85%+ accuracy"
git push origin main
```

## 📞 Support

If you need help:
1. Check `advanced_training_log.txt` for errors
2. Review model comparison results
3. Verify dataset is properly loaded
4. Ensure all dependencies are installed

## 🎉 Success Criteria

Your training is successful when you see:
```
🏆 Best Model: [Algorithm]
   Accuracy: 85.XX%  ✅
   ROC AUC: 0.85XX   ✅
   F1 Score: 0.75XX  ✅

💾 Model saved to: models/heart_attack_model.pkl
```

Good luck with your training! 🚀
