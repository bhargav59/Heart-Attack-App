# Using Real Indian Dataset Features (26 Features)

## 🎯 What Changed?

Instead of mapping the rich Indian dataset (26 features) to basic heart disease features (13 features), we now **use ALL original Indian features** including:

- ✅ **Smoking** (not available in old mapping)
- ✅ **Alcohol Consumption** (not available in old mapping)
- ✅ **Physical Activity** (not available in old mapping)
- ✅ **Diet Score** (not available in old mapping)
- ✅ **Stress Level** (not available in old mapping)
- ✅ **Air Pollution Exposure** (not available in old mapping)
- ✅ **Healthcare Access** (not available in old mapping)
- ✅ **Health Insurance** (not available in old mapping)
- ✅ **Emergency Response Time** (not available in old mapping)
- ✅ **Annual Income** (not available in old mapping)
- ✅ Plus 16 more features!

## 📊 Complete Feature List

### 1. Demographics (2 features)
- `Age` - Patient age (20-100 years)
- `Gender` - Male/Female

### 2. Medical Conditions (5 features)
- `Diabetes` - Has diabetes (0/1)
- `Hypertension` - Has high blood pressure (0/1)
- `Obesity` - Is obese (0/1)
- `Family_History` - Family history of heart disease (0/1)
- `Heart_Attack_History` - Previous heart attack (0/1)

### 3. Lifestyle Factors (4 features)
- `Smoking` - Current smoker (0/1)
- `Alcohol_Consumption` - Drinks alcohol (0/1)
- `Physical_Activity` - Physically active (0/1)
- `Diet_Score` - Diet quality (0-10 scale)

### 4. Blood Work (4 features)
- `Cholesterol_Level` - Total cholesterol (mg/dL)
- `Triglyceride_Level` - Triglycerides (mg/dL)
- `LDL_Level` - LDL "bad" cholesterol (mg/dL)
- `HDL_Level` - HDL "good" cholesterol (mg/dL)

### 5. Blood Pressure (2 features)
- `Systolic_BP` - Systolic blood pressure (mmHg)
- `Diastolic_BP` - Diastolic blood pressure (mmHg)

### 6. Environmental & Social (2 features)
- `Air_Pollution_Exposure` - Exposed to air pollution (0/1)
- `Stress_Level` - Stress level (1-10 scale)

### 7. Healthcare Access (2 features)
- `Healthcare_Access` - Has access to healthcare (0/1)
- `Health_Insurance` - Has health insurance (0/1)

### 8. Socioeconomic (2 features)
- `Emergency_Response_Time` - Emergency response time (minutes)
- `Annual_Income` - Annual income (INR)

**Total: 24 original features + engineered features**

## 🚀 New Training Script

### File: `train_with_indian_features.py`

This new script:
1. ✅ Uses ALL 26 Indian dataset features (no mapping!)
2. ✅ Applies SMOTE for class balancing
3. ✅ Creates powerful interaction features:
   - `age_x_cholesterol` - Age-cholesterol interaction
   - `lifestyle_score` - Composite lifestyle risk (smoking + alcohol + activity + diet)
   - `lipid_risk` - Blood lipid risk score
   - `bp_risk` - Blood pressure risk score
   - `total_risk_factors` - Count of all risk factors
   - `healthcare_quality` - Healthcare access quality score
   - And more!
4. ✅ Trains 6 algorithms with hyperparameter tuning
5. ✅ Builds ensemble model (voting classifier)
6. ✅ Targets 85%+ accuracy

### Run Training

```bash
# Install dependencies
pip install imbalanced-learn

# Run advanced training
python train_with_indian_features.py
```

**Expected output:**
```
🏆 BEST MODEL: Ensemble (or best single model)
   Accuracy: 85%+ (TARGET!)
   ROC AUC: 0.85+
```

## 🌐 New Frontend

### File: `app_indian.py`

Streamlit app with **all 26 features** organized into sections:
- Demographics
- Medical Conditions (checkboxes)
- Lifestyle Factors (checkboxes + diet slider)
- Blood Test Results (4 numeric inputs)
- Blood Pressure (2 numeric inputs)
- Environmental & Social (checkboxes + stress slider)
- Socioeconomic (income + emergency time)

### Run Frontend

```bash
streamlit run app_indian.py
```

## 📝 New API Schema

### File: `backend/schemas_indian.py`

Pydantic model `IndianHeartInput` with:
- All 26 fields with validation
- Proper ranges and descriptions
- Automatic Gender encoding (Male/Female → 1/0)
- `to_feature_dict()` method for model input

## 🔄 Migration Path

### Option 1: Use New System (Recommended)
```bash
# Train with new features
python train_with_indian_features.py

# Run new frontend
streamlit run app_indian.py
```

### Option 2: Keep Old System
```bash
# Old training (mapped features)
python improve_model.py

# Old frontend (13 features)
streamlit run app.py
```

## 📈 Expected Improvements

### Why This Should Achieve 85%+ Accuracy

1. **More Features = More Information**
   - Old: 13 mapped features
   - New: 24 original + ~10 engineered = 34 features
   - More data points → Better predictions

2. **Better Features**
   - **Smoking**: Major heart attack risk factor
   - **Alcohol**: Important lifestyle indicator
   - **Physical Activity**: Strong protective factor
   - **Diet Score**: Nutritional impact
   - **Stress Level**: Psychological risk
   - **Air Pollution**: Environmental risk
   - **Healthcare Access**: Treatment availability
   - **Income**: Socioeconomic health indicator

3. **No Information Loss**
   - Old system mapped 26 → 13 features (lost information!)
   - New system uses all 26 features directly

4. **Richer Interactions**
   - Lifestyle score combines 4 lifestyle factors
   - Lipid risk combines 4 blood markers
   - Total risk factors count all medical conditions

## 🎯 Feature Importance (Expected)

Based on medical knowledge, top features should be:

1. **Age** - #1 predictor
2. **Heart_Attack_History** - Previous = high risk
3. **Diabetes** - Major risk factor
4. **Hypertension** - Major risk factor
5. **Smoking** - Major risk factor
6. **Cholesterol_Level** - Blood marker
7. **Family_History** - Genetic risk
8. **Systolic_BP** - Blood pressure
9. **Obesity** - Major risk factor
10. **Physical_Activity** - Protective factor

## 📊 Training Tips

### For 85%+ Accuracy

1. **Use SMOTE** - Essential for imbalanced data
   ```python
   smote = SMOTE(sampling_strategy=0.8)  # 80% minority class
   ```

2. **Feature Engineering** - Already included:
   - Age interactions
   - Lifestyle composite scores
   - Blood work risk scores
   - Healthcare quality metrics

3. **Hyperparameter Tuning** - Already included:
   - GridSearchCV for top 3 models
   - 5-fold cross-validation
   - Extensive parameter ranges

4. **Ensemble** - Already included:
   - Voting classifier with top 3 models
   - Soft voting (weighted probabilities)

## 🔍 Troubleshooting

### If Accuracy < 85%

1. **Increase SMOTE ratio**
   ```python
   smote = SMOTE(sampling_strategy=0.9)  # Try 90%
   ```

2. **Add more interaction features**
   ```python
   X['smoking_x_age'] = X['Smoking'] * X['Age']
   X['diabetes_x_chol'] = X['Diabetes'] * X['Cholesterol_Level']
   ```

3. **Use more ensemble models**
   ```python
   ensemble = VotingClassifier([
       ('gb', gb_model),
       ('xgb', xgb_model),
       ('lgbm', lgbm_model),
       ('catboost', catboost_model),  # Add more
       ('rf', rf_model)
   ])
   ```

4. **Increase hyperparameter search**
   ```python
   param_grid = {
       'n_estimators': [200, 300, 500, 1000],  # More options
       'learning_rate': [0.01, 0.05, 0.1, 0.2],
       # ... more parameters
   }
   ```

## 📦 Files Summary

| File | Purpose |
|------|---------|
| `train_with_indian_features.py` | Main training script (all 26 features) |
| `app_indian.py` | Streamlit frontend (all 26 features) |
| `backend/schemas_indian.py` | API schemas (all 26 features) |
| `improve_model.py` | Old training (13 mapped features) ⚠️ |
| `app.py` | Old frontend (13 mapped features) ⚠️ |
| `backend/schemas.py` | Old API schemas (13 features) ⚠️ |

## 🎉 Ready to Train!

You now have:
- ✅ Complete training script with all Indian features
- ✅ Full-featured frontend with 26 inputs
- ✅ Proper API schemas
- ✅ Feature engineering
- ✅ SMOTE balancing
- ✅ Hyperparameter tuning
- ✅ Ensemble model

Run on your powerful machine:
```bash
git pull origin main
pip install imbalanced-learn
python train_with_indian_features.py
```

**Target: 85%+ accuracy with real Indian dataset features!** 🚀
