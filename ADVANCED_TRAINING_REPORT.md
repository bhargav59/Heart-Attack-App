# Advanced Model Training Results

## Overview

Date: November 11, 2025  
Training Script: `advanced_model_training.py`  
Dataset: Indian Heart Attack Prediction (10,000 records, 26 features)  
Target: 85%+ accuracy  
**Achieved: 65.60% accuracy with LightGBM**

## Training Pipeline

### 1. Feature Engineering

- **Original features**: 13 (mapped from Indian dataset)
- **Engineered features**: 22 total
- **New features added**: 9
  - Interaction features: `age_chol`, `age_trtbps`, `chol_trtbps`, `age_thalachh`
  - Risk composite: `cardiovascular_risk`
  - Categorical binning: `age_group`, `chol_category`, `bp_category`, `hr_category`

### 2. Class Imbalance Handling

- **Before SMOTE**: 30.1% high risk, 69.9% low risk
- **After SMOTE**: Balanced 50/50 split (5,594 samples each class)
- **Normalization**: StandardScaler applied to all features

### 3. Hyperparameter Tuning

Performed RandomizedSearchCV (20 iterations, 5-fold CV) on:

- **Random Forest**: CV ROC 0.8026, Test Acc 0.6335
- **Gradient Boosting**: CV ROC 0.8181, Test Acc 0.6510
- **XGBoost**: CV ROC 0.8266, Test Acc 0.6115
- **LightGBM**: CV ROC 0.8101, Test Acc 0.6560 ✅

### 4. Ensemble Methods

- **Voting Classifier** (soft voting): 0.6480 accuracy
- **Stacking Classifier** (LogReg meta): 0.6385 accuracy
- Neither ensemble improved over single LightGBM

## Final Results

### Best Model: LightGBM (Tuned)

```
Accuracy:  65.60%
Precision: 0.3619 (high risk class)
Recall:    0.1897 (high risk class)
F1 Score:  0.2489
ROC AUC:   0.4889 (barely above random)
```

### Confusion Matrix

```
              Predicted
              Low   High
Actual Low   1198   201
       High   487   114

True Negatives:  1198 (correctly identified low risk)
False Positives: 201  (predicted high, actually low)
False Negatives: 487  (predicted low, actually high) ⚠️
True Positives:  114  (correctly identified high risk)
```

### Classification Report

```
              precision    recall  f1-score   support
    Low Risk       0.71      0.86      0.78      1399
   High Risk       0.36      0.19      0.25       601
    accuracy                           0.66      2000
   macro avg       0.54      0.52      0.51      2000
weighted avg       0.61      0.66      0.62      2000
```

## Key Findings

### 1. Poor High-Risk Detection (19% Recall)

- Model misses **81% of actual high-risk cases** (487 false negatives vs 114 true positives)
- This is **clinically dangerous** - high-risk patients incorrectly cleared as safe
- Likely due to severe class imbalance in test set (70/30) despite SMOTE balancing training

### 2. Low Confidence (36% Precision for High Risk)

- When model predicts high risk, it's only correct 36% of the time
- Suggests weak signal in features or noisy labels

### 3. ROC AUC Near Random (0.49)

- Model barely better than coin flip for ranking predictions
- Large gap between CV ROC (0.80+) and test ROC (0.49) indicates **overfitting**

### 4. Train-Test Performance Gap

- CV scores during tuning: 0.80-0.83 ROC AUC
- Test scores: 0.49 ROC AUC, 0.66 accuracy
- Suggests:
  - Data leakage during cross-validation (SMOTE applied before CV split?)
  - Synthetic features in Indian dataset don't generalize
  - Test set distribution differs from training

## Root Causes

### 1. Heuristic Feature Mapping

The `map_indian_columns()` function uses proxy mappings:

- `Stress_Level → cp` (chest pain type)
- `Physical_Activity → exng` (exercise-induced angina)
- `HDL_Level → thall` (thallium scan result)
- These proxies **don't capture true clinical relationships**

### 2. Synthetic/Weak Target Labels

- Indian dataset's `Heart_Attack_Risk` column may be synthetically generated
- Low correlation with actual outcomes
- Explains why model learns spurious patterns that don't generalize

### 3. Feature Engineering Didn't Help

- 9 additional engineered features added no predictive power
- Interaction terms (age×chol, etc.) and categorical bins didn't improve metrics
- Suggests underlying signal is too weak

### 4. SMOTE May Have Introduced Artifacts

- Synthetic minority samples created from k-nearest neighbors
- May have generated unrealistic feature combinations
- Explains CV-test performance gap

## Model Artifacts Saved

1. **models/heart_attack_model.pkl** (2.4 MB)
   - LightGBM classifier with 22-feature input
2. **models/scaler.pkl** (1.3 KB)
   - StandardScaler fitted on SMOTE-balanced training data
3. **models/feature_engineer.pkl** (52 B)
   - Pickled `create_advanced_features` function (deprecated - use ml/feature_engineering.py)
4. **ml/feature_engineering.py** (new)
   - Importable feature engineering module
   - Used automatically by MLService when advanced model detected
5. **advanced_model_results.csv**
   - Comparison table of all 6 models trained

## Integration with App

### Backend Updates

- **backend/ml_service.py**: Now detects advanced models (22 features) and applies feature engineering automatically
- **backend/config.py**: Added `ROOT_DIR` alias for clarity
- **backend/main.py**: Fixed probability interpretation (class 1 = high risk, not class 0)

### Model Version

- Updated `MODEL_VERSION = "v2_advanced"`
- Frontend and `/predict` endpoint now use advanced LightGBM model

### Backward Compatibility

- Service gracefully falls back to 13-feature models if no feature engineering available
- Legacy LogisticRegression model still supported

## Recommendations

### Immediate Actions

1. **Do NOT use this model in production** - 19% recall for high-risk cases is unacceptable
2. Document model limitations clearly in UI disclaimers
3. Consider reverting to simpler LogisticRegression baseline until better data available

### Long-Term Improvements

1. **Acquire Real Clinical Data**

   - Replace heuristic mappings with actual medical records
   - Validate target labels against diagnoses
   - Ensure data collection protocol matches intended use case

2. **Feature Selection & Domain Expertise**

   - Consult cardiologists to identify truly predictive features
   - Remove noise and synthetic proxies
   - Focus on evidence-based risk factors (troponin, ECG, family history)

3. **Optimize for Recall (Minimize False Negatives)**

   - Use class weights to penalize missing high-risk cases
   - Adjust decision threshold to favor sensitivity over specificity
   - Evaluate using PR-AUC instead of ROC-AUC for imbalanced data

4. **Calibration & Confidence**

   - Apply Platt scaling or isotonic regression to calibrate probabilities
   - Add prediction confidence intervals
   - Implement model monitoring to detect drift

5. **Alternative Approaches**
   - Try semi-supervised learning with medical literature knowledge
   - Use transfer learning from validated cardiac models
   - Ensemble with rule-based clinical scoring (Framingham, ASCVD)

## Conclusion

While the training pipeline successfully demonstrated:

- ✅ Advanced feature engineering
- ✅ SMOTE for class balancing
- ✅ Comprehensive hyperparameter tuning
- ✅ Ensemble methods

The **65.6% accuracy falls far short of the 85% target** due to:

- ❌ Weak signal in heuristically-mapped features
- ❌ Synthetic/noisy target labels
- ❌ Severe class imbalance causing poor recall
- ❌ Overfitting (CV vs test performance gap)

**The model is not ready for clinical use.** Further work requires acquiring high-quality, validated medical data and clinical domain expertise.
