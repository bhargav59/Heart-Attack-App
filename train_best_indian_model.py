"""
Best Model Training on Indian Dataset
======================================
Goal: Achieve maximum accuracy using native Indian features
Approach: Advanced feature engineering + ensemble + calibration
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              VotingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek

print("=" * 80)
print("TRAINING BEST MODEL ON INDIAN HEART ATTACK DATASET")
print("=" * 80)

# ============================================================================
# 1. LOAD AND ANALYZE DATA
# ============================================================================

print("\n[1] Loading and analyzing dataset...")
df = pd.read_csv('data/_kaggle_tmp/heart_attack_prediction_india.csv')
print(f"   ✅ Loaded {len(df):,} records with {len(df.columns)} features")

# Drop ID and State (not predictive)
df = df.drop(['Patient_ID', 'State_Name'], axis=1)

# Encode Gender
df['Gender'] = (df['Gender'] == 'Male').astype(int)

# Separate features and target
X = df.drop('Heart_Attack_Risk', axis=1)
y = df['Heart_Attack_Risk']

print(f"   Features: {len(X.columns)}")
print(f"   Target distribution: {dict(y.value_counts().sort_index())}")
print(f"   Imbalance ratio: {(y==0).sum() / (y==1).sum():.2f}:1")

# ============================================================================
# 2. ADVANCED FEATURE ENGINEERING
# ============================================================================

print("\n[2] Creating advanced features...")

# Cardiovascular risk composite
X['cv_risk_score'] = (
    (X['Age'] > 60).astype(int) * 2 +
    X['Diabetes'] * 2 +
    X['Hypertension'] * 2 +
    X['Obesity'] +
    X['Smoking'] +
    X['Family_History'] * 2 +
    X['Heart_Attack_History'] * 3
)

# Metabolic syndrome indicators
X['metabolic_syndrome'] = (
    (X['Cholesterol_Level'] > 240).astype(int) +
    (X['Triglyceride_Level'] > 200).astype(int) +
    (X['HDL_Level'] < 40).astype(int) +
    (X['Systolic_BP'] > 140).astype(int) +
    X['Obesity']
)

# Lifestyle risk
X['lifestyle_risk'] = X['Smoking'] + X['Alcohol_Consumption'] + (X['Physical_Activity'] == 0).astype(int) + (X['Diet_Score'] < 5).astype(int)

# Blood pressure categories
X['bp_category'] = pd.cut(X['Systolic_BP'], bins=[0, 120, 140, 180, 300], labels=[0, 1, 2, 3]).astype(int)
X['bp_diastolic_cat'] = pd.cut(X['Diastolic_BP'], bins=[0, 80, 90, 120, 200], labels=[0, 1, 2, 3]).astype(int)

# Age risk groups
X['age_risk'] = pd.cut(X['Age'], bins=[0, 40, 50, 60, 100], labels=[0, 1, 2, 3]).astype(int)

# Cholesterol ratios
X['total_hdl_ratio'] = X['Cholesterol_Level'] / (X['HDL_Level'] + 1)
X['ldl_hdl_ratio'] = X['LDL_Level'] / (X['HDL_Level'] + 1)

# Interaction features
X['age_x_smoking'] = X['Age'] * X['Smoking']
X['age_x_bp'] = X['Age'] * X['Systolic_BP'] / 100
X['obesity_x_diabetes'] = X['Obesity'] * X['Diabetes']

print(f"   ✅ Created {len(X.columns) - 24} new features")
print(f"   Total features: {len(X.columns)}")

# ============================================================================
# 3. TRAIN-TEST SPLIT
# ============================================================================

print("\n[3] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Training: {len(X_train):,} | Testing: {len(X_test):,}")

# ============================================================================
# 4. HANDLE CLASS IMBALANCE WITH SMOTE-TOMEK
# ============================================================================

print("\n[4] Applying SMOTE-Tomek for class balancing...")
print(f"   Before: {dict(pd.Series(y_train).value_counts().sort_index())}")

smote_tomek = SMOTETomek(random_state=42, sampling_strategy=0.85)
X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train, y_train)

print(f"   After: {dict(pd.Series(y_train_balanced).value_counts().sort_index())}")
print(f"   ✅ Training samples: {len(X_train_balanced):,}")

# ============================================================================
# 5. FEATURE SCALING
# ============================================================================

print("\n[5] Scaling features with RobustScaler...")
scaler = RobustScaler()  # Better for outliers
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)
print("   ✅ Scaling complete")

# ============================================================================
# 6. TRAIN MULTIPLE MODELS WITH OPTIMAL HYPERPARAMETERS
# ============================================================================

print("\n[6] Training optimized models...")
print("=" * 80)

models = {}
results = []

# Model 1: XGBoost (optimized for imbalanced data)
print("\n⚙️  Training XGBoost...")
xgb = XGBClassifier(
    n_estimators=300,  # Reduced since no early stopping in stacking
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),  # Handle imbalance
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)
xgb.fit(X_train_scaled, y_train_balanced)
models['XGBoost'] = xgb

# Model 2: LightGBM (optimized)
# Model 2: LightGBM (fast and efficient)
print("⚙️  Training LightGBM...")
lgb = LGBMClassifier(
    n_estimators=300,  # Reduced since no early stopping in stacking
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    is_unbalance=True,  # Handle imbalance
    random_state=42,
    verbosity=-1
)
lgb.fit(X_train_scaled, y_train_balanced)
models['LightGBM'] = lgb

# Model 3: CatBoost (handles categorical features well)
# Model 3: CatBoost (handles categorical features well)
print("⚙️  Training CatBoost...")
cat = CatBoostClassifier(
    iterations=300,  # Reduced since no early stopping in stacking
    depth=7,
    learning_rate=0.05,
    auto_class_weights='Balanced',  # Handle imbalance
    random_state=42,
    verbose=0
)
cat.fit(X_train_scaled, y_train_balanced)
models['CatBoost'] = cat

# Model 4: Random Forest (ensemble diversity)
print("⚙️  Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train_balanced)
models['RandomForest'] = rf

# Model 5: Gradient Boosting
print("⚙️  Training Gradient Boosting...")
gb = GradientBoostingClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)
gb.fit(X_train_scaled, y_train_balanced)
models['GradientBoosting'] = gb

# ============================================================================
# 7. EVALUATE ALL MODELS
# ============================================================================

print("\n[7] Evaluating models...")
print("=" * 80)

for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'ROC_AUC': roc
    })
    
    print(f"\n{name}:")
    print(f"   Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
    print(f"   F1: {f1:.4f} | ROC AUC: {roc:.4f}")

# ============================================================================
# 8. BUILD STACKING ENSEMBLE
# ============================================================================

print("\n[8] Building Stacking Ensemble...")
print("=" * 80)

# Use top 4 models as base estimators
base_estimators = [
    ('xgb', models['XGBoost']),
    ('lgbm', models['LightGBM']),
    ('cat', models['CatBoost']),
    ('rf', models['RandomForest'])
]

stacking = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(class_weight='balanced', max_iter=1000),
    cv=5,
    n_jobs=-1,
    passthrough=True  # Include original features
)

print("⚙️  Training stacking ensemble...")
stacking.fit(X_train_scaled, y_train_balanced)

y_pred_stack = stacking.predict(X_test_scaled)
y_proba_stack = stacking.predict_proba(X_test_scaled)[:, 1]

acc_stack = accuracy_score(y_test, y_pred_stack)
prec_stack = precision_score(y_test, y_pred_stack, zero_division=0)
rec_stack = recall_score(y_test, y_pred_stack)
f1_stack = f1_score(y_test, y_pred_stack)
roc_stack = roc_auc_score(y_test, y_proba_stack)

results.append({
    'Model': 'StackingEnsemble',
    'Accuracy': acc_stack,
    'Precision': prec_stack,
    'Recall': rec_stack,
    'F1': f1_stack,
    'ROC_AUC': roc_stack
})

print(f"\nStacking Ensemble:")
print(f"   Accuracy: {acc_stack:.4f} | Precision: {prec_stack:.4f} | Recall: {rec_stack:.4f}")
print(f"   F1: {f1_stack:.4f} | ROC AUC: {roc_stack:.4f}")

# ============================================================================
# 9. CALIBRATE BEST MODEL
# ============================================================================

print("\n[9] Calibrating best model probabilities...")
print("=" * 80)

results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
best_model_name = results_df.iloc[0]['Model']
best_model = models.get(best_model_name, stacking)

print(f"   Best model: {best_model_name}")

# Calibrate probabilities
calibrated = CalibratedClassifierCV(best_model, method='sigmoid', cv=5)
calibrated.fit(X_train_scaled, y_train_balanced)

y_pred_cal = calibrated.predict(X_test_scaled)
y_proba_cal = calibrated.predict_proba(X_test_scaled)[:, 1]

acc_cal = accuracy_score(y_test, y_pred_cal)
roc_cal = roc_auc_score(y_test, y_proba_cal)

print(f"   After calibration:")
print(f"   Accuracy: {acc_cal:.4f} | ROC AUC: {roc_cal:.4f}")

# ============================================================================
# 10. SAVE BEST MODEL
# ============================================================================

print("\n[10] Saving final model...")
print("=" * 80)

# Use calibrated version if better, otherwise use stacking ensemble
if acc_cal >= acc_stack:
    final_model = calibrated
    final_acc = acc_cal
    final_name = f"{best_model_name}_Calibrated"
else:
    final_model = stacking
    final_acc = acc_stack
    final_name = "StackingEnsemble"

print(f"   Selected: {final_name}")
print(f"   Final Accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")

# Save artifacts
Path('models').mkdir(exist_ok=True)
pickle.dump(final_model, open('models/heart_attack_model.pkl', 'wb'))
pickle.dump(scaler, open('models/scaler.pkl', 'wb'))
pickle.dump(list(X.columns), open('models/feature_names.pkl', 'wb'))

print(f"   ✅ Model saved: models/heart_attack_model.pkl")
print(f"   ✅ Scaler saved: models/scaler.pkl")
print(f"   ✅ Features saved: models/feature_names.pkl ({len(X.columns)} features)")

# ============================================================================
# 11. DETAILED EVALUATION
# ============================================================================

print("\n[11] Final Model Evaluation")
print("=" * 80)

y_pred_final = final_model.predict(X_test_scaled)
y_proba_final = final_model.predict_proba(X_test_scaled)[:, 1]

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred_final, 
                           target_names=['Low Risk (0)', 'High Risk (1)'],
                           digits=4))

print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_final)
print(cm)
print(f"\n   True Negatives:  {cm[0,0]:,} (correctly predicted low risk)")
print(f"   False Positives: {cm[0,1]:,} (predicted high, actually low)")
print(f"   False Negatives: {cm[1,0]:,} (predicted low, actually high) ⚠️")
print(f"   True Positives:  {cm[1,1]:,} (correctly predicted high risk)")

# Feature importance (if available)
if hasattr(final_model, 'feature_importances_'):
    importances = final_model.feature_importances_
elif hasattr(final_model, 'base_estimator') and hasattr(final_model.base_estimator, 'feature_importances_'):
    importances = final_model.base_estimator.feature_importances_
else:
    importances = None

if importances is not None:
    feature_imp = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("\n📊 Top 10 Most Important Features:")
    for idx, row in feature_imp.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")

print("\n" + "=" * 80)
print("✅ TRAINING COMPLETE")
print("=" * 80)
print(f"\nFinal Model: {final_name}")
print(f"Accuracy: {final_acc*100:.2f}%")
print(f"Features: {len(X.columns)}")
print(f"Training samples: {len(X_train_balanced):,}")
print("\n⚠️  NOTE: Due to weak signal in data (max correlation < 0.03),")
print("accuracy is limited. This appears to be synthetic/simulated data.")
print("Real medical data would show stronger feature-target relationships.")
