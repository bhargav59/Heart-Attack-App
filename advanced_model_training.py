"""
Advanced Model Training for Heart Attack Prediction
Target: Achieve 85%+ accuracy through:
1. SMOTE for class balancing
2. Advanced feature engineering
3. Comprehensive hyperparameter tuning
4. Ensemble methods
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              VotingClassifier, StackingClassifier, AdaBoostClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, classification_report, 
                            confusion_matrix)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("ADVANCED MODEL TRAINING FOR HEART ATTACK PREDICTION")
print("Target Accuracy: 85%+")
print("=" * 70)

# Load data
print("\n📊 Loading dataset...")
csv_path = "data/_kaggle_tmp/heart_attack_prediction_india.csv"
df = pd.read_csv(csv_path)
print(f"   ✅ Loaded {len(df)} records with {len(df.columns)} columns")

# Map Indian dataset columns to standard features
print("\n🔄 Mapping Indian dataset to standard features...")
from ml.train import map_indian_columns
df_mapped = map_indian_columns(df)
print(f"   ✅ Mapped to {len(df_mapped.columns)} standard features")

# Separate features and target
X = df_mapped.drop('target', axis=1)
y = df_mapped['target']

print(f"\n📈 Class distribution:")
print(f"   High Risk (1): {sum(y == 1)} ({sum(y == 1)/len(y)*100:.1f}%)")
print(f"   Low Risk (0):  {sum(y == 0)} ({sum(y == 0)/len(y)*100:.1f}%)")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✂️  Data split:")
print(f"   Training:   {len(X_train)} samples")
print(f"   Testing:    {len(X_test)} samples")

# ============================================================================
# STEP 1: FEATURE ENGINEERING
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1: ADVANCED FEATURE ENGINEERING")
print("=" * 70)

def create_advanced_features(X_df):
    """Create interaction and domain-specific features"""
    X_new = X_df.copy()
    
    # Interaction features
    X_new['age_chol'] = X_new['age'] * X_new['chol']
    X_new['age_trestbps'] = X_new['age'] * X_new['trestbps']
    X_new['chol_trestbps'] = X_new['chol'] * X_new['trestbps']
    
    # Risk scores
    X_new['cardiovascular_risk'] = (
        (X_new['age'] > 60).astype(int) +
        (X_new['chol'] > 240).astype(int) +
        (X_new['trestbps'] > 140).astype(int) +
        (X_new['thalach'] < 100).astype(int)
    )
    
    # Age groups
    X_new['age_group'] = pd.cut(X_new['age'], bins=[0, 40, 55, 70, 100], labels=[0, 1, 2, 3])
    X_new['age_group'] = X_new['age_group'].astype(int)
    
    # Cholesterol categories
    X_new['chol_category'] = pd.cut(X_new['chol'], bins=[0, 200, 240, 999], labels=[0, 1, 2])
    X_new['chol_category'] = X_new['chol_category'].astype(int)
    
    # Blood pressure categories
    X_new['bp_category'] = pd.cut(X_new['trestbps'], bins=[0, 120, 140, 999], labels=[0, 1, 2])
    X_new['bp_category'] = X_new['bp_category'].astype(int)
    
    return X_new

X_train_eng = create_advanced_features(X_train)
X_test_eng = create_advanced_features(X_test)

print(f"\n✅ Feature engineering complete:")
print(f"   Original features: {X_train.shape[1]}")
print(f"   Engineered features: {X_train_eng.shape[1]}")
print(f"   New features added: {X_train_eng.shape[1] - X_train.shape[1]}")

# ============================================================================
# STEP 2: CLASS BALANCING WITH SMOTE
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: ADDRESSING CLASS IMBALANCE WITH SMOTE")
print("=" * 70)

# Apply SMOTE
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_eng, y_train)

print(f"\n✅ SMOTE applied:")
print(f"   Before SMOTE:")
print(f"      High Risk (1): {sum(y_train == 1)} samples")
print(f"      Low Risk (0):  {sum(y_train == 0)} samples")
print(f"   After SMOTE:")
print(f"      High Risk (1): {sum(y_train_balanced == 1)} samples")
print(f"      Low Risk (0):  {sum(y_train_balanced == 0)} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test_eng)

print(f"\n✅ Features scaled with StandardScaler")

# ============================================================================
# STEP 3: HYPERPARAMETER TUNING
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: COMPREHENSIVE HYPERPARAMETER TUNING")
print("=" * 70)

results = {}

# --- Random Forest Tuning ---
print("\n🔧 Tuning Random Forest...")
rf_params = {
    'n_estimators': [200, 300, 500],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced', 'balanced_subsample']
}

rf_random = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params,
    n_iter=20,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42,
    verbose=1
)
rf_random.fit(X_train_scaled, y_train_balanced)
best_rf = rf_random.best_estimator_

y_pred_rf = best_rf.predict(X_test_scaled)
results['Random Forest (Tuned)'] = {
    'model': best_rf,
    'accuracy': accuracy_score(y_test, y_pred_rf),
    'precision': precision_score(y_test, y_pred_rf, zero_division=0),
    'recall': recall_score(y_test, y_pred_rf),
    'f1': f1_score(y_test, y_pred_rf),
    'roc_auc': roc_auc_score(y_test, best_rf.predict_proba(X_test_scaled)[:, 1]),
    'params': rf_random.best_params_
}

print(f"   ✅ Best ROC AUC: {rf_random.best_score_:.4f}")
print(f"   ✅ Test Accuracy: {results['Random Forest (Tuned)']['accuracy']:.4f}")

# --- Gradient Boosting Tuning ---
print("\n🔧 Tuning Gradient Boosting...")
gb_params = {
    'n_estimators': [200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0]
}

gb_random = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    gb_params,
    n_iter=20,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42,
    verbose=1
)
gb_random.fit(X_train_scaled, y_train_balanced)
best_gb = gb_random.best_estimator_

y_pred_gb = best_gb.predict(X_test_scaled)
results['Gradient Boosting (Tuned)'] = {
    'model': best_gb,
    'accuracy': accuracy_score(y_test, y_pred_gb),
    'precision': precision_score(y_test, y_pred_gb, zero_division=0),
    'recall': recall_score(y_test, y_pred_gb),
    'f1': f1_score(y_test, y_pred_gb),
    'roc_auc': roc_auc_score(y_test, best_gb.predict_proba(X_test_scaled)[:, 1]),
    'params': gb_random.best_params_
}

print(f"   ✅ Best ROC AUC: {gb_random.best_score_:.4f}")
print(f"   ✅ Test Accuracy: {results['Gradient Boosting (Tuned)']['accuracy']:.4f}")

# --- XGBoost Tuning ---
print("\n🔧 Tuning XGBoost...")
xgb_params = {
    'n_estimators': [200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7, 10],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'scale_pos_weight': [1, 2, 3]
}

xgb_random = RandomizedSearchCV(
    xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
    xgb_params,
    n_iter=20,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42,
    verbose=1
)
xgb_random.fit(X_train_scaled, y_train_balanced)
best_xgb = xgb_random.best_estimator_

y_pred_xgb = best_xgb.predict(X_test_scaled)
results['XGBoost (Tuned)'] = {
    'model': best_xgb,
    'accuracy': accuracy_score(y_test, y_pred_xgb),
    'precision': precision_score(y_test, y_pred_xgb, zero_division=0),
    'recall': recall_score(y_test, y_pred_xgb),
    'f1': f1_score(y_test, y_pred_xgb),
    'roc_auc': roc_auc_score(y_test, best_xgb.predict_proba(X_test_scaled)[:, 1]),
    'params': xgb_random.best_params_
}

print(f"   ✅ Best ROC AUC: {xgb_random.best_score_:.4f}")
print(f"   ✅ Test Accuracy: {results['XGBoost (Tuned)']['accuracy']:.4f}")

# --- LightGBM Tuning ---
print("\n🔧 Tuning LightGBM...")
lgb_params = {
    'n_estimators': [200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [31, 50, 70],
    'max_depth': [5, 7, 10, -1],
    'min_child_samples': [20, 30, 50],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'class_weight': ['balanced', None]
}

lgb_random = RandomizedSearchCV(
    lgb.LGBMClassifier(random_state=42, verbose=-1),
    lgb_params,
    n_iter=20,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42,
    verbose=1
)
lgb_random.fit(X_train_scaled, y_train_balanced)
best_lgb = lgb_random.best_estimator_

y_pred_lgb = best_lgb.predict(X_test_scaled)
results['LightGBM (Tuned)'] = {
    'model': best_lgb,
    'accuracy': accuracy_score(y_test, y_pred_lgb),
    'precision': precision_score(y_test, y_pred_lgb, zero_division=0),
    'recall': recall_score(y_test, y_pred_lgb),
    'f1': f1_score(y_test, y_pred_lgb),
    'roc_auc': roc_auc_score(y_test, best_lgb.predict_proba(X_test_scaled)[:, 1]),
    'params': lgb_random.best_params_
}

print(f"   ✅ Best ROC AUC: {lgb_random.best_score_:.4f}")
print(f"   ✅ Test Accuracy: {results['LightGBM (Tuned)']['accuracy']:.4f}")

# ============================================================================
# STEP 4: ENSEMBLE METHODS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: BUILDING ENSEMBLE MODELS")
print("=" * 70)

# Voting Classifier (Soft Voting)
print("\n🔧 Building Voting Classifier...")
voting_clf = VotingClassifier(
    estimators=[
        ('rf', best_rf),
        ('gb', best_gb),
        ('xgb', best_xgb),
        ('lgb', best_lgb)
    ],
    voting='soft',
    n_jobs=-1
)
voting_clf.fit(X_train_scaled, y_train_balanced)

y_pred_voting = voting_clf.predict(X_test_scaled)
results['Voting Ensemble'] = {
    'model': voting_clf,
    'accuracy': accuracy_score(y_test, y_pred_voting),
    'precision': precision_score(y_test, y_pred_voting, zero_division=0),
    'recall': recall_score(y_test, y_pred_voting),
    'f1': f1_score(y_test, y_pred_voting),
    'roc_auc': roc_auc_score(y_test, voting_clf.predict_proba(X_test_scaled)[:, 1])
}

print(f"   ✅ Voting Ensemble Accuracy: {results['Voting Ensemble']['accuracy']:.4f}")

# Stacking Classifier
print("\n🔧 Building Stacking Classifier...")
stacking_clf = StackingClassifier(
    estimators=[
        ('rf', best_rf),
        ('gb', best_gb),
        ('xgb', best_xgb),
        ('lgb', best_lgb)
    ],
    final_estimator=LogisticRegression(max_iter=1000, class_weight='balanced'),
    cv=5,
    n_jobs=-1
)
stacking_clf.fit(X_train_scaled, y_train_balanced)

y_pred_stacking = stacking_clf.predict(X_test_scaled)
results['Stacking Ensemble'] = {
    'model': stacking_clf,
    'accuracy': accuracy_score(y_test, y_pred_stacking),
    'precision': precision_score(y_test, y_pred_stacking, zero_division=0),
    'recall': recall_score(y_test, y_pred_stacking),
    'f1': f1_score(y_test, y_pred_stacking),
    'roc_auc': roc_auc_score(y_test, stacking_clf.predict_proba(X_test_scaled)[:, 1])
}

print(f"   ✅ Stacking Ensemble Accuracy: {results['Stacking Ensemble']['accuracy']:.4f}")

# ============================================================================
# FINAL RESULTS
# ============================================================================
print("\n" + "=" * 70)
print("FINAL RESULTS: ALL MODELS")
print("=" * 70)

results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [r['accuracy'] for r in results.values()],
    'Precision': [r['precision'] for r in results.values()],
    'Recall': [r['recall'] for r in results.values()],
    'F1 Score': [r['f1'] for r in results.values()],
    'ROC AUC': [r['roc_auc'] for r in results.values()]
}).sort_values('Accuracy', ascending=False)

print("\n" + results_df.to_string(index=False))

# Find best model
best_model_name = results_df.iloc[0]['Model']
best_accuracy = results_df.iloc[0]['Accuracy']

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   Accuracy:  {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print(f"   Precision: {results_df.iloc[0]['Precision']:.4f}")
print(f"   Recall:    {results_df.iloc[0]['Recall']:.4f}")
print(f"   F1 Score:  {results_df.iloc[0]['F1 Score']:.4f}")
print(f"   ROC AUC:   {results_df.iloc[0]['ROC AUC']:.4f}")

# Check if we achieved 85% accuracy
if best_accuracy >= 0.85:
    print(f"\n✅ SUCCESS! Achieved target accuracy of 85%+")
else:
    print(f"\n⚠️  Warning: Target accuracy 85% not reached (got {best_accuracy*100:.2f}%)")
    print(f"   This may be due to dataset limitations (synthetic features)")

# Save best model
best_model = results[best_model_name]['model']

print(f"\n💾 Saving best model...")
with open('models/heart_attack_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print(f"   ✅ Model saved to: models/heart_attack_model.pkl")

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print(f"   ✅ Scaler saved to: models/scaler.pkl")

# Save feature engineering function
with open('models/feature_engineer.pkl', 'wb') as f:
    pickle.dump(create_advanced_features, f)
print(f"   ✅ Feature engineering function saved to: models/feature_engineer.pkl")

# Save results
results_df.to_csv('advanced_model_results.csv', index=False)
print(f"   ✅ Results saved to: advanced_model_results.csv")

# Detailed classification report
print(f"\n" + "=" * 70)
print(f"DETAILED CLASSIFICATION REPORT - {best_model_name}")
print("=" * 70)
y_pred_best = best_model.predict(X_test_scaled)
print(classification_report(y_test, y_pred_best, target_names=['Low Risk', 'High Risk']))

print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_best)
print(cm)
print(f"   True Negatives:  {cm[0, 0]}")
print(f"   False Positives: {cm[0, 1]}")
print(f"   False Negatives: {cm[1, 0]}")
print(f"   True Positives:  {cm[1, 1]}")

print("\n" + "=" * 70)
print("ADVANCED MODEL TRAINING COMPLETE!")
print("=" * 70)
