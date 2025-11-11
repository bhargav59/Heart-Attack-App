#!/usr/bin/env python3
"""
Model Improvement Script with Multiple Algorithms and EvalML
Tests various ML algorithms and uses AutoML to find the best model
"""
import sys
import warnings
from pathlib import Path
import time

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, 
    recall_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import joblib

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
from ml.train import map_indian_columns, FEATURES, TARGET
from backend.config import MODEL_DIR, MODEL_PATH, SCALER_PATH

warnings.filterwarnings('ignore')

def load_and_prepare_data(csv_path):
    """Load and prepare the dataset"""
    print(f"📁 Loading dataset from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Shape: {df.shape}")
    
    # Map Indian dataset columns
    if 'Heart_Attack_Risk' in df.columns:
        print("   Mapping Indian dataset columns...")
        df = map_indian_columns(df)
    
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)
    
    print(f"   Features: {X.shape[1]}")
    print(f"   Samples: {len(X)}")
    print(f"   Class distribution: {dict(y.value_counts())}")
    
    return X, y

def evaluate_model(name, model, X_train, X_test, y_train, y_test, scaler=None):
    """Train and evaluate a single model"""
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Scale data if scaler provided
    if scaler:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    # Train
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    metrics = {
        'model': name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'training_time': train_time
    }
    
    # Cross-validation score
    try:
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        metrics['cv_accuracy'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
    except:
        metrics['cv_accuracy'] = None
        metrics['cv_std'] = None
    
    print(f"\n✅ Results:")
    print(f"   Accuracy:       {metrics['accuracy']:.4f}")
    print(f"   Precision:      {metrics['precision']:.4f}")
    print(f"   Recall:         {metrics['recall']:.4f}")
    print(f"   F1 Score:       {metrics['f1_score']:.4f}")
    print(f"   ROC AUC:        {metrics['roc_auc']:.4f}")
    if metrics['cv_accuracy']:
        print(f"   CV Accuracy:    {metrics['cv_accuracy']:.4f} (±{metrics['cv_std']:.4f})")
    print(f"   Training Time:  {metrics['training_time']:.2f}s")
    
    return metrics, model, scaler

def compare_models(X_train, X_test, y_train, y_test):
    """Compare multiple ML algorithms"""
    print("\n" + "="*70)
    print("PHASE 1: Traditional ML Algorithms Comparison")
    print("="*70)
    
    models = [
        ('Logistic Regression', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42), True),
        ('Random Forest', RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1), False),
        ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42), True),
        ('XGBoost', XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric='logloss'), True),
        ('LightGBM', LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1), True),
        ('CatBoost', CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1, random_state=42, verbose=0), False),
    ]
    
    results = []
    trained_models = {}
    
    for name, model, use_scaler in models:
        scaler = StandardScaler() if use_scaler else None
        metrics, trained_model, fitted_scaler = evaluate_model(
            name, model, X_train, X_test, y_train, y_test, scaler
        )
        results.append(metrics)
        trained_models[name] = (trained_model, fitted_scaler)
    
    return pd.DataFrame(results), trained_models

def use_evalml_automl(X_train, X_test, y_train, y_test):
    """Use EvalML for automated machine learning"""
    print("\n" + "="*70)
    print("PHASE 2: EvalML AutoML Search")
    print("="*70)
    
    try:
        from evalml import AutoMLSearch
        from evalml.problem_types import ProblemTypes
        
        print("\n🔍 Running AutoML search...")
        print("   This may take several minutes...")
        
        automl = AutoMLSearch(
            X_train=X_train,
            y_train=y_train,
            problem_type=ProblemTypes.BINARY,
            max_iterations=10,  # Limit iterations for speed
            optimize_thresholds=True,
            max_time=300,  # 5 minutes max
            verbose=True
        )
        
        automl.search()
        
        print("\n✅ AutoML Search Complete!")
        print(f"   Best Pipeline: {automl.best_pipeline.name}")
        print(f"   Best Score: {automl.best_score:.4f}")
        
        # Evaluate best pipeline
        best_pipeline = automl.best_pipeline
        y_pred = best_pipeline.predict(X_test).to_series()
        y_pred_proba = best_pipeline.predict_proba(X_test).iloc[:, 1]
        
        evalml_metrics = {
            'model': 'EvalML Best',
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'cv_accuracy': automl.best_score,
            'cv_std': None,
            'training_time': None
        }
        
        print(f"\n📊 EvalML Best Model Results:")
        print(f"   Accuracy:    {evalml_metrics['accuracy']:.4f}")
        print(f"   Precision:   {evalml_metrics['precision']:.4f}")
        print(f"   Recall:      {evalml_metrics['recall']:.4f}")
        print(f"   F1 Score:    {evalml_metrics['f1_score']:.4f}")
        print(f"   ROC AUC:     {evalml_metrics['roc_auc']:.4f}")
        
        return evalml_metrics, best_pipeline
        
    except Exception as e:
        print(f"\n⚠️  EvalML failed: {e}")
        print("   Continuing with traditional models only...")
        return None, None

def main():
    # Load data
    dataset_path = "data/_kaggle_tmp/heart_attack_prediction_india.csv"
    X, y = load_and_prepare_data(dataset_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"\n📊 Train-Test Split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Testing:  {len(X_test)} samples")
    
    # Phase 1: Compare traditional models
    results_df, trained_models = compare_models(X_train, X_test, y_train, y_test)
    
    # Phase 2: EvalML AutoML
    evalml_metrics, evalml_pipeline = use_evalml_automl(X_train, X_test, y_train, y_test)
    if evalml_metrics:
        results_df = pd.concat([results_df, pd.DataFrame([evalml_metrics])], ignore_index=True)
    
    # Display comparison
    print("\n" + "="*70)
    print("FINAL COMPARISON: All Models")
    print("="*70)
    print("\n" + results_df.to_string(index=False))
    
    # Find best model
    best_idx = results_df['roc_auc'].idxmax()
    best_model_name = results_df.loc[best_idx, 'model']
    best_roc_auc = results_df.loc[best_idx, 'roc_auc']
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   ROC AUC: {best_roc_auc:.4f}")
    
    # Save best model
    print(f"\n💾 Saving best model...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    if best_model_name == 'EvalML Best' and evalml_pipeline:
        # Save EvalML pipeline
        evalml_pipeline.save(str(MODEL_DIR / "evalml_pipeline.pkl"))
        print(f"   Saved EvalML pipeline to: {MODEL_DIR}/evalml_pipeline.pkl")
        print(f"   ⚠️  Note: To use this model, update ml_service.py to load EvalML pipeline")
    else:
        # Save traditional model
        model, scaler = trained_models[best_model_name]
        joblib.dump(model, MODEL_PATH)
        if scaler:
            joblib.dump(scaler, SCALER_PATH)
        else:
            # Create identity scaler for models that don't need scaling
            scaler = StandardScaler()
            scaler.fit(X_train)
            joblib.dump(scaler, SCALER_PATH)
        
        print(f"   ✅ Model saved to: {MODEL_PATH}")
        print(f"   ✅ Scaler saved to: {SCALER_PATH}")
    
    print(f"\n{'='*70}")
    print("Model improvement complete!")
    print(f"{'='*70}")
    
    return results_df, best_model_name

if __name__ == "__main__":
    results, best_name = main()
