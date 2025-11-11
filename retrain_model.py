#!/usr/bin/env python3
"""
Script to retrain the heart attack prediction model with the Indian dataset.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ml.train import train_on_csv

def main():
    dataset_path = "data/_kaggle_tmp/heart_attack_prediction_india.csv"
    print(f"🚀 Training model with Indian dataset: {dataset_path}")
    print("-" * 60)
    
    try:
        metrics, features, model_version, class_dist, cm = train_on_csv(dataset_path)
        
        print("\n✅ Training completed successfully!")
        print(f"\n📊 Model Version: {model_version}")
        print(f"\n📈 Metrics:")
        print(f"  - Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  - F1 Score:  {metrics['f1']:.4f}")
        print(f"  - ROC AUC:   {metrics['roc_auc']:.4f}" if metrics['roc_auc'] else "  - ROC AUC:   N/A")
        
        print(f"\n🎯 Class Distribution:")
        for cls, count in class_dist.items():
            print(f"  - Class {cls}: {count} samples")
        
        print(f"\n📋 Confusion Matrix:")
        print(f"  [[{cm[0][0]}, {cm[0][1]}],")
        print(f"   [{cm[1][0]}, {cm[1][1]}]]")
        
        print(f"\n💾 Model saved to: models/heart_attack_model.pkl")
        print(f"💾 Scaler saved to: models/scaler.pkl")
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
