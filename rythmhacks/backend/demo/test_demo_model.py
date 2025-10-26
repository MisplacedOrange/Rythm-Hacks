#!/usr/bin/env python3
"""
Test script to verify the demo model works correctly
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def test_demo_model():
    print("=" * 60)
    print("🧪 Testing Demo Model")
    print("=" * 60)
    
    # Load model
    print("\n1️⃣ Loading model...")
    try:
        model = joblib.load('heart_disease_model.pkl')
        print(f"   ✅ Model loaded: {type(model).__name__}")
        print(f"   📊 Estimators: {model.n_estimators}")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        return
    
    # Load test data
    print("\n2️⃣ Loading test data...")
    try:
        test_data = pd.read_csv('heart_disease_test_data.csv')
        print(f"   ✅ Test data loaded: {test_data.shape}")
        print(f"   📊 Columns: {list(test_data.columns)}")
    except Exception as e:
        print(f"   ❌ Failed to load test data: {e}")
        return
    
    # Prepare data
    print("\n3️⃣ Preparing features and target...")
    X_test = test_data.drop('target', axis=1)
    y_test = test_data['target']
    print(f"   ✅ Features shape: {X_test.shape}")
    print(f"   ✅ Target shape: {y_test.shape}")
    print(f"   📊 Target distribution: {dict(y_test.value_counts())}")
    
    # Make predictions
    print("\n4️⃣ Making predictions...")
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        print(f"   ✅ Predictions generated: {y_pred.shape}")
        print(f"   📊 Prediction distribution: {dict(pd.Series(y_pred).value_counts())}")
    except Exception as e:
        print(f"   ❌ Failed to make predictions: {e}")
        return
    
    # Calculate metrics
    print("\n5️⃣ Calculating performance metrics...")
    try:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"   ✅ Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   ✅ Precision: {precision:.4f}")
        print(f"   ✅ Recall:    {recall:.4f}")
        print(f"   ✅ F1 Score:  {f1:.4f}")
    except Exception as e:
        print(f"   ❌ Failed to calculate metrics: {e}")
        return
    
    # Feature importance
    print("\n6️⃣ Checking feature importance...")
    try:
        importances = model.feature_importances_
        feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        # Get top 5 features
        top_indices = np.argsort(importances)[-5:][::-1]
        print("   ✅ Top 5 important features:")
        for idx in top_indices:
            print(f"      {feature_names[idx]}: {importances[idx]:.4f}")
    except Exception as e:
        print(f"   ⚠️  Feature importance not available: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Demo Model Test Complete!")
    print("=" * 60)
    print("\n📝 Summary:")
    print(f"   • Model type: RandomForestClassifier")
    print(f"   • Test samples: {len(y_test)}")
    print(f"   • Performance: {accuracy*100:.1f}% accuracy")
    print(f"   • Ready for upload: ✅")
    print("\n🎯 Next Steps:")
    print("   1. Upload 'heart_disease_model.pkl' in the dashboard")
    print("   2. Upload 'heart_disease_test_data.csv' for metrics")
    print("   3. Select 'target' as the target column")
    print("   4. View the 2x2 performance charts!")
    print()

if __name__ == "__main__":
    test_demo_model()
