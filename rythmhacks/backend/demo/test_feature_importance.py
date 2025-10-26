#!/usr/bin/env python3
"""
Test that feature importance is working
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from app.core.model_analyzer import ModelAnalyzer

# Load test data
df = pd.read_csv('health_test_data.csv')
X_test = df.drop('target', axis=1).values
y_test = df['target'].values
feature_names = list(df.columns[:-1])

print("=" * 60)
print("🔍 Testing Feature Importance")
print("=" * 60)
print(f"\nFeature names: {feature_names}")

# Analyze model
metrics = ModelAnalyzer.analyze_model(
    model_path='health_model.pkl',
    framework='sklearn',
    X_test=X_test,
    y_test=y_test,
    feature_names=feature_names,
    model_type='classifier'
)

print("\n" + "=" * 60)
print("📊 Results")
print("=" * 60)

# Check if feature importance exists
if metrics.get('feature_importance'):
    print("\n✅ Feature Importance Found!")
    fi = metrics['feature_importance']
    print(f"\nFeatures: {fi['features']}")
    print(f"\nImportances:")
    for feat, imp in zip(fi['features'], fi['importances']):
        bar = '█' * int(imp * 100)
        print(f"  {feat:20s} {bar} {imp:.4f} ({imp*100:.1f}%)")
else:
    print("\n❌ No feature importance in metrics!")
    print(f"Available keys: {list(metrics.keys())}")

print()
