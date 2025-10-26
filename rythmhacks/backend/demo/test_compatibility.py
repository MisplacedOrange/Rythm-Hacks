#!/usr/bin/env python3
"""
Quick test to verify demo models work with the backend
Run this BEFORE uploading to dashboard to ensure compatibility
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from app.core.model_analyzer import ModelAnalyzer

def test_model(model_path, csv_path, model_name):
    print(f"\n{'='*60}")
    print(f"🧪 Testing {model_name}")
    print(f"{'='*60}")
    
    try:
        # Load test data
        print(f"\n1️⃣ Loading test data from {csv_path}...")
        df = pd.read_csv(csv_path)
        X_test = df.drop('target', axis=1).values
        y_test = df['target'].values
        print(f"   ✓ Data loaded: {X_test.shape} features, {y_test.shape} targets")
        
        # Analyze model (this is what the backend does)
        print(f"\n2️⃣ Analyzing model with backend code...")
        metrics = ModelAnalyzer.analyze_model(
            model_path=model_path,
            framework='sklearn',  # Backend will auto-detect PyTorch
            X_test=X_test,
            y_test=y_test,
            model_type='classifier'
        )
        
        # Display results
        print(f"\n3️⃣ Results:")
        print(f"   ✓ Framework: {metrics['framework']}")
        print(f"   ✓ Model type: {metrics['model_type']}")
        print(f"\n   📊 Metrics:")
        for key, value in metrics['overall_metrics'].items():
            if isinstance(value, (int, float)):
                print(f"      {key}: {value:.4f}")
        
        print(f"\n   ✅ {model_name} PASSED - Ready for dashboard upload!")
        return True
        
    except Exception as e:
        print(f"\n   ❌ {model_name} FAILED")
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("🔬 Demo Models Compatibility Test")
    print("=" * 60)
    print("\nThis script tests if demo models work with the backend")
    print("Run this BEFORE uploading to the dashboard\n")
    
    results = {}
    
    # Test health model (BEST - has feature importance!)
    if os.path.exists('health_model.pkl'):
        print("\n" + "⭐" * 30)
        print("Testing HEALTH MODEL (Best Choice - Has Feature Importance!)")
        print("⭐" * 30)
        results['health'] = test_model(
            'health_model.pkl',
            'health_test_data.csv',
            'Health Predictor (with Feature Importance)'
        )
    else:
        print("\n⚠️  health_model.pkl not found - run create_feature_rich_model.py")
        results['health'] = False
    
    # Test sklearn model
    if os.path.exists('simple_demo_model.pkl'):
        results['sklearn'] = test_model(
            'simple_demo_model.pkl',
            'simple_demo_test.csv',
            'sklearn RandomForest'
        )
    else:
        print("\n⚠️  simple_demo_model.pkl not found - run create_simple_model.py first")
        results['sklearn'] = False
    
    # Test PyTorch model
    if os.path.exists('pytorch_demo_model.pkl'):
        results['pytorch'] = test_model(
            'pytorch_demo_model.pkl',
            'pytorch_demo_test.csv',
            'PyTorch MLP'
        )
    else:
        print("\n⚠️  pytorch_demo_model.pkl not found - run create_pytorch_model.py first")
        results['pytorch'] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} - {name}")
    
    print(f"\n   Total: {passed}/{total} models ready for upload")
    
    if passed == total:
        print("\n🎉 All models passed! You can upload them to the dashboard.")
        print("\n📤 Recommended: Upload the Health Model first!")
        print("\n⭐ BEST OPTION - Health Model (has Feature Importance):")
        print("   1. Go to dashboard")
        print("   2. Upload health_model.pkl")
        print("   3. Upload health_test_data.csv")
        print("   4. Click 'Calculate Metrics'")
        print("   5. See ALL charts including Feature Importance! 📊")
        print("\n   Features shown in chart:")
        print("   - age, blood_pressure, cholesterol")
        print("   - heart_rate, bmi, glucose_level")
        print("   - exercise_hours, sleep_hours, stress_level")
        print("   - alcohol_consumption, smoking_status, family_history")
    else:
        print("\n⚠️  Some models failed. Check errors above.")
        print("   Try recreating models with:")
        print("   • python create_simple_model.py")
        print("   • python create_pytorch_model.py")
    
    print()

if __name__ == "__main__":
    # Change to demo directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
