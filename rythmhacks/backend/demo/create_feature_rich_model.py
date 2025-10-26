#!/usr/bin/env python3
"""
Create a demo model with meaningful feature names and guaranteed feature importance
Uses real-world-like feature names for better visualization
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

def create_feature_rich_model():
    print("=" * 60)
    print("🌟 Creating Feature-Rich Demo Model")
    print("=" * 60)
    
    # Generate synthetic data
    print("\n1️⃣ Generating synthetic health data...")
    X, y = make_classification(
        n_samples=1000,
        n_features=12,  # 12 meaningful features
        n_informative=10,
        n_redundant=2,
        n_classes=2,
        random_state=42,
        flip_y=0.05,
        class_sep=1.2
    )
    
    # Create meaningful feature names (health-related)
    feature_names = [
        'age',
        'blood_pressure',
        'cholesterol',
        'heart_rate',
        'bmi',
        'glucose_level',
        'exercise_hours',
        'sleep_hours',
        'stress_level',
        'alcohol_consumption',
        'smoking_status',
        'family_history'
    ]
    
    print(f"   Dataset shape: {X.shape}")
    print(f"   Features: {', '.join(feature_names)}")
    print(f"   Classes: {np.unique(y)}")
    print(f"   Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Split data
    print("\n2️⃣ Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Train RandomForest (guaranteed to have feature_importances_)
    print("\n3️⃣ Training RandomForest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"   Training accuracy: {train_score:.4f} ({train_score*100:.2f}%)")
    print(f"   Test accuracy: {test_score:.4f} ({test_score*100:.2f}%)")
    
    # Show feature importances
    print("\n4️⃣ Feature Importances:")
    importances = model.feature_importances_
    feature_importance_pairs = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )
    for feat, imp in feature_importance_pairs[:5]:
        print(f"   {feat:20s}: {imp:.4f} ({imp*100:.1f}%)")
    
    # Save model
    print("\n5️⃣ Saving model...")
    model_path = 'health_model.pkl'
    joblib.dump(model, model_path)
    print(f"   ✓ Model saved to: {model_path}")
    
    # Save test data with meaningful column names
    print("\n6️⃣ Saving test data with feature names...")
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df['target'] = y_test
    
    csv_path = 'health_test_data.csv'
    test_df.to_csv(csv_path, index=False)
    print(f"   ✓ Test data saved to: {csv_path}")
    print(f"   Rows: {len(test_df)}, Columns: {len(test_df.columns)}")
    print(f"   Column names: {list(test_df.columns)}")
    
    # Show sample
    print("\n7️⃣ Sample of test data:")
    print(test_df.head(3))
    
    print("\n" + "=" * 60)
    print("✅ Feature-Rich Model Created Successfully!")
    print("=" * 60)
    print("\n📦 Generated Files:")
    print(f"   • {model_path} - RandomForest with meaningful features")
    print(f"   • {csv_path} - Test data with health-related feature names")
    
    print("\n🎯 Feature Importance:")
    print("   This model has GUARANTEED feature importance!")
    print("   The dashboard will show a bar chart with these features:")
    for feat, imp in feature_importance_pairs[:5]:
        bar = '█' * int(imp * 50)
        print(f"   {feat:20s} {bar} {imp:.1%}")
    
    print("\n🚀 How to Use:")
    print("   1. Go to your dashboard")
    print(f"   2. Upload '{model_path}' as your model")
    print(f"   3. Upload '{csv_path}' as test data")
    print("   4. Click 'Calculate Metrics'")
    print("   5. See Feature Importance chart! 📊")
    
    print("\n💡 Why Feature Importance Will Show:")
    print("   ✅ RandomForest has .feature_importances_ attribute")
    print("   ✅ CSV has meaningful column names (not feature_0, feature_1)")
    print("   ✅ Frontend sends feature_names from CSV headers")
    print("   ✅ Backend extracts and returns importance data")
    print()

if __name__ == "__main__":
    create_feature_rich_model()
