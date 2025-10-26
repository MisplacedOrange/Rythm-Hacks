#!/usr/bin/env python3
"""
Create a simple working model and test dataset
This ensures the model and data are perfectly compatible
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

def create_simple_model():
    print("=" * 60)
    print("🏗️  Creating Simple Demo Model")
    print("=" * 60)
    
    # Generate synthetic data
    print("\n1️⃣ Generating synthetic classification data...")
    X, y = make_classification(
        n_samples=1000,
        n_features=10,  # Simple 10 features
        n_informative=7,
        n_redundant=2,
        n_classes=2,
        random_state=42,
        flip_y=0.1
    )
    
    print(f"   Dataset shape: {X.shape}")
    print(f"   Classes: {np.unique(y)}")
    print(f"   Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Split data
    print("\n2️⃣ Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Train model
    print("\n3️⃣ Training RandomForest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"   Training accuracy: {train_score:.4f} ({train_score*100:.2f}%)")
    print(f"   Test accuracy: {test_score:.4f} ({test_score*100:.2f}%)")
    
    # Save model
    print("\n4️⃣ Saving model...")
    model_path = 'simple_demo_model.pkl'
    joblib.dump(model, model_path)
    print(f"   ✓ Model saved to: {model_path}")
    print(f"   File size: {joblib.load(model_path).__sizeof__()} bytes")
    
    # Save test data as CSV
    print("\n5️⃣ Saving test data...")
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(X_test.shape[1])]
    
    # Create DataFrame with features + target
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df['target'] = y_test
    
    csv_path = 'simple_demo_test.csv'
    test_df.to_csv(csv_path, index=False)
    print(f"   ✓ Test data saved to: {csv_path}")
    print(f"   Rows: {len(test_df)}, Columns: {len(test_df.columns)}")
    print(f"   Columns: {list(test_df.columns)}")
    
    # Show sample of data
    print("\n6️⃣ Sample of test data:")
    print(test_df.head())
    
    print("\n" + "=" * 60)
    print("✅ Model and Test Data Created Successfully!")
    print("=" * 60)
    print("\n📦 Generated Files:")
    print(f"   • {model_path} - RandomForest model (sklearn)")
    print(f"   • {csv_path} - Test data with 10 features + target")
    
    print("\n🚀 How to Use:")
    print("   1. Go to your dashboard")
    print(f"   2. Upload '{model_path}' as your model")
    print(f"   3. Upload '{csv_path}' as test data")
    print("   4. Click 'Calculate Metrics'")
    print("   5. See your performance charts! 📊")
    
    print("\n💡 This model is guaranteed to work because:")
    print("   • Model and test data are from the same dataset")
    print("   • Same number of features (10)")
    print("   • Same preprocessing (none needed)")
    print("   • Compatible sklearn version")
    print()

if __name__ == "__main__":
    create_simple_model()
