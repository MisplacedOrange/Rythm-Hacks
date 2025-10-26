#!/usr/bin/env python3
"""
Create a simple PyTorch model and test dataset
This ensures the PyTorch model and data are perfectly compatible
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Simple PyTorch model WITHOUT BatchNorm to avoid dimension issues
class SimplePyTorchMLP(nn.Module):
    """Simple MLP without BatchNorm - more robust for inference"""
    def __init__(self, input_size=10, hidden_size=32, num_classes=2):
        super(SimplePyTorchMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

def create_pytorch_model():
    print("=" * 60)
    print("🔥 Creating Simple PyTorch Model")
    print("=" * 60)
    
    # Generate synthetic data
    print("\n1️⃣ Generating synthetic classification data...")
    X, y = make_classification(
        n_samples=1000,
        n_features=10,  # 10 features to match model input
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
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create model
    print("\n3️⃣ Creating PyTorch model...")
    model = SimplePyTorchMLP(input_size=10, hidden_size=32, num_classes=2)
    print(f"   Model architecture:")
    print(f"   {model}")
    
    # Train model
    print("\n4️⃣ Training model...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    epochs = 50
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    # Evaluate
    print("\n5️⃣ Evaluating model...")
    model.eval()
    with torch.no_grad():
        # Training accuracy
        train_outputs = model(X_train_tensor)
        _, train_predicted = torch.max(train_outputs, 1)
        train_acc = (train_predicted == y_train_tensor).sum().item() / len(y_train)
        
        # Test accuracy
        test_outputs = model(X_test_tensor)
        _, test_predicted = torch.max(test_outputs, 1)
        test_acc = (test_predicted == y_test_tensor).sum().item() / len(y_test)
        
        print(f"   Training accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"   Test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Save model (using joblib for compatibility with backend)
    print("\n6️⃣ Saving PyTorch model...")
    model.eval()  # Set to eval mode before saving
    model_path = 'pytorch_demo_model.pkl'
    joblib.dump(model, model_path)
    print(f"   ✓ Model saved to: {model_path}")
    
    # Verify model can be loaded
    print("\n7️⃣ Verifying model can be loaded...")
    loaded_model = joblib.load(model_path)
    print(f"   ✓ Model loaded successfully: {type(loaded_model).__name__}")
    print(f"   Has .model attr: {hasattr(loaded_model, 'model')}")
    print(f"   Has .forward: {hasattr(loaded_model, 'forward')}")
    
    # Save test data as CSV
    print("\n8️⃣ Saving test data...")
    feature_names = [f'feature_{i}' for i in range(X_test.shape[1])]
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df['target'] = y_test
    
    csv_path = 'pytorch_demo_test.csv'
    test_df.to_csv(csv_path, index=False)
    print(f"   ✓ Test data saved to: {csv_path}")
    print(f"   Rows: {len(test_df)}, Columns: {len(test_df.columns)}")
    print(f"   Columns: {list(test_df.columns)}")
    
    # Show sample
    print("\n9️⃣ Sample of test data:")
    print(test_df.head())
    
    print("\n" + "=" * 60)
    print("✅ PyTorch Model and Test Data Created Successfully!")
    print("=" * 60)
    print("\n📦 Generated Files:")
    print(f"   • {model_path} - PyTorch MLP model (no BatchNorm)")
    print(f"   • {csv_path} - Test data with 10 features + target")
    
    print("\n🚀 How to Use:")
    print("   1. Go to your dashboard")
    print(f"   2. Upload '{model_path}' as your model")
    print("   3. Select framework: 'sklearn' (joblib will auto-detect PyTorch)")
    print(f"   4. Upload '{csv_path}' as test data")
    print("   5. Click 'Calculate Metrics'")
    print("   6. See your performance charts! 🔥")
    
    print("\n💡 This PyTorch model is guaranteed to work because:")
    print("   • NO BatchNorm layers (avoids dimension issues)")
    print("   • Model has .model attribute (compatible with custom_models.py)")
    print("   • Model and test data have matching dimensions (10 features)")
    print("   • Saved in eval mode for inference")
    print("   • Same preprocessing (none needed)")
    print()

if __name__ == "__main__":
    create_pytorch_model()
