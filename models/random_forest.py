import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torchvision import datasets, transforms
import joblib
import time
import sys
from pathlib import Path

# Load MNIST dataset
def load_mnist_data():
    print("Loading MNIST dataset...")
    
    # Simple transformation to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Load datasets
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Convert to numpy arrays and flatten
    X_train = train_dataset.data.numpy().reshape(-1, 784) / 255.0
    y_train = train_dataset.targets.numpy()
    
    X_test = test_dataset.data.numpy().reshape(-1, 784) / 255.0
    y_test = test_dataset.targets.numpy()
    
    print(f"Training set: {X_train.shape}, Labels: {y_train.shape}")
    print(f"Test set: {X_test.shape}, Labels: {y_test.shape}")
    
    return X_train, y_train, X_test, y_test

# Train Random Forest
def train_random_forest(X_train, y_train, n_estimators=100):
    print(f"\nTraining Random Forest with {n_estimators} trees...")
    
    # Initialize Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,  # Use all available cores
        verbose=1
    )
    
    # Train the model
    start_time = time.time()
    rf_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    
    return rf_model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    print("\nEvaluating model...")
    
    # Make predictions
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Prediction completed in {prediction_time:.2f} seconds")
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Try to save a confusion matrix plot and feature importances
    # Robust import of top-level visualize module (works when running from models/)
    def _import_visualize():
        try:
            import visualize
            return visualize
        except ModuleNotFoundError:
            repo_root = Path(__file__).resolve().parent.parent
            repo_root_str = str(repo_root)
            if repo_root_str not in sys.path:
                sys.path.insert(0, repo_root_str)
            import visualize
            return visualize

    try:
        viz = _import_visualize()
        viz.plot_confusion_matrix(y_test, y_pred, out_file='rf_confusion_matrix.png') 
        print('Saved confusion matrix: rf_confusion_matrix.png')
    except Exception as e:
        print(f'Could not save confusion matrix plot: {e}')

    try:
        viz = _import_visualize()
        viz.plot_feature_importances(model.feature_importances_, top_n=20, out_file='rf_feature_importances.png')
        print('Saved feature importances: rf_feature_importances.png')
    except Exception as e:
        print(f'Could not save feature importances plot: {e}')

    # If features correspond to MNIST pixels (784), also save a 28x28 heatmap
    try:
        if getattr(model, 'feature_importances_', None) is not None and len(model.feature_importances_) == 28*28:
            viz = _import_visualize()
            viz.plot_pixel_importances(model.feature_importances_, shape=(28,28), out_file='rf_pixel_importances.png')
            print('Saved pixel importances heatmap: rf_pixel_importances.png')
    except Exception as e:
        print(f'Could not save pixel importances heatmap: {e}')

    return accuracy, y_pred

# Feature importance
def show_feature_importance(model, top_n=20):
    print(f"\nTop {top_n} Most Important Features (pixels):")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    for i, idx in enumerate(indices):
        row = idx // 28
        col = idx % 28
        print(f"{i+1}. Pixel [{row}, {col}] (index {idx}): {importances[idx]:.4f}")

# Main execution
def main():
    # Load data
    X_train, y_train, X_test, y_test = load_mnist_data()
    
    # Train Random Forest
    rf_model = train_random_forest(X_train, y_train, n_estimators=100)
    
    # Evaluate
    accuracy, predictions = evaluate_model(rf_model, X_test, y_test)
    
    # Show feature importance
    show_feature_importance(rf_model)
    
    # Save the model
    joblib.dump(rf_model, 'mnist_random_forest.pkl')
    print("\nModel saved as mnist_random_forest.pkl")
    
    # Print model info
    print(f"\nModel Info:")
    print(f"Number of trees: {rf_model.n_estimators}")
    print(f"Max depth: {rf_model.max_depth}")
    print(f"Number of features: {rf_model.n_features_in_}")

if __name__ == '__main__':
    main()