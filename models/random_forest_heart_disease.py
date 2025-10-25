import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import joblib
import time
import matplotlib.pyplot as plt

# Load Heart Disease dataset
def load_heart_disease_data():
    print("Loading Heart Disease dataset from UCI...")
    
    try:
        from ucimlrepo import fetch_ucirepo
        
        # Fetch dataset
        heart_disease = fetch_ucirepo(id=45)
        
        # Get features and targets
        X = heart_disease.data.features
        y = heart_disease.data.targets
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Convert target to binary: 0 = no disease, 1+ = disease present
        y = (y > 0).astype(int).values.ravel()
        
        print(f"Dataset shape: {X.shape}")
        print(f"Features: {list(X.columns)}")
        print(f"Class distribution: No disease={np.sum(y==0)}, Disease={np.sum(y==1)}")
        
        return X, y, X.columns.tolist()
        
    except Exception as e:
        print(f"Error loading from ucimlrepo: {e}")
        print("Please install: pip install ucimlrepo")
        return None, None, None

# Train Random Forest
def train_random_forest(X_train, y_train, n_estimators=200):
    print(f"\nTraining Random Forest with {n_estimators} trees...")
    
    # Initialize Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',  # Handle class imbalance
        verbose=0
    )
    
    # Train the model
    start_time = time.time()
    rf_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    
    return rf_model

# Hyperparameter tuning
def tune_hyperparameters(X_train, y_train):
    print("\nPerforming hyperparameter tuning...")
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
    
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='accuracy', 
        verbose=1, n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

# Evaluate the model
def evaluate_model(model, X_train, y_train, X_test, y_test):
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Training predictions
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"\nTraining Accuracy: {train_accuracy * 100:.2f}%")
    
    # Test predictions
    start_time = time.time()
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    prediction_time = time.time() - start_time
    
    # Calculate metrics
    test_accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Prediction completed in {prediction_time:.4f} seconds")
    print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    
    # Classification report
    print("\n" + "-"*50)
    print("Classification Report:")
    print("-"*50)
    print(classification_report(y_test, y_pred, target_names=['No Disease', 'Disease']))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"\nTrue Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
    
    return test_accuracy, y_pred, y_pred_proba

# Cross-validation
def perform_cross_validation(model, X, y, cv=5):
    print(f"\nPerforming {cv}-fold cross-validation...")
    
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return cv_scores

# Feature importance analysis
def analyze_feature_importance(model, feature_names):
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*50)
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nFeature Ranking:")
    for i, idx in enumerate(indices):
        print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    return importances, indices

# Main execution
def main():
    # Load data
    X, y, feature_names = load_heart_disease_data()
    
    if X is None or y is None:
        return
    
    # Convert to numpy if pandas DataFrame
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Optional: Standardize features (Random Forest doesn't require it, but can help)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest (choose one approach)
    print("\nOption 1: Training with default good parameters...")
    rf_model = train_random_forest(X_train_scaled, y_train, n_estimators=200)
    
    # Uncomment below for hyperparameter tuning (takes longer)
    # print("\nOption 2: Training with hyperparameter tuning...")
    # rf_model = tune_hyperparameters(X_train_scaled, y_train)
    
    # Cross-validation
    cv_scores = perform_cross_validation(rf_model, X_train_scaled, y_train, cv=5)
    
    # Evaluate
    test_accuracy, predictions, probabilities = evaluate_model(
        rf_model, X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    # Feature importance
    importances, indices = analyze_feature_importance(rf_model, feature_names)
    
    # Save the model
    model_data = {
        'model': rf_model,
        'scaler': scaler,
        'feature_names': feature_names,
        'test_accuracy': test_accuracy
    }
    
    joblib.dump(model_data, 'heart_disease_rf.pkl')
    print("\n" + "="*50)
    print("Model and scaler saved as heart_disease_rf.pkl")
    
    # Print final summary
    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    print(f"Model: Random Forest with {rf_model.n_estimators} trees")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
    print(f"Most important feature: {feature_names[indices[0]]}")
    print("="*50)

if __name__ == '__main__':
    main()