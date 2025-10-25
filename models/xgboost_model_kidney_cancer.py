import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

# Load TCGA Kidney Cancer dataset
def load_kidney_cancer_data():
    print("Loading sample kidney cancer dataset...")
    print("Note: This is a simplified version for testing the model.")
    
    try:
        # Create a small synthetic dataset for testing
        np.random.seed(42)
        n_samples = 100
        n_features = 1000  # Reduced from 60,660 for testing
        n_classes = 3
        
        # Generate synthetic features (gene expression data)
        X = np.random.normal(0, 1, (n_samples, n_features))
        
        # Generate synthetic labels (cancer types)
        y = np.random.randint(0, n_classes, n_samples)
        
        # Create class names
        class_names = np.array(['Type_A', 'Type_B', 'Type_C'])
        
        print(f"\nDataset shape: {X.shape}")
        print(f"Number of features (genes): {X.shape[1]}")
        print(f"Number of samples: {X.shape[0]}")
        print(f"\nTarget classes:")
        unique_counts = np.bincount(y)
        for i, count in enumerate(unique_counts):
            print(f"{class_names[i]}: {count}")
        
        # Encode labels
        le = LabelEncoder()
        le.classes_ = class_names
        y_encoded = y  # Already numeric, no need to encode
        
        print(f"\nEncoded classes: {le.classes_}")
        
        # Convert to pandas DataFrame for consistency
        X = pd.DataFrame(X, columns=[f'Gene_{i}' for i in range(n_features)])
        
        return X, y_encoded, le
        
    except Exception as e:
        print(f"Error creating synthetic dataset: {e}")
        return None, None, None

# Feature selection using multiple methods
def select_features(X_train, y_train, X_test, n_features=500, method='mutual_info'):
    print(f"\n{'='*60}")
    print(f"FEATURE SELECTION: Reducing from {X_train.shape[1]} to {n_features} features")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    if method == 'mutual_info':
        print("Using Mutual Information for feature selection...")
        selector = SelectKBest(mutual_info_classif, k=n_features)
    else:
        print("Using ANOVA F-test for feature selection...")
        selector = SelectKBest(f_classif, k=n_features)
    
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Get selected feature indices
    selected_indices = selector.get_support(indices=True)
    
    elapsed = time.time() - start_time
    print(f"Feature selection completed in {elapsed:.2f} seconds")
    print(f"Selected features shape: {X_train_selected.shape}")
    
    return X_train_selected, X_test_selected, selector, selected_indices

# Optional: PCA for additional dimensionality reduction
def apply_pca(X_train, X_test, n_components=100):
    print(f"\nApplying PCA to reduce to {n_components} components...")
    
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f"Explained variance: {explained_var:.4f}")
    
    return X_train_pca, X_test_pca, pca

# Train XGBoost model
def train_xgboost(X_train, y_train, X_val, y_val, n_classes=3):
    print(f"\n{'='*60}")
    print("TRAINING XGBOOST MODEL")
    print(f"{'='*60}")
    
    # XGBoost parameters optimized for high-dimensional genomic data
    params = {
        'objective': 'multi:softmax',
        'num_class': n_classes,
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist',
        'eval_metric': 'mlogloss'
    }
    
    print("\nModel parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Create model
    model = xgb.XGBClassifier(**params)
    
    # Train model
    start_time = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    return model

# Comprehensive evaluation
def evaluate_model(model, X_train, y_train, X_test, y_test, label_encoder):
    print(f"\n{'='*60}")
    print("MODEL EVALUATION")
    print(f"{'='*60}")
    
    # Training predictions
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    
    print(f"\nTraining Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Training F1 Score: {train_f1:.4f}")
    
    # Test predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test F1 Score: {test_f1:.4f}")
    
    # Classification report
    print(f"\n{'-'*60}")
    print("Classification Report:")
    print(f"{'-'*60}")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Per-class accuracy
    print("\nPer-class accuracy:")
    for i, class_name in enumerate(label_encoder.classes_):
        class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        print(f"  {class_name}: {class_acc * 100:.2f}%")
    
    return test_accuracy, test_f1, y_pred, y_pred_proba

# Cross-validation
def perform_cross_validation(X, y, n_splits=5):
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION ({n_splits}-fold)")
    print(f"{'='*60}")
    
    # Simplified model for CV
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(np.unique(y)),
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
    
    print(f"\nCV Scores: {[f'{score:.4f}' for score in cv_scores]}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return cv_scores

# Feature importance analysis
def analyze_feature_importance(model, selected_indices, original_feature_names, top_n=20):
    print(f"\n{'='*60}")
    print(f"TOP {top_n} MOST IMPORTANT GENES")
    print(f"{'='*60}")
    
    importances = model.feature_importances_
    
    # Get top features
    top_indices = np.argsort(importances)[::-1][:top_n]
    
    print("\nFeature Ranking:")
    for i, idx in enumerate(top_indices):
        original_idx = selected_indices[idx]
        gene_name = original_feature_names[original_idx]
        importance = importances[idx]
        print(f"{i+1}. {gene_name}: {importance:.6f}")
    
    return importances

# Main execution
def main():
    print(f"{'='*60}")
    print("TCGA KIDNEY CANCER CLASSIFICATION WITH XGBOOST")
    print(f"{'='*60}")
    
    # Load data
    X, y, label_encoder = load_kidney_cancer_data()
    
    if X is None or y is None:
        return
    
    # Store original feature names
    if isinstance(X, pd.DataFrame):
        original_feature_names = X.columns.tolist()
        X = X.values
    else:
        original_feature_names = [f"Gene_{i}" for i in range(X.shape[1])]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Further split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"\nData split:")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    
    # Feature selection (reduce from 60k to 500 most informative features)
    X_train_selected, X_test_selected, selector, selected_indices = select_features(
        X_train, y_train, X_test, n_features=500, method='mutual_info'
    )
    X_val_selected = selector.transform(X_val)
    
    # Optional: Apply PCA for further reduction (uncomment if needed)
    # X_train_selected, X_test_selected, pca = apply_pca(X_train_selected, X_test_selected, n_components=100)
    # X_val_selected = pca.transform(X_val_selected)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_val_scaled = scaler.transform(X_val_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # Train XGBoost
    xgb_model = train_xgboost(
        X_train_scaled, y_train, 
        X_val_scaled, y_val,
        n_classes=len(label_encoder.classes_)
    )
    
    # Evaluate
    test_accuracy, test_f1, predictions, probabilities = evaluate_model(
        xgb_model, X_train_scaled, y_train, X_test_scaled, y_test, label_encoder
    )
    
    # Cross-validation on selected features
    cv_scores = perform_cross_validation(X_train_scaled, y_train, n_splits=5)
    
    # Feature importance
    importances = analyze_feature_importance(
        xgb_model, selected_indices, original_feature_names, top_n=20
    )
    
    # Save the complete pipeline
    model_data = {
        'model': xgb_model,
        'selector': selector,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'selected_indices': selected_indices,
        'feature_names': original_feature_names,
        'test_accuracy': test_accuracy,
        'test_f1': test_f1
    }
    
    joblib.dump(model_data, 'kidney_cancer_xgboost.pkl')
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Model: XGBoost Classifier")
    print(f"Original features: 60,660 genes")
    print(f"Selected features: {X_train_selected.shape[1]}")
    print(f"Classes: {', '.join(label_encoder.classes_)}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
    print(f"\nModel saved as kidney_cancer_xgboost.pkl")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()