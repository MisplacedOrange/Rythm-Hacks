"""
Metrics Calculation Engine
Calculate comprehensive ML performance metrics
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    precision_recall_fscore_support,
    mean_squared_error,
    r2_score,
    mean_absolute_error
)
from sklearn.preprocessing import label_binarize
from typing import Dict, List, Any, Optional


class MetricsCalculator:
    """Calculate and format metrics for visualization"""
    
    @staticmethod
    def calculate_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        class_labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (for ROC curve)
            class_labels: Names of classes
        
        Returns:
            - confusion_matrix: 2D array
            - roc_data: ROC curve data (if probabilities provided)
            - accuracy, precision, recall, f1_score
        """
        # Get unique classes
        unique_classes = np.unique(y_true)
        n_classes = len(unique_classes)
        
        if class_labels is None:
            class_labels = [f"Class {i}" for i in unique_classes]
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Overall metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # ROC curve data (if probabilities provided)
        roc_data = None
        if y_pred_proba is not None:
            roc_data = MetricsCalculator._calculate_roc_multiclass(
                y_true, y_pred_proba, n_classes, class_labels, unique_classes
            )
        
        return {
            'confusion_matrix': cm.tolist(),
            'class_labels': class_labels,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_data': roc_data,
            'n_classes': n_classes
        }
    
    @staticmethod
    def _calculate_roc_multiclass(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        n_classes: int,
        class_labels: List[str],
        unique_classes: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate ROC curve for multi-class classification"""
        try:
            # Binarize labels
            y_true_bin = label_binarize(y_true, classes=unique_classes)
            
            # Handle binary classification (label_binarize returns 1D array)
            if n_classes == 2:
                y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
            
            fpr = []
            tpr = []
            roc_auc = []
            
            for i in range(n_classes):
                # Get probabilities for this class
                if y_pred_proba.ndim == 1:
                    # Binary classification
                    proba_i = y_pred_proba if i == 1 else 1 - y_pred_proba
                else:
                    proba_i = y_pred_proba[:, i]
                
                fpr_i, tpr_i, _ = roc_curve(y_true_bin[:, i], proba_i)
                fpr.append(fpr_i.tolist())
                tpr.append(tpr_i.tolist())
                roc_auc.append(float(auc(fpr_i, tpr_i)))
            
            return {
                'classes': class_labels,
                'fpr': fpr,
                'tpr': tpr,
                'auc': roc_auc
            }
        except Exception as e:
            print(f"Error calculating ROC curve: {e}")
            return None
    
    @staticmethod
    def calculate_regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate regression metrics"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        # Calculate adjusted R² (requires n and p)
        n = len(y_true)
        # Assuming simple model, p=1 for now
        p = 1
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'adjusted_r2': float(adj_r2)
        }
    
    @staticmethod
    def extract_feature_importance(
        model: Any,
        feature_names: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Extract feature importance from various model types
        
        Supports:
            - Tree-based models (sklearn)
            - Linear models (sklearn)
            - Neural networks (basic)
        """
        importances = None
        
        # Tree-based models (RandomForest, DecisionTree, etc.)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        
        # Linear models (coefficient magnitude)
        elif hasattr(model, 'coef_'):
            coef = model.coef_
            if len(coef.shape) > 1:
                # Multi-class: average across classes
                importances = np.mean(np.abs(coef), axis=0)
            else:
                importances = np.abs(coef)
        
        if importances is None:
            return None
        
        # Normalize to 0-1
        if importances.sum() > 0:
            importances = importances / importances.sum()
        
        return {
            'features': feature_names,
            'importances': importances.tolist()
        }
