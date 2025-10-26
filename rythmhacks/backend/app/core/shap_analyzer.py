"""
SHAP Analyzer - Generate SHAP values for model interpretability
Supports tree-based, neural network, and linear models
"""

import numpy as np
from typing import Dict, Any, Optional, List

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP library not installed. Install with: pip install shap")


class ShapAnalyzer:
    """Generate SHAP values for model explainability"""
    
    @staticmethod
    def calculate_shap_values(
        model: Any,
        X: np.ndarray,
        feature_names: List[str],
        model_type: str = 'tree',
        max_samples: int = 100
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate SHAP values for model predictions
        
        Args:
            model: Trained model
            X: Input features (sample data for background)
            feature_names: Names of features
            model_type: 'tree', 'linear', 'neural', or 'auto'
            max_samples: Maximum samples to use for explanation
        
        Returns:
            {
                'features': List of feature names,
                'shap_values': Average absolute SHAP values,
                'base_value': Expected value,
                'sample_shap_values': SHAP values for each sample (limited)
            }
        """
        if not SHAP_AVAILABLE:
            return None
        
        try:
            # Limit samples for performance
            if len(X) > max_samples:
                # Use random sample
                indices = np.random.choice(len(X), max_samples, replace=False)
                X_sample = X[indices]
            else:
                X_sample = X
            
            # Auto-detect model type if needed
            if model_type == 'auto':
                model_type = ShapAnalyzer._detect_model_type(model)
            
            # Create appropriate explainer
            if model_type == 'tree':
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
            
            elif model_type == 'linear':
                explainer = shap.LinearExplainer(model, X_sample)
                shap_values = explainer.shap_values(X_sample)
            
            elif model_type == 'neural':
                # Use KernelExplainer for neural networks (slower but works for any model)
                background = shap.sample(X_sample, min(100, len(X_sample)))
                explainer = shap.KernelExplainer(model.predict, background)
                shap_values = explainer.shap_values(X_sample)
            
            else:
                # Default to Kernel explainer
                background = shap.sample(X_sample, min(100, len(X_sample)))
                explainer = shap.KernelExplainer(model.predict, background)
                shap_values = explainer.shap_values(X_sample)
            
            # Handle multi-class output (shap_values is list of arrays)
            if isinstance(shap_values, list):
                # For multi-class, average across classes
                shap_values_avg = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            else:
                shap_values_avg = np.abs(shap_values)
            
            # Calculate mean absolute SHAP values per feature
            mean_shap = np.mean(shap_values_avg, axis=0)
            
            # Get base value
            base_value = explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = float(np.mean(base_value))
            else:
                base_value = float(base_value)
            
            return {
                'features': feature_names,
                'importances': mean_shap.tolist(),  # Average absolute SHAP values
                'base_value': base_value,
                'shap_type': model_type,
                'n_samples': len(X_sample)
            }
        
        except Exception as e:
            print(f"Error calculating SHAP values: {e}")
            return None
    
    @staticmethod
    def _detect_model_type(model: Any) -> str:
        """Auto-detect model type for SHAP explainer selection"""
        model_class = model.__class__.__name__.lower()
        
        # Tree-based models
        if any(keyword in model_class for keyword in [
            'forest', 'tree', 'gradient', 'boosting', 'xgb', 'lgb', 'catboost'
        ]):
            return 'tree'
        
        # Linear models
        elif any(keyword in model_class for keyword in [
            'linear', 'logistic', 'ridge', 'lasso'
        ]):
            return 'linear'
        
        # Neural networks
        elif any(keyword in model_class for keyword in [
            'neural', 'mlp', 'sequential', 'model'
        ]):
            return 'neural'
        
        # Default to kernel explainer (works for any model)
        return 'kernel'
