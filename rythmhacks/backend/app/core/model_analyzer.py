"""
Model Analyzer - Load and analyze uploaded ML models
Extract predictions and calculate comprehensive metrics
"""

import pickle
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from app.core.metrics import MetricsCalculator
from app.core.shap_analyzer import ShapAnalyzer

# Import custom model definitions for PyTorch
try:
    from app.core import custom_models
    CUSTOM_MODELS_AVAILABLE = True
except ImportError:
    CUSTOM_MODELS_AVAILABLE = False
    print("Custom models module not available")


class ModelAnalyzer:
    """Analyze uploaded ML models and generate performance metrics"""
    
    @staticmethod
    def load_model(model_path: Path, framework: str) -> Any:
        """Load model from file based on framework"""
        if framework == 'sklearn':
            try:
                return pickle.load(open(model_path, 'rb'))
            except:
                return joblib.load(model_path)
        
        elif framework == 'pytorch':
            try:
                import torch
                import sys
                import __main__
                
                print(f"Loading PyTorch model from: {model_path}")
                print(f"Custom models available: {CUSTOM_MODELS_AVAILABLE}")
                
                # Make custom models available during unpickling
                if CUSTOM_MODELS_AVAILABLE:
                    print("Registering custom model classes...")
                    
                    # Check if custom_models has TORCH_AVAILABLE
                    if not hasattr(custom_models, 'TORCH_AVAILABLE') or not custom_models.TORCH_AVAILABLE:
                        raise Exception(
                            "PyTorch is not available in custom_models module. "
                            "Make sure PyTorch is properly installed."
                        )
                    
                    # Register all custom models in __main__ namespace
                    for name, model_class in custom_models.CUSTOM_MODELS.items():
                        if model_class is not None:
                            setattr(__main__, name, model_class)
                            print(f"  ✓ Registered: {name}")
                        else:
                            print(f"  ✗ Skipped (None): {name}")
                    
                    # Also register in sys.modules for multiprocessing
                    sys.modules['__mp_main__'] = custom_models
                    
                    # Make classes available in current module too
                    for name, model_class in custom_models.CUSTOM_MODELS.items():
                        if model_class is not None:
                            globals()[name] = model_class
                else:
                    print("⚠️  Custom models module not available")
                
                # Load the model with custom classes available
                print("Calling torch.load...")
                model = torch.load(
                    model_path, 
                    map_location='cpu',
                    weights_only=False  # Required for models with custom classes
                )
                print(f"✓ Model loaded successfully: {type(model)}")
                return model
                
            except ImportError as e:
                raise Exception(f"PyTorch not installed: {str(e)}")
            except AttributeError as e:
                error_msg = str(e)
                # Extract class name from error message
                import re
                match = re.search(r"Can't get attribute '(\w+)'", error_msg)
                class_name = match.group(1) if match else "unknown"
                
                raise Exception(
                    f"Custom PyTorch model class '{class_name}' not found.\n\n"
                    f"To fix this:\n"
                    f"1. Open backend/app/core/custom_models.py\n"
                    f"2. Add your '{class_name}' class definition\n"
                    f"3. Add it to CUSTOM_MODELS dictionary\n"
                    f"4. Restart the backend server\n\n"
                    f"Original error: {error_msg}"
                )
            except Exception as e:
                print(f"❌ Error loading PyTorch model: {type(e).__name__}: {str(e)}")
                raise Exception(f"Failed to load PyTorch model: {str(e)}")
        
        elif framework == 'keras':
            try:
                from tensorflow import keras
                return keras.models.load_model(model_path)
            except ImportError:
                raise Exception("TensorFlow/Keras not installed")
        
        raise ValueError(f"Unsupported framework: {framework}")
    
    @staticmethod
    def analyze_model(
        model_path: Path,
        framework: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: Optional[list] = None,
        model_type: str = 'classifier'
    ) -> Dict[str, Any]:
        """
        Analyze model performance on test data
        
        Args:
            model_path: Path to model file
            framework: Model framework (sklearn, pytorch, keras)
            X_test: Test features
            y_test: Test labels
            feature_names: Names of features
            model_type: 'classifier' or 'regressor'
        
        Returns:
            Complete metrics dictionary including:
            - predictions
            - overall_metrics (accuracy, f1, etc.)
            - confusion_matrix (classification)
            - roc_data (classification)
            - feature_importance
        """
        # Load model
        model = ModelAnalyzer.load_model(model_path, framework)
        
        # Generate predictions
        if framework == 'sklearn':
            y_pred, y_pred_proba = ModelAnalyzer._predict_sklearn(
                model, X_test, model_type
            )
        elif framework == 'pytorch':
            y_pred, y_pred_proba = ModelAnalyzer._predict_pytorch(
                model, X_test, model_type
            )
        elif framework == 'keras':
            y_pred, y_pred_proba = ModelAnalyzer._predict_keras(
                model, X_test, model_type
            )
        else:
            raise ValueError(f"Unsupported framework: {framework}")
        
        # Calculate metrics
        if model_type == 'classifier':
            metrics = MetricsCalculator.calculate_classification_metrics(
                y_test, y_pred, y_pred_proba
            )
        else:  # regressor
            metrics = MetricsCalculator.calculate_regression_metrics(
                y_test, y_pred
            )
        
        # Extract feature importance
        feature_importance = None
        if feature_names:
            # Try standard feature importance first
            feature_importance = MetricsCalculator.extract_feature_importance(
                model, feature_names
            )
            
            # If not available, try SHAP (more comprehensive but slower)
            if feature_importance is None and framework == 'sklearn':
                try:
                    feature_importance = ShapAnalyzer.calculate_shap_values(
                        model, X_test, feature_names, model_type='auto'
                    )
                except Exception as e:
                    print(f"SHAP calculation failed: {e}")
        
        # Combine all results
        return {
            'model_type': model_type,
            'framework': framework,
            'overall_metrics': {
                k: v for k, v in metrics.items()
                if k in ['accuracy', 'precision', 'recall', 'f1_score', 
                        'mse', 'rmse', 'mae', 'r2', 'adjusted_r2']
            },
            'confusion_matrix': metrics.get('confusion_matrix'),
            'class_labels': metrics.get('class_labels'),
            'roc_data': metrics.get('roc_data'),
            'feature_importance': feature_importance,
            'n_test_samples': len(y_test)
        }
    
    @staticmethod
    def _predict_sklearn(
        model: Any,
        X: np.ndarray,
        model_type: str
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Generate predictions from sklearn model"""
        y_pred = model.predict(X)
        
        y_pred_proba = None
        if model_type == 'classifier' and hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X)
        
        return y_pred, y_pred_proba
    
    @staticmethod
    def _predict_pytorch(
        model: Any,
        X: np.ndarray,
        model_type: str
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Generate predictions from PyTorch model"""
        import torch
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X)
        
        # Set model to eval mode
        if hasattr(model, 'eval'):
            model.eval()
        
        # Generate predictions
        with torch.no_grad():
            if callable(model):
                outputs = model(X_tensor)
            else:
                # Model might be state dict
                raise Exception("PyTorch model must be a callable model, not state dict")
        
        if model_type == 'classifier':
            # Get probabilities
            if hasattr(outputs, 'softmax'):
                y_pred_proba = outputs.softmax(dim=1).numpy()
            else:
                y_pred_proba = torch.softmax(outputs, dim=1).numpy()
            
            # Get class predictions
            y_pred = y_pred_proba.argmax(axis=1)
            
            return y_pred, y_pred_proba
        else:
            # Regression
            y_pred = outputs.numpy().squeeze()
            return y_pred, None
    
    @staticmethod
    def _predict_keras(
        model: Any,
        X: np.ndarray,
        model_type: str
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Generate predictions from Keras model"""
        predictions = model.predict(X, verbose=0)
        
        if model_type == 'classifier':
            y_pred_proba = predictions
            y_pred = predictions.argmax(axis=1)
            return y_pred, y_pred_proba
        else:
            # Regression
            y_pred = predictions.squeeze()
            return y_pred, None
