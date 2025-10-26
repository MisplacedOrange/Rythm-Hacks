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
        
        # Register custom models for ALL frameworks (needed for joblib/pickle too)
        if CUSTOM_MODELS_AVAILABLE:
            import sys
            import __main__
            
            # Register all custom models in __main__ namespace
            for name, model_class in custom_models.CUSTOM_MODELS.items():
                if model_class is not None:
                    setattr(__main__, name, model_class)
            
            # Register in sys.modules['__mp_main__'] for multiprocessing/joblib
            if '__mp_main__' not in sys.modules:
                import types
                mp_module = types.ModuleType('__mp_main__')
                sys.modules['__mp_main__'] = mp_module
            else:
                mp_module = sys.modules['__mp_main__']
            
            # Add all custom model classes to __mp_main__
            for name, model_class in custom_models.CUSTOM_MODELS.items():
                if model_class is not None:
                    setattr(mp_module, name, model_class)
            
            # Make classes available in current module too
            for name, model_class in custom_models.CUSTOM_MODELS.items():
                if model_class is not None:
                    globals()[name] = model_class
        
        if framework == 'sklearn':
            try:
                model = pickle.load(open(model_path, 'rb'))
            except:
                model = joblib.load(model_path)
            
            # Handle wrapped models (dict with 'model' key)
            if isinstance(model, dict):
                if 'model' in model:
                    print(f"📦 Unwrapping model from dict (key: 'model')")
                    return model['model']
                elif 'state_dict' in model:
                    raise Exception(
                        "Model is a state_dict. Please save the full model object, "
                        "not just state_dict. Use: joblib.dump(model, 'model.pkl')"
                    )
                else:
                    raise Exception(
                        f"Model is a dict but doesn't contain 'model' key. "
                        f"Available keys: {list(model.keys())}"
                    )
            
            return model
        
        elif framework == 'pytorch':
            try:
                import torch
                
                print(f"Loading PyTorch model from: {model_path}")
                print(f"Custom models available: {CUSTOM_MODELS_AVAILABLE}")
                
                if not CUSTOM_MODELS_AVAILABLE:
                    print("⚠️  Custom models module not available")
                
                # Load the model (custom classes already registered above)
                print("Calling torch.load...")
                loaded = torch.load(
                    model_path, 
                    map_location='cpu',
                    weights_only=False  # Required for models with custom classes
                )
                
                # Handle different PyTorch save formats
                if isinstance(loaded, dict):
                    if 'model' in loaded:
                        print(f"📦 Unwrapping model from dict (key: 'model')")
                        model = loaded['model']
                    elif 'state_dict' in loaded:
                        raise Exception(
                            "Model is a state_dict. Please save the full model object, "
                            "not just state_dict. Use: torch.save(model, 'model.pt')"
                        )
                    elif 'model_state_dict' in loaded:
                        raise Exception(
                            "Model contains model_state_dict. Please save the full model object, "
                            "not just the state dict."
                        )
                    else:
                        raise Exception(
                            f"PyTorch model is a dict but doesn't contain 'model' key. "
                            f"Available keys: {list(loaded.keys())}"
                        )
                else:
                    model = loaded
                
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
        
        print(f"\n🔍 Model loaded:")
        print(f"   Type: {type(model)}")
        print(f"   Has .model attr: {hasattr(model, 'model')}")
        print(f"   Has .predict: {hasattr(model, 'predict')}")
        print(f"   Has .forward: {hasattr(model, 'forward')}")
        print(f"   Callable: {callable(model)}")
        
        # Auto-detect actual model type (handle PyTorch models saved with joblib)
        try:
            import torch
            if isinstance(model, torch.nn.Module):
                print(f"   ✓ Detected as PyTorch model (torch.nn.Module)")
                framework = 'pytorch'
                
                # FIX: If model is missing 'model' attribute but HAS the sequential layers
                # This happens when model was saved with old version of HeartDiseaseMLP
                if not hasattr(model, 'model'):
                    print(f"   ⚠️ PyTorch model missing .model attribute - attempting fix")
                    # Check if this is a HeartDiseaseMLP that needs patching
                    if type(model).__name__ == 'HeartDiseaseMLP':
                        # Try to reconstruct self.model from existing layers
                        # The old forward() might have used individual layers instead of self.model
                        print(f"   Attempting to patch HeartDiseaseMLP...")
                        # Instead of patching, let's just override the forward method to work directly
                        original_forward = model.forward
                        def patched_forward(x):
                            # Try calling parent's forward if it exists
                            # Otherwise use the layers directly
                            if hasattr(model, '_modules') and len(model._modules) > 0:
                                # Use Sequential on existing modules
                                import torch.nn as nn
                                sequential = nn.Sequential(*model._modules.values())
                                return sequential(x)
                            else:
                                # Fallback to original forward (will probably fail)
                                return original_forward(x)
                        model.forward = lambda x: patched_forward(x)
                        print(f"   ✓ Patched forward method to use _modules directly")
                print(f"   Switching to PyTorch prediction method")
            else:
                print(f"   ✓ Not a PyTorch model")
        except ImportError:
            print(f"   ⚠️ PyTorch not available for detection")
        except Exception as e:
            print(f"   ⚠️ Error during PyTorch compatibility fix: {e}")
        
        print(f"   Final framework: {framework}\n")
        
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
