"""
Regression API Routes
Endpoints for training and predicting with regression models
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
import pickle
from pathlib import Path

from app.core.regression import RegressionEngine
from app.utils.helpers import generate_unique_id

router = APIRouter(prefix="/api/regression")


# Pydantic Models
class RegressionTrainRequest(BaseModel):
    x: List[float]
    y: List[float]
    regression_type: str = 'linear'
    params: Optional[Dict[str, Any]] = {}


class RegressionPredictRequest(BaseModel):
    session_id: str
    x: List[float]


# Helper Functions
def save_regression_session(engine: RegressionEngine, X: np.ndarray, y: np.ndarray) -> str:
    """Save regression model and data to session storage"""
    session_id = generate_unique_id()
    session_dir = Path("data/regression_sessions")
    session_dir.mkdir(parents=True, exist_ok=True)
    
    session_data = {
        'engine': engine,
        'X': X.tolist(),
        'y': y.tolist()
    }
    
    session_path = session_dir / f"{session_id}.pkl"
    with open(session_path, 'wb') as f:
        pickle.dump(session_data, f)
    
    return session_id


def load_regression_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Load regression model from session storage"""
    session_path = Path(f"data/regression_sessions/{session_id}.pkl")
    if not session_path.exists():
        return None
    
    with open(session_path, 'rb') as f:
        return pickle.load(f)


# API Endpoints
@router.post("/train")
async def train_regression(request: RegressionTrainRequest):
    """
    Train a regression model
    
    Request body:
        x: Feature values
        y: Target values
        regression_type: 'linear', 'polynomial', 'ridge', 'lasso', 'elasticnet'
        params: Additional parameters (e.g., degree for polynomial, alpha for ridge/lasso)
    
    Returns:
        session_id: Model session identifier
        metrics: Training metrics (R², MAE, MSE, RMSE)
        coefficients: Model parameters
        regression_line: Points for visualization
    """
    try:
        # Convert to numpy arrays
        X = np.array(request.x)
        y = np.array(request.y)
        
        # Validate data
        if len(X) != len(y):
            raise HTTPException(400, "X and y must have same length")
        
        if len(X) < 2:
            raise HTTPException(400, "Need at least 2 data points")
        
        # Initialize and train model
        engine = RegressionEngine(
            regression_type=request.regression_type,
            **request.params
        )
        engine.fit(X, y)
        
        # Calculate metrics
        metrics = engine.calculate_metrics(X, y)
        
        # Get coefficients
        coefficients = engine.get_coefficients()
        
        # Generate regression line for visualization
        x_min, x_max = float(X.min()), float(X.max())
        regression_line = engine.generate_regression_line(
            x_range=(x_min, x_max),
            num_points=100
        )
        
        # Calculate residuals for diagnostics
        residuals = engine.get_residuals(X, y)
        
        # Save session
        session_id = save_regression_session(engine, X, y)
        
        return {
            'session_id': session_id,
            'regression_type': request.regression_type,
            'metrics': metrics,
            'coefficients': coefficients,
            'regression_line': regression_line,
            'residuals': residuals.tolist(),
            'data_points': {
                'x': X.tolist(),
                'y': y.tolist()
            }
        }
    
    except Exception as e:
        raise HTTPException(500, f"Training failed: {str(e)}")


@router.post("/predict")
async def predict_regression(request: RegressionPredictRequest):
    """
    Make predictions with trained regression model
    
    Request body:
        session_id: Model session identifier
        x: Input values for prediction
    
    Returns:
        predictions: Predicted values
    """
    # Load model from session
    session_data = load_regression_session(request.session_id)
    
    if not session_data:
        raise HTTPException(404, "Session not found")
    
    try:
        engine = session_data['engine']
        X = np.array(request.x)
        predictions = engine.predict(X)
        
        return {
            'predictions': predictions.tolist()
        }
    
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")


@router.get("/types")
async def get_regression_types():
    """Get available regression types and their parameters"""
    return {
        'types': [
            {
                'id': 'linear',
                'name': 'Linear Regression',
                'description': 'Ordinary least squares regression',
                'parameters': []
            },
            {
                'id': 'polynomial',
                'name': 'Polynomial Regression',
                'description': 'Polynomial features with linear regression',
                'parameters': [
                    {
                        'name': 'degree',
                        'type': 'int',
                        'default': 2,
                        'min': 1,
                        'max': 10,
                        'description': 'Degree of polynomial features'
                    }
                ]
            },
            {
                'id': 'ridge',
                'name': 'Ridge Regression',
                'description': 'Linear regression with L2 regularization',
                'parameters': [
                    {
                        'name': 'alpha',
                        'type': 'float',
                        'default': 1.0,
                        'min': 0.0,
                        'max': 100.0,
                        'description': 'Regularization strength'
                    }
                ]
            },
            {
                'id': 'lasso',
                'name': 'Lasso Regression',
                'description': 'Linear regression with L1 regularization',
                'parameters': [
                    {
                        'name': 'alpha',
                        'type': 'float',
                        'default': 1.0,
                        'min': 0.0,
                        'max': 100.0,
                        'description': 'Regularization strength'
                    }
                ]
            },
            {
                'id': 'elasticnet',
                'name': 'ElasticNet Regression',
                'description': 'Linear regression with L1 and L2 regularization',
                'parameters': [
                    {
                        'name': 'alpha',
                        'type': 'float',
                        'default': 1.0,
                        'min': 0.0,
                        'max': 100.0,
                        'description': 'Regularization strength'
                    },
                    {
                        'name': 'l1_ratio',
                        'type': 'float',
                        'default': 0.5,
                        'min': 0.0,
                        'max': 1.0,
                        'description': 'Mix of L1 and L2 (0=L2, 1=L1)'
                    }
                ]
            }
        ]
    }
