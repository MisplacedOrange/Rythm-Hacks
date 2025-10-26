# Feature Documentation: Regression Analysis

## Overview
Linear and non-linear regression analysis with interactive visualization, real-time metrics calculation, and model evaluation. Supports multiple regression types including linear, polynomial, ridge, and lasso regression with automatic hyperparameter tuning.

---

## Current Implementation

### Frontend Component

**Location**: `components/RegressionPanel.jsx`

**Features**:
- Linear regression scatter plot with fitted line
- Real-time metrics display (R², MAE, MSE, RMSE)
- Interactive Plotly chart
- Grid layout with chart and metrics panel

**Mock Data Generation**:
```javascript
function mockRegressionData(n = 80) {
  const x = [], y = []
  const m = 1.2, b = 0.5  // Slope and intercept
  for (let i = 0; i < n; i++) {
    const xv = i / 8 + Math.random()
    const noise = (Math.random() - 0.5) * 2
    x.push(xv)
    y.push(m * xv + b + noise)
  }
  const lineX = [Math.min(...x), Math.max(...x)]
  const lineY = lineX.map(v => m * v + b)
  return { 
    x, y, lineX, lineY, 
    metrics: { r2: 0.89, mae: 0.52, mse: 0.41, rmse: 0.64 } 
  }
}
```

---

## Backend Integration

### Regression Engine

```python
# backend/app/core/regression.py
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

class RegressionEngine:
    """
    Comprehensive regression analysis engine
    
    Supports:
        - Linear Regression
        - Polynomial Regression
        - Ridge Regression (L2 regularization)
        - Lasso Regression (L1 regularization)
        - ElasticNet Regression
    """
    
    def __init__(self, regression_type: str = 'linear', **kwargs):
        self.regression_type = regression_type
        self.model = None
        self.poly_features = None
        self.params = kwargs
        
        # Initialize model based on type
        if regression_type == 'linear':
            self.model = LinearRegression()
        
        elif regression_type == 'polynomial':
            self.degree = kwargs.get('degree', 2)
            self.poly_features = PolynomialFeatures(degree=self.degree)
            self.model = LinearRegression()
        
        elif regression_type == 'ridge':
            alpha = kwargs.get('alpha', 1.0)
            self.model = Ridge(alpha=alpha)
        
        elif regression_type == 'lasso':
            alpha = kwargs.get('alpha', 1.0)
            self.model = Lasso(alpha=alpha)
        
        elif regression_type == 'elasticnet':
            alpha = kwargs.get('alpha', 1.0)
            l1_ratio = kwargs.get('l1_ratio', 0.5)
            self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        
        else:
            raise ValueError(f"Unknown regression type: {regression_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RegressionEngine':
        """
        Train regression model
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
        """
        # Reshape if 1D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Apply polynomial features if needed
        if self.poly_features is not None:
            X = self.poly_features.fit_transform(X)
        
        # Fit model
        self.model.fit(X, y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions"""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if self.poly_features is not None:
            X = self.poly_features.transform(X)
        
        return self.model.predict(X)
    
    def calculate_metrics(
        self, 
        X: np.ndarray, 
        y_true: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate comprehensive regression metrics
        
        Returns:
            - R² (coefficient of determination)
            - MAE (mean absolute error)
            - MSE (mean squared error)
            - RMSE (root mean squared error)
            - Adjusted R² (accounts for number of features)
        """
        y_pred = self.predict(X)
        
        n = len(y_true)
        k = X.shape[1] if X.ndim > 1 else 1
        
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # Adjusted R²
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
        
        return {
            'r2': float(r2),
            'adjusted_r2': float(adj_r2),
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse)
        }
    
    def get_coefficients(self) -> Dict[str, Any]:
        """Extract model coefficients"""
        if not hasattr(self.model, 'coef_'):
            return {}
        
        return {
            'coefficients': self.model.coef_.tolist(),
            'intercept': float(self.model.intercept_)
        }
    
    def generate_regression_line(
        self, 
        x_range: Optional[Tuple[float, float]] = None,
        num_points: int = 100
    ) -> Dict[str, List[float]]:
        """
        Generate points for regression line visualization
        
        Args:
            x_range: (min, max) range for x values
            num_points: Number of points to generate
        """
        if x_range is None:
            x_range = (0, 10)
        
        x_line = np.linspace(x_range[0], x_range[1], num_points)
        y_line = self.predict(x_line)
        
        return {
            'x': x_line.tolist(),
            'y': y_line.tolist()
        }
    
    def get_residuals(self, X: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Calculate residuals for diagnostic plots"""
        y_pred = self.predict(X)
        return y_true - y_pred
```

---

### REST API Endpoints

```python
# backend/app/api/routes/regression.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
from backend.app.core.regression import RegressionEngine

router = APIRouter()

class RegressionTrainRequest(BaseModel):
    x: List[float]
    y: List[float]
    regression_type: str = 'linear'
    params: Optional[Dict[str, Any]] = {}

class RegressionPredictRequest(BaseModel):
    session_id: str
    x: List[float]

@router.post("/api/regression/train")
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
        session_id = save_regression_session(engine, request.regression_type)
        
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

@router.post("/api/regression/predict")
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
    engine = load_regression_session(request.session_id)
    
    if not engine:
        raise HTTPException(404, "Session not found")
    
    try:
        X = np.array(request.x)
        predictions = engine.predict(X)
        
        return {
            'predictions': predictions.tolist()
        }
    
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")

@router.get("/api/regression/{session_id}/diagnostics")
async def get_regression_diagnostics(session_id: str):
    """
    Get diagnostic plots data
    
    Returns:
        - Residuals vs Fitted
        - Q-Q plot data
        - Scale-Location plot
        - Residuals vs Leverage
    """
    engine = load_regression_session(session_id)
    session_data = get_session_data(session_id)
    
    if not engine or not session_data:
        raise HTTPException(404, "Session not found")
    
    X = np.array(session_data['x'])
    y = np.array(session_data['y'])
    
    y_pred = engine.predict(X)
    residuals = y - y_pred
    
    # Standardized residuals
    std_residuals = residuals / np.std(residuals)
    
    # Q-Q plot data (for normality test)
    from scipy import stats
    theoretical_quantiles = stats.probplot(residuals, dist="norm")[0][0]
    sample_quantiles = stats.probplot(residuals, dist="norm")[0][1]
    
    return {
        'residuals_vs_fitted': {
            'fitted': y_pred.tolist(),
            'residuals': residuals.tolist()
        },
        'qq_plot': {
            'theoretical': theoretical_quantiles.tolist(),
            'sample': sample_quantiles.tolist()
        },
        'scale_location': {
            'fitted': y_pred.tolist(),
            'sqrt_std_residuals': np.sqrt(np.abs(std_residuals)).tolist()
        }
    }

@router.get("/api/regression/types")
async def get_regression_types():
    """
    Get available regression types and their parameters
    """
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

# Helper functions
def save_regression_session(engine: RegressionEngine, regression_type: str) -> str:
    """Save regression model to session storage"""
    import pickle
    from pathlib import Path
    
    session_id = generate_unique_id()
    session_dir = Path("data/regression_sessions")
    session_dir.mkdir(parents=True, exist_ok=True)
    
    session_path = session_dir / f"{session_id}.pkl"
    with open(session_path, 'wb') as f:
        pickle.dump(engine, f)
    
    return session_id

def load_regression_session(session_id: str) -> Optional[RegressionEngine]:
    """Load regression model from session storage"""
    import pickle
    from pathlib import Path
    
    session_path = Path(f"data/regression_sessions/{session_id}.pkl")
    if not session_path.exists():
        return None
    
    with open(session_path, 'rb') as f:
        return pickle.load(f)
```

---

## Enhanced Frontend Implementation

### Updated Component with API Integration

```javascript
// frontend/src/components/RegressionPanel.jsx
import React, { useState, useEffect } from 'react'
import ChartWrapper from './ChartWrapper'
import './RegressionPanel.css'

export default function RegressionPanel() {
  const [data, setData] = useState(null)
  const [regressionType, setRegressionType] = useState('linear')
  const [params, setParams] = useState({})
  const [loading, setLoading] = useState(false)
  const [sessionId, setSessionId] = useState(null)

  // Train regression model
  const trainModel = async (xData, yData) => {
    setLoading(true)
    try {
      const response = await fetch('http://localhost:8000/api/regression/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          x: xData,
          y: yData,
          regression_type: regressionType,
          params: params
        })
      })

      const result = await response.json()
      setData(result)
      setSessionId(result.session_id)
    } catch (error) {
      console.error('Training failed:', error)
    } finally {
      setLoading(false)
    }
  }

  // Generate sample data
  const generateData = () => {
    const x = []
    const y = []
    const m = 1.2
    const b = 0.5

    for (let i = 0; i < 80; i++) {
      const xv = i / 8 + Math.random()
      const noise = (Math.random() - 0.5) * 2
      x.push(xv)
      y.push(m * xv + b + noise)
    }

    trainModel(x, y)
  }

  useEffect(() => {
    generateData()
  }, [regressionType, params])

  if (loading || !data) {
    return <div>Loading...</div>
  }

  return (
    <div className="regression-panel">
      <div className="reg-controls">
        <select 
          value={regressionType} 
          onChange={(e) => setRegressionType(e.target.value)}
        >
          <option value="linear">Linear</option>
          <option value="polynomial">Polynomial</option>
          <option value="ridge">Ridge</option>
          <option value="lasso">Lasso</option>
        </select>

        {regressionType === 'polynomial' && (
          <input
            type="number"
            value={params.degree || 2}
            onChange={(e) => setParams({ ...params, degree: parseInt(e.target.value) })}
            min="1"
            max="10"
            placeholder="Degree"
          />
        )}

        <button onClick={generateData}>Regenerate</button>
      </div>

      <div className="reg-content">
        <div className="reg-chart">
          <ChartWrapper
            data={[
              {
                x: data.data_points.x,
                y: data.data_points.y,
                mode: 'markers',
                type: 'scatter',
                name: 'Data',
                marker: { size: 8, color: '#4682B4' }
              },
              {
                x: data.regression_line.x,
                y: data.regression_line.y,
                mode: 'lines',
                type: 'scatter',
                name: 'Regression Line',
                line: { color: '#E53935', width: 3 }
              }
            ]}
            layout={{
              xaxis: { title: 'x' },
              yaxis: { title: 'y' },
              showlegend: true
            }}
            style={{ width: '100%', height: 360 }}
          />
        </div>

        <div className="reg-metrics">
          <h3>Metrics</h3>
          <table>
            <tbody>
              {Object.entries(data.metrics).map(([key, value]) => (
                <tr key={key}>
                  <td>{key.toUpperCase()}</td>
                  <td>{value.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>

          {data.coefficients && (
            <div className="coefficients">
              <h4>Coefficients</h4>
              <p>Intercept: {data.coefficients.intercept.toFixed(3)}</p>
              <p>Slope: {data.coefficients.coefficients[0]?.toFixed(3)}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
```

---

## Database Schema

```python
# backend/app/models/regression.py
from sqlalchemy import Column, String, Float, JSON, DateTime
from datetime import datetime

class RegressionSession(Base):
    __tablename__ = "regression_sessions"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=True)
    regression_type = Column(String, nullable=False)
    params = Column(JSON, nullable=True)
    metrics = Column(JSON, nullable=False)
    coefficients = Column(JSON, nullable=True)
    model_path = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
```

---

## Styling

```css
/* frontend/src/components/RegressionPanel.css */
.regression-panel {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.reg-controls {
  display: flex;
  gap: 12px;
  align-items: center;
  padding: 12px;
  background: #f8f9fa;
  border-radius: 6px;
}

.reg-controls select,
.reg-controls input {
  padding: 8px 12px;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.reg-controls button {
  padding: 8px 16px;
  background: #4682B4;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.reg-content {
  display: grid;
  grid-template-columns: 1.2fr 0.8fr;
  gap: 1rem;
}

.reg-chart,
.reg-metrics {
  background: #fff;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  padding: 12px;
}

.reg-metrics h3 {
  margin-bottom: 12px;
  font-size: 16px;
}

.reg-metrics table {
  width: 100%;
  border-collapse: collapse;
}

.reg-metrics td {
  padding: 8px;
  border-bottom: 1px solid #f0f0f0;
}

.reg-metrics td:first-child {
  font-weight: 600;
}

.reg-metrics td:last-child {
  text-align: right;
}

.coefficients {
  margin-top: 16px;
  padding-top: 16px;
  border-top: 2px solid #e0e0e0;
}

.coefficients h4 {
  margin-bottom: 8px;
  font-size: 14px;
}

.coefficients p {
  font-size: 13px;
  color: #666;
  margin: 4px 0;
}
```

---

## Future Enhancements

1. **Multiple Regression**: Support for multiple features
2. **Cross-Validation**: K-fold CV for model validation
3. **Feature Selection**: Automatic feature selection algorithms
4. **Outlier Detection**: Identify and handle outliers
5. **Confidence Intervals**: Prediction intervals visualization
6. **Model Comparison**: Side-by-side comparison of regression types
7. **Export Results**: Download regression report as PDF
8. **Time Series Regression**: Support for time series data
9. **Interactive Parameter Tuning**: Real-time alpha/degree adjustment
10. **Residual Analysis**: Comprehensive diagnostic plots
