"""
Regression Engine for MediLytica
Supports Linear, Polynomial, Ridge, Lasso, and ElasticNet regression
Based on feature-regression-analysis.md documentation
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Any, Optional, Tuple


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
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else r2
        
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
