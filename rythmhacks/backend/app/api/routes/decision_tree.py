"""
Decision Tree API Routes
Endpoints for training and predicting with decision tree models
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

from app.core.decision_tree import DecisionTreeEngine
from app.utils.helpers import generate_unique_id

router = APIRouter(prefix="/api/decision-tree")


# Pydantic Models
class DecisionTreeTrainRequest(BaseModel):
    X: List[List[float]]
    y: List[float]
    task_type: str = 'classification'  # 'classification' or 'regression'
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    criterion: Optional[str] = None
    feature_names: Optional[List[str]] = None
    class_names: Optional[List[str]] = None


class DecisionTreePredictRequest(BaseModel):
    session_id: str
    X: List[List[float]]


# Helper Functions
def save_tree_session(engine: DecisionTreeEngine, X: np.ndarray, y: np.ndarray, task_type: str) -> str:
    """Save decision tree model to session"""
    session_id = generate_unique_id()
    session_dir = Path("data/tree_sessions")
    session_dir.mkdir(parents=True, exist_ok=True)
    
    session_data = {
        'engine': engine,
        'X': X.tolist(),
        'y': y.tolist(),
        'task_type': task_type
    }
    
    session_path = session_dir / f"{session_id}.pkl"
    with open(session_path, 'wb') as f:
        pickle.dump(session_data, f)
    
    return session_id


def load_tree_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Load decision tree model from session"""
    session_path = Path(f"data/tree_sessions/{session_id}.pkl")
    if not session_path.exists():
        return None
    
    with open(session_path, 'rb') as f:
        return pickle.load(f)


# API Endpoints
@router.post("/train")
async def train_decision_tree(request: DecisionTreeTrainRequest):
    """
    Train a decision tree model
    
    Request body:
        X: Feature matrix
        y: Target values
        task_type: 'classification' or 'regression'
        max_depth: Maximum tree depth (None for unlimited)
        min_samples_split: Minimum samples to split node
        min_samples_leaf: Minimum samples in leaf node
        criterion: Split criterion ('gini', 'entropy', 'squared_error', etc.)
        feature_names: Optional feature names
        class_names: Optional class names (classification only)
    
    Returns:
        session_id: Model session identifier
        tree_structure: Complete tree hierarchy
        feature_importance: Feature importance scores
        metrics: Model performance metrics
        tree_depth: Actual tree depth
        n_leaves: Number of leaf nodes
    """
    try:
        # Convert to numpy arrays
        X = np.array(request.X)
        y = np.array(request.y)
        
        # Validate data
        if X.shape[0] != len(y):
            raise HTTPException(400, "X and y must have same number of samples")
        
        if X.shape[0] < 2:
            raise HTTPException(400, "Need at least 2 samples")
        
        # Initialize and train model
        engine = DecisionTreeEngine(
            task_type=request.task_type,
            max_depth=request.max_depth,
            min_samples_split=request.min_samples_split,
            min_samples_leaf=request.min_samples_leaf,
            criterion=request.criterion
        )
        
        engine.fit(X, y)
        
        # Set custom feature/class names if provided
        if request.feature_names:
            engine.feature_names_ = request.feature_names
        if request.class_names and request.task_type == 'classification':
            engine.class_names_ = request.class_names
        
        # Extract tree structure
        tree_structure = engine.get_tree_structure()
        
        # Get feature importance
        feature_importance = engine.get_feature_importance()
        
        # Calculate metrics
        y_pred = engine.predict(X)
        
        if request.task_type == 'classification':
            metrics = {
                'accuracy': float(accuracy_score(y, y_pred)),
                'train_score': float(engine.model.score(X, y))
            }
        else:
            metrics = {
                'mse': float(mean_squared_error(y, y_pred)),
                'r2': float(r2_score(y, y_pred)),
                'train_score': float(engine.model.score(X, y))
            }
        
        # Save session
        session_id = save_tree_session(engine, X, y, request.task_type)
        
        return {
            'session_id': session_id,
            'task_type': request.task_type,
            'tree_structure': tree_structure,
            'feature_importance': feature_importance,
            'metrics': metrics,
            'tree_depth': engine.get_tree_depth(),
            'n_leaves': engine.get_n_leaves(),
            'text_rules': engine.export_text_rules()
        }
    
    except Exception as e:
        raise HTTPException(500, f"Training failed: {str(e)}")


@router.post("/predict")
async def predict_decision_tree(request: DecisionTreePredictRequest):
    """
    Make predictions with trained decision tree
    
    Request body:
        session_id: Model session identifier
        X: Input features
    
    Returns:
        predictions: Predicted values/classes
        decision_paths: Node IDs traversed for each sample
    """
    session_data = load_tree_session(request.session_id)
    
    if not session_data:
        raise HTTPException(404, "Session not found")
    
    try:
        engine = session_data['engine']
        X = np.array(request.X)
        predictions = engine.predict(X)
        decision_paths = engine.get_decision_path(X)
        
        return {
            'predictions': predictions.tolist(),
            'decision_paths': decision_paths
        }
    
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")


@router.get("/{session_id}/structure")
async def get_tree_structure(session_id: str):
    """Get complete tree structure for visualization"""
    session_data = load_tree_session(session_id)
    
    if not session_data:
        raise HTTPException(404, "Session not found")
    
    engine = session_data['engine']
    
    return {
        'tree_structure': engine.get_tree_structure(),
        'feature_importance': engine.get_feature_importance(),
        'tree_depth': engine.get_tree_depth(),
        'n_leaves': engine.get_n_leaves()
    }


@router.get("/parameters")
async def get_tree_parameters():
    """Get available decision tree parameters"""
    return {
        'task_types': [
            {
                'id': 'classification',
                'name': 'Classification',
                'criteria': ['gini', 'entropy', 'log_loss']
            },
            {
                'id': 'regression',
                'name': 'Regression',
                'criteria': ['squared_error', 'friedman_mse', 'absolute_error']
            }
        ],
        'parameters': [
            {
                'name': 'max_depth',
                'type': 'int',
                'default': None,
                'min': 1,
                'max': 50,
                'description': 'Maximum depth of tree (None for unlimited)'
            },
            {
                'name': 'min_samples_split',
                'type': 'int',
                'default': 2,
                'min': 2,
                'max': 100,
                'description': 'Minimum samples required to split node'
            },
            {
                'name': 'min_samples_leaf',
                'type': 'int',
                'default': 1,
                'min': 1,
                'max': 50,
                'description': 'Minimum samples required in leaf node'
            }
        ]
    }
