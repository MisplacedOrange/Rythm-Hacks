# Feature Documentation: Decision Tree Visualization

## Overview
Interactive decision tree classifier and regressor with visual tree structure, node splitting criteria, feature importance, and pruning capabilities. Provides both classification and regression decision trees with comprehensive visualization.

---

## Current Implementation

### Frontend Component

**Location**: `components/DecisionTreePanel.jsx`

**Features**:
- Static SVG tree visualization
- Hierarchical node structure
- Split conditions displayed on nodes
- Color-coded nodes by class prediction
- Root, internal, and leaf node differentiation

**Current Visualization**:
```javascript
// Simple static SVG tree with:
// - Root node: "root: x < 3.1"
// - Internal nodes with split conditions
// - Leaf nodes with class predictions
// - Orange/blue color scheme matching app theme
```

---

## Backend Integration

### Decision Tree Engine

```python
# backend/app/core/decision_tree.py
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.tree import _tree
from typing import Dict, List, Any, Optional, Tuple
import json

class DecisionTreeEngine:
    """
    Decision tree training and visualization engine
    
    Supports:
        - Classification trees
        - Regression trees
        - Pruning (pre and post)
        - Feature importance
        - Tree structure extraction
    """
    
    def __init__(
        self, 
        task_type: str = 'classification',
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = None,
        **kwargs
    ):
        self.task_type = task_type
        
        # Set default criterion based on task
        if criterion is None:
            criterion = 'gini' if task_type == 'classification' else 'squared_error'
        
        # Initialize appropriate model
        if task_type == 'classification':
            self.model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                criterion=criterion,
                **kwargs
            )
        elif task_type == 'regression':
            self.model = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                criterion=criterion,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeEngine':
        """Train decision tree"""
        self.model.fit(X, y)
        self.feature_names_ = [f"Feature {i}" for i in range(X.shape[1])]
        self.class_names_ = None
        
        if self.task_type == 'classification':
            self.class_names_ = [f"Class {i}" for i in range(len(np.unique(y)))]
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)
    
    def get_tree_structure(self) -> Dict[str, Any]:
        """
        Extract complete tree structure for visualization
        
        Returns hierarchical JSON structure with:
            - Node IDs
            - Split features and thresholds
            - Impurity/value
            - Sample counts
            - Class predictions
        """
        tree = self.model.tree_
        
        def extract_node(node_id: int) -> Dict[str, Any]:
            """Recursively extract node information"""
            # Check if leaf node
            if tree.feature[node_id] == _tree.TREE_UNDEFINED:
                # Leaf node
                if self.task_type == 'classification':
                    class_idx = np.argmax(tree.value[node_id][0])
                    prediction = self.class_names_[class_idx] if self.class_names_ else f"Class {class_idx}"
                else:
                    prediction = float(tree.value[node_id][0][0])
                
                return {
                    'id': int(node_id),
                    'type': 'leaf',
                    'prediction': prediction,
                    'samples': int(tree.n_node_samples[node_id]),
                    'value': tree.value[node_id].tolist(),
                    'impurity': float(tree.impurity[node_id])
                }
            
            # Internal node
            feature_idx = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            
            return {
                'id': int(node_id),
                'type': 'internal',
                'feature': self.feature_names_[feature_idx],
                'feature_index': int(feature_idx),
                'threshold': float(threshold),
                'condition': f"{self.feature_names_[feature_idx]} <= {threshold:.3f}",
                'samples': int(tree.n_node_samples[node_id]),
                'impurity': float(tree.impurity[node_id]),
                'left': extract_node(tree.children_left[node_id]),
                'right': extract_node(tree.children_right[node_id])
            }
        
        return extract_node(0)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        importances = self.model.feature_importances_
        
        return {
            feature: float(importance)
            for feature, importance in zip(self.feature_names_, importances)
        }
    
    def get_tree_depth(self) -> int:
        """Get actual tree depth"""
        return int(self.model.get_depth())
    
    def get_n_leaves(self) -> int:
        """Get number of leaf nodes"""
        return int(self.model.get_n_leaves())
    
    def get_decision_path(self, X: np.ndarray) -> List[List[int]]:
        """
        Get decision path for samples
        
        Returns list of node IDs traversed for each sample
        """
        paths = self.model.decision_path(X)
        
        result = []
        for i in range(X.shape[0]):
            path = paths[i].indices.tolist()
            result.append(path)
        
        return result
    
    def prune_tree(self, ccp_alpha: float = 0.0) -> 'DecisionTreeEngine':
        """
        Prune tree using cost complexity pruning
        
        Args:
            ccp_alpha: Complexity parameter for pruning
        """
        if self.task_type == 'classification':
            self.model = DecisionTreeClassifier(
                ccp_alpha=ccp_alpha,
                max_depth=self.model.max_depth,
                min_samples_split=self.model.min_samples_split,
                min_samples_leaf=self.model.min_samples_leaf
            )
        else:
            self.model = DecisionTreeRegressor(
                ccp_alpha=ccp_alpha,
                max_depth=self.model.max_depth,
                min_samples_split=self.model.min_samples_split,
                min_samples_leaf=self.model.min_samples_leaf
            )
        
        return self
    
    def export_text_rules(self) -> str:
        """Export tree as text rules"""
        return export_text(
            self.model,
            feature_names=self.feature_names_
        )
```

---

### REST API Endpoints

```python
# backend/app/api/routes/decision_tree.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
from backend.app.core.decision_tree import DecisionTreeEngine

router = APIRouter()

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

@router.post("/api/decision-tree/train")
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
        from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
        
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
        session_id = save_tree_session(engine, request.task_type)
        
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

@router.post("/api/decision-tree/predict")
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
    engine = load_tree_session(request.session_id)
    
    if not engine:
        raise HTTPException(404, "Session not found")
    
    try:
        X = np.array(request.X)
        predictions = engine.predict(X)
        decision_paths = engine.get_decision_path(X)
        
        return {
            'predictions': predictions.tolist(),
            'decision_paths': decision_paths
        }
    
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")

@router.get("/api/decision-tree/{session_id}/structure")
async def get_tree_structure(session_id: str):
    """Get complete tree structure for visualization"""
    engine = load_tree_session(session_id)
    
    if not engine:
        raise HTTPException(404, "Session not found")
    
    return {
        'tree_structure': engine.get_tree_structure(),
        'feature_importance': engine.get_feature_importance(),
        'tree_depth': engine.get_tree_depth(),
        'n_leaves': engine.get_n_leaves()
    }

@router.post("/api/decision-tree/{session_id}/prune")
async def prune_tree(session_id: str, ccp_alpha: float = 0.01):
    """
    Prune decision tree using cost complexity pruning
    
    Args:
        ccp_alpha: Complexity parameter (higher = more pruning)
    
    Returns updated tree structure
    """
    engine = load_tree_session(session_id)
    session_data = get_tree_session_data(session_id)
    
    if not engine or not session_data:
        raise HTTPException(404, "Session not found")
    
    # Reload original data
    X = np.array(session_data['X'])
    y = np.array(session_data['y'])
    
    # Create pruned tree
    engine.prune_tree(ccp_alpha)
    engine.fit(X, y)
    
    # Update session
    update_tree_session(session_id, engine)
    
    return {
        'tree_structure': engine.get_tree_structure(),
        'tree_depth': engine.get_tree_depth(),
        'n_leaves': engine.get_n_leaves(),
        'ccp_alpha': ccp_alpha
    }

@router.get("/api/decision-tree/parameters")
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
                'criteria': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
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

# Helper functions
def save_tree_session(engine: DecisionTreeEngine, task_type: str) -> str:
    """Save decision tree model to session"""
    import pickle
    from pathlib import Path
    
    session_id = generate_unique_id()
    session_dir = Path("data/tree_sessions")
    session_dir.mkdir(parents=True, exist_ok=True)
    
    session_path = session_dir / f"{session_id}.pkl"
    with open(session_path, 'wb') as f:
        pickle.dump(engine, f)
    
    return session_id

def load_tree_session(session_id: str) -> Optional[DecisionTreeEngine]:
    """Load decision tree model from session"""
    import pickle
    from pathlib import Path
    
    session_path = Path(f"data/tree_sessions/{session_id}.pkl")
    if not session_path.exists():
        return None
    
    with open(session_path, 'rb') as f:
        return pickle.load(f)
```

---

## Enhanced Frontend Implementation

### Updated Component with API Integration

```javascript
// frontend/src/components/DecisionTreePanel.jsx
import React, { useState, useEffect } from 'react'
import './DecisionTreePanel.css'

export default function DecisionTreePanel() {
  const [treeData, setTreeData] = useState(null)
  const [sessionId, setSessionId] = useState(null)
  const [loading, setLoading] = useState(false)
  const [maxDepth, setMaxDepth] = useState(3)

  // Train decision tree
  const trainTree = async () => {
    setLoading(true)
    
    // Generate sample classification data
    const X = []
    const y = []
    
    for (let i = 0; i < 100; i++) {
      const x1 = Math.random() * 10
      const x2 = Math.random() * 10
      X.push([x1, x2])
      
      // Simple decision boundary
      if (x1 < 5) {
        y.push(x2 < 5 ? 0 : 1)
      } else {
        y.push(x2 < 5 ? 1 : 0)
      }
    }

    try {
      const response = await fetch('http://localhost:8000/api/decision-tree/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          X: X,
          y: y,
          task_type: 'classification',
          max_depth: maxDepth,
          feature_names: ['Feature 1', 'Feature 2'],
          class_names: ['Class A', 'Class B']
        })
      })

      const result = await response.json()
      setTreeData(result)
      setSessionId(result.session_id)
    } catch (error) {
      console.error('Training failed:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    trainTree()
  }, [maxDepth])

  if (loading || !treeData) {
    return <div>Loading tree...</div>
  }

  return (
    <div className="decision-tree-panel">
      <div className="tree-controls">
        <label>
          Max Depth:
          <input
            type="number"
            value={maxDepth}
            onChange={(e) => setMaxDepth(parseInt(e.target.value))}
            min="1"
            max="10"
          />
        </label>
        <button onClick={trainTree}>Retrain</button>
        <div className="tree-stats">
          <span>Depth: {treeData.tree_depth}</span>
          <span>Leaves: {treeData.n_leaves}</span>
          <span>Accuracy: {(treeData.metrics.accuracy * 100).toFixed(1)}%</span>
        </div>
      </div>

      <div className="tree-visualization">
        <TreeNode node={treeData.tree_structure} />
      </div>

      <div className="feature-importance">
        <h4>Feature Importance</h4>
        {Object.entries(treeData.feature_importance).map(([feature, importance]) => (
          <div key={feature} className="importance-bar">
            <span>{feature}</span>
            <div className="bar">
              <div 
                className="bar-fill" 
                style={{ width: `${importance * 100}%` }}
              />
            </div>
            <span>{(importance * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  )
}

// Recursive tree node component
function TreeNode({ node, level = 0 }) {
  if (node.type === 'leaf') {
    return (
      <div className="tree-node leaf-node" style={{ marginLeft: level * 40 }}>
        <div className="node-content">
          <strong>{node.prediction}</strong>
          <div className="node-stats">
            Samples: {node.samples}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="tree-node internal-node" style={{ marginLeft: level * 40 }}>
      <div className="node-content">
        <div className="node-condition">{node.condition}</div>
        <div className="node-stats">
          Samples: {node.samples} | Impurity: {node.impurity.toFixed(3)}
        </div>
      </div>
      <div className="node-children">
        <div className="branch left-branch">
          <span className="branch-label">True</span>
          <TreeNode node={node.left} level={level + 1} />
        </div>
        <div className="branch right-branch">
          <span className="branch-label">False</span>
          <TreeNode node={node.right} level={level + 1} />
        </div>
      </div>
    </div>
  )
}
```

---

## Styling

```css
/* frontend/src/components/DecisionTreePanel.css */
.decision-tree-panel {
  background: #fff;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  padding: 16px;
}

.tree-controls {
  display: flex;
  gap: 16px;
  align-items: center;
  margin-bottom: 16px;
  padding: 12px;
  background: #f8f9fa;
  border-radius: 6px;
}

.tree-controls label {
  display: flex;
  align-items: center;
  gap: 8px;
}

.tree-controls input {
  padding: 6px 10px;
  border: 1px solid #ddd;
  border-radius: 4px;
  width: 80px;
}

.tree-controls button {
  padding: 8px 16px;
  background: #4682B4;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.tree-stats {
  display: flex;
  gap: 16px;
  margin-left: auto;
  font-size: 14px;
  color: #666;
}

.tree-visualization {
  overflow-x: auto;
  padding: 20px;
  border: 1px solid #e0e0e0;
  border-radius: 6px;
  margin-bottom: 16px;
}

.tree-node {
  margin: 10px 0;
}

.node-content {
  padding: 12px 16px;
  border-radius: 6px;
  display: inline-block;
  min-width: 200px;
}

.internal-node .node-content {
  background: #4682B4;
  color: white;
}

.leaf-node .node-content {
  background: #F4A460;
  color: #000;
}

.node-condition {
  font-weight: 600;
  margin-bottom: 4px;
}

.node-stats {
  font-size: 12px;
  opacity: 0.9;
}

.node-children {
  margin-left: 20px;
  border-left: 2px solid #ddd;
  padding-left: 20px;
}

.branch {
  margin: 10px 0;
}

.branch-label {
  font-size: 12px;
  color: #666;
  font-weight: 600;
  margin-right: 8px;
}

.feature-importance {
  padding-top: 16px;
  border-top: 2px solid #e0e0e0;
}

.feature-importance h4 {
  margin-bottom: 12px;
  font-size: 14px;
}

.importance-bar {
  display: grid;
  grid-template-columns: 120px 1fr 60px;
  align-items: center;
  gap: 12px;
  margin-bottom: 8px;
}

.bar {
  height: 20px;
  background: #f0f0f0;
  border-radius: 4px;
  overflow: hidden;
}

.bar-fill {
  height: 100%;
  background: linear-gradient(90deg, #4682B4, #5B9BD5);
  transition: width 0.3s;
}
```

---

## Database Schema

```python
# backend/app/models/decision_tree.py
from sqlalchemy import Column, String, Integer, JSON, DateTime
from datetime import datetime

class DecisionTreeSession(Base):
    __tablename__ = "decision_tree_sessions"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=True)
    task_type = Column(String, nullable=False)
    max_depth = Column(Integer, nullable=True)
    tree_depth = Column(Integer, nullable=False)
    n_leaves = Column(Integer, nullable=False)
    feature_importance = Column(JSON, nullable=False)
    metrics = Column(JSON, nullable=False)
    model_path = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
```

---

## Future Enhancements

1. **Random Forest Integration**: Ensemble of decision trees
2. **Gradient Boosting**: XGBoost, LightGBM integration
3. **Interactive Pruning**: Click nodes to prune
4. **Path Highlighting**: Show decision path for specific samples
5. **Export Tree**: Save as PNG/SVG/PDF
6. **Rule Extraction**: Generate if-then rules
7. **Feature Engineering**: Auto-suggest feature combinations
8. **Cross-Validation**: K-fold CV for tree validation
9. **Ensemble Voting**: Combine multiple trees
10. **Real-time Training**: WebSocket streaming for large datasets
