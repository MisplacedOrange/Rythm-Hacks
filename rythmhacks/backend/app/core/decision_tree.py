"""
Decision Tree Engine for MediLytica
Supports Classification and Regression Trees
Based on feature-decision-tree.md documentation
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.tree import _tree
from typing import Dict, List, Any, Optional


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
    
    def export_text_rules(self) -> str:
        """Export tree as text rules"""
        return export_text(
            self.model,
            feature_names=self.feature_names_
        )
