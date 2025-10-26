# Feature Documentation: Sidebar Algorithm Navigation

## Overview
Hierarchical sidebar navigation for browsing and selecting ML algorithms across different categories (Neural Networks, Traditional ML, Ensemble Methods). Enables quick algorithm switching and maintains training configuration context.

---

## Current Implementation

### Frontend Component

**Location**: `Dashboard.jsx` (integrated)

**State Management**:
```javascript
const [selectedAlgorithm, setSelectedAlgorithm] = useState('Neural Network')
const [expandedCategories, setExpandedCategories] = useState({
  'Neural Networks': true,
  'Traditional ML': false,
  'Ensemble Methods': false
})
```

**Algorithm Categories**:
```javascript
const algorithms = {
  'Neural Networks': [
    'Feedforward NN',
    'Convolutional NN',
    'Recurrent NN',
    'Transformer',
    'Autoencoder'
  ],
  'Traditional ML': [
    'Logistic Regression',
    'Decision Tree',
    'K-Nearest Neighbors',
    'Support Vector Machine',
    'Naive Bayes'
  ],
  'Ensemble Methods': [
    'Random Forest',
    'Gradient Boosting',
    'AdaBoost',
    'XGBoost',
    'LightGBM'
  ]
}
```

---

## Backend Integration

### Algorithm Registry System

```python
# backend/app/core/algorithm_registry.py
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from enum import Enum
import importlib

class AlgorithmCategory(str, Enum):
    NEURAL_NETWORKS = "Neural Networks"
    TRADITIONAL_ML = "Traditional ML"
    ENSEMBLE_METHODS = "Ensemble Methods"
    PREPROCESSING = "Preprocessing"
    DIMENSIONALITY_REDUCTION = "Dimensionality Reduction"

class AlgorithmType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"

class HyperparameterSchema(BaseModel):
    name: str
    type: str  # 'int', 'float', 'categorical', 'bool'
    default: Any
    min: Optional[float] = None
    max: Optional[float] = None
    options: Optional[List[Any]] = None  # For categorical
    description: str
    advanced: bool = False  # Hide in basic mode

class AlgorithmMetadata(BaseModel):
    id: str
    name: str
    display_name: str
    category: AlgorithmCategory
    algorithm_type: List[AlgorithmType]
    description: str
    hyperparameters: List[HyperparameterSchema]
    requires_gpu: bool = False
    supports_multiclass: bool = True
    supports_regression: bool = False
    module_path: str  # Python module to import
    class_name: str   # Class to instantiate
    documentation_url: Optional[str] = None
    complexity: str = "medium"  # 'low', 'medium', 'high'
    training_time: str = "medium"  # 'fast', 'medium', 'slow'

class AlgorithmRegistry:
    """Central registry for all available ML algorithms"""
    
    def __init__(self):
        self._algorithms: Dict[str, AlgorithmMetadata] = {}
        self._load_algorithms()
    
    def _load_algorithms(self):
        """Load all algorithm definitions"""
        
        # Neural Networks
        self.register(AlgorithmMetadata(
            id="feedforward_nn",
            name="feedforward_nn",
            display_name="Feedforward Neural Network",
            category=AlgorithmCategory.NEURAL_NETWORKS,
            algorithm_type=[AlgorithmType.CLASSIFICATION, AlgorithmType.REGRESSION],
            description="Multi-layer perceptron with configurable architecture",
            hyperparameters=[
                HyperparameterSchema(
                    name="hidden_layers",
                    type="list[int]",
                    default=[64, 32],
                    description="Number of neurons in each hidden layer"
                ),
                HyperparameterSchema(
                    name="learning_rate",
                    type="float",
                    default=0.001,
                    min=1e-6,
                    max=1.0,
                    description="Learning rate for optimizer"
                ),
                HyperparameterSchema(
                    name="activation",
                    type="categorical",
                    default="relu",
                    options=["relu", "tanh", "sigmoid", "leaky_relu", "elu"],
                    description="Activation function for hidden layers"
                ),
                HyperparameterSchema(
                    name="optimizer",
                    type="categorical",
                    default="adam",
                    options=["adam", "sgd", "rmsprop", "adagrad"],
                    description="Optimization algorithm"
                ),
                HyperparameterSchema(
                    name="dropout_rate",
                    type="float",
                    default=0.0,
                    min=0.0,
                    max=0.9,
                    description="Dropout rate for regularization",
                    advanced=True
                ),
                HyperparameterSchema(
                    name="batch_size",
                    type="int",
                    default=32,
                    min=1,
                    max=512,
                    description="Training batch size"
                ),
                HyperparameterSchema(
                    name="epochs",
                    type="int",
                    default=100,
                    min=1,
                    max=10000,
                    description="Number of training epochs"
                ),
            ],
            module_path="backend.models.neural_networks",
            class_name="FeedforwardNN",
            complexity="medium",
            training_time="medium"
        ))
        
        # Random Forest
        self.register(AlgorithmMetadata(
            id="random_forest",
            name="random_forest",
            display_name="Random Forest",
            category=AlgorithmCategory.ENSEMBLE_METHODS,
            algorithm_type=[AlgorithmType.CLASSIFICATION, AlgorithmType.REGRESSION],
            description="Ensemble of decision trees with bagging",
            hyperparameters=[
                HyperparameterSchema(
                    name="n_estimators",
                    type="int",
                    default=100,
                    min=1,
                    max=1000,
                    description="Number of trees in the forest"
                ),
                HyperparameterSchema(
                    name="max_depth",
                    type="int",
                    default=None,
                    min=1,
                    max=100,
                    description="Maximum depth of trees (None for unlimited)"
                ),
                HyperparameterSchema(
                    name="min_samples_split",
                    type="int",
                    default=2,
                    min=2,
                    max=100,
                    description="Minimum samples required to split node"
                ),
                HyperparameterSchema(
                    name="criterion",
                    type="categorical",
                    default="gini",
                    options=["gini", "entropy", "log_loss"],
                    description="Function to measure split quality"
                ),
            ],
            module_path="sklearn.ensemble",
            class_name="RandomForestClassifier",
            complexity="low",
            training_time="fast"
        ))
        
        # Add more algorithms...
    
    def register(self, algorithm: AlgorithmMetadata):
        """Register an algorithm"""
        self._algorithms[algorithm.id] = algorithm
    
    def get(self, algorithm_id: str) -> Optional[AlgorithmMetadata]:
        """Get algorithm metadata by ID"""
        return self._algorithms.get(algorithm_id)
    
    def list_by_category(self, category: AlgorithmCategory) -> List[AlgorithmMetadata]:
        """List all algorithms in a category"""
        return [
            algo for algo in self._algorithms.values()
            if algo.category == category
        ]
    
    def list_all(self) -> List[AlgorithmMetadata]:
        """List all algorithms"""
        return list(self._algorithms.values())
    
    def get_categories(self) -> List[AlgorithmCategory]:
        """Get all available categories"""
        return list(AlgorithmCategory)
    
    def instantiate(self, algorithm_id: str, hyperparameters: Dict[str, Any]):
        """Dynamically instantiate an algorithm with given hyperparameters"""
        metadata = self.get(algorithm_id)
        if not metadata:
            raise ValueError(f"Algorithm {algorithm_id} not found")
        
        # Import module
        module = importlib.import_module(metadata.module_path)
        algorithm_class = getattr(module, metadata.class_name)
        
        # Validate and prepare hyperparameters
        validated_params = self._validate_hyperparameters(metadata, hyperparameters)
        
        # Instantiate
        return algorithm_class(**validated_params)
    
    def _validate_hyperparameters(
        self, 
        metadata: AlgorithmMetadata, 
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate hyperparameters against schema"""
        validated = {}
        
        for param_schema in metadata.hyperparameters:
            value = params.get(param_schema.name, param_schema.default)
            
            # Type validation
            if param_schema.type == "int":
                value = int(value)
                if param_schema.min is not None:
                    value = max(value, int(param_schema.min))
                if param_schema.max is not None:
                    value = min(value, int(param_schema.max))
            
            elif param_schema.type == "float":
                value = float(value)
                if param_schema.min is not None:
                    value = max(value, param_schema.min)
                if param_schema.max is not None:
                    value = min(value, param_schema.max)
            
            elif param_schema.type == "categorical":
                if value not in param_schema.options:
                    raise ValueError(
                        f"{param_schema.name} must be one of {param_schema.options}"
                    )
            
            validated[param_schema.name] = value
        
        return validated

# Global registry instance
algorithm_registry = AlgorithmRegistry()
```

---

### REST API Endpoints

```python
# backend/app/api/routes/algorithms.py
from fastapi import APIRouter, HTTPException
from typing import List, Optional
from backend.app.core.algorithm_registry import (
    algorithm_registry,
    AlgorithmCategory,
    AlgorithmMetadata
)

router = APIRouter()

@router.get("/api/algorithms")
async def list_algorithms(
    category: Optional[AlgorithmCategory] = None,
    algorithm_type: Optional[str] = None
):
    """
    Get list of all available algorithms
    
    Query params:
        category: Filter by category
        algorithm_type: Filter by type (classification, regression, etc.)
    """
    if category:
        algorithms = algorithm_registry.list_by_category(category)
    else:
        algorithms = algorithm_registry.list_all()
    
    # Filter by algorithm type if specified
    if algorithm_type:
        algorithms = [
            algo for algo in algorithms
            if algorithm_type in [t.value for t in algo.algorithm_type]
        ]
    
    return {
        'algorithms': [
            {
                'id': algo.id,
                'name': algo.display_name,
                'category': algo.category.value,
                'description': algo.description,
                'types': [t.value for t in algo.algorithm_type],
                'complexity': algo.complexity,
                'training_time': algo.training_time,
                'requires_gpu': algo.requires_gpu
            }
            for algo in algorithms
        ]
    }

@router.get("/api/algorithms/categories")
async def list_categories():
    """Get all algorithm categories"""
    categories = algorithm_registry.get_categories()
    
    # Count algorithms per category
    category_counts = {}
    for category in categories:
        algorithms = algorithm_registry.list_by_category(category)
        category_counts[category.value] = len(algorithms)
    
    return {
        'categories': [
            {
                'name': cat.value,
                'count': category_counts[cat.value],
                'id': cat.name.lower()
            }
            for cat in categories
        ]
    }

@router.get("/api/algorithms/{algorithm_id}")
async def get_algorithm_details(algorithm_id: str):
    """
    Get detailed information about a specific algorithm
    
    Returns:
        - Full metadata
        - Hyperparameter schemas
        - Documentation links
        - Example usage
    """
    metadata = algorithm_registry.get(algorithm_id)
    
    if not metadata:
        raise HTTPException(404, f"Algorithm {algorithm_id} not found")
    
    return {
        'id': metadata.id,
        'name': metadata.display_name,
        'category': metadata.category.value,
        'description': metadata.description,
        'algorithm_types': [t.value for t in metadata.algorithm_type],
        'hyperparameters': [
            {
                'name': hp.name,
                'type': hp.type,
                'default': hp.default,
                'min': hp.min,
                'max': hp.max,
                'options': hp.options,
                'description': hp.description,
                'advanced': hp.advanced
            }
            for hp in metadata.hyperparameters
        ],
        'requirements': {
            'gpu': metadata.requires_gpu,
            'multiclass': metadata.supports_multiclass,
            'regression': metadata.supports_regression
        },
        'performance': {
            'complexity': metadata.complexity,
            'training_time': metadata.training_time
        },
        'documentation_url': metadata.documentation_url
    }

@router.get("/api/algorithms/{algorithm_id}/hyperparameters")
async def get_hyperparameters(algorithm_id: str, advanced: bool = False):
    """
    Get hyperparameter schemas for an algorithm
    
    Args:
        advanced: Include advanced hyperparameters
    """
    metadata = algorithm_registry.get(algorithm_id)
    
    if not metadata:
        raise HTTPException(404, f"Algorithm {algorithm_id} not found")
    
    hyperparameters = metadata.hyperparameters
    if not advanced:
        hyperparameters = [hp for hp in hyperparameters if not hp.advanced]
    
    return {
        'hyperparameters': [
            {
                'name': hp.name,
                'type': hp.type,
                'default': hp.default,
                'min': hp.min,
                'max': hp.max,
                'options': hp.options,
                'description': hp.description
            }
            for hp in hyperparameters
        ]
    }

@router.post("/api/algorithms/{algorithm_id}/validate-params")
async def validate_hyperparameters(algorithm_id: str, params: dict):
    """
    Validate hyperparameters without training
    
    Returns validation errors or success
    """
    metadata = algorithm_registry.get(algorithm_id)
    
    if not metadata:
        raise HTTPException(404, f"Algorithm {algorithm_id} not found")
    
    try:
        validated = algorithm_registry._validate_hyperparameters(metadata, params)
        return {
            'valid': True,
            'validated_params': validated
        }
    except Exception as e:
        return {
            'valid': False,
            'errors': [str(e)]
        }
```

---

## Enhanced Frontend Implementation

### Dynamic Sidebar with API Integration

```javascript
// frontend/src/components/AlgorithmSidebar.jsx
import React, { useState, useEffect } from 'react'
import './AlgorithmSidebar.css'

export default function AlgorithmSidebar({ onAlgorithmSelect, selectedId }) {
  const [categories, setCategories] = useState([])
  const [algorithms, setAlgorithms] = useState({})
  const [expandedCategories, setExpandedCategories] = useState({})
  const [loading, setLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')

  useEffect(() => {
    fetchCategories()
    fetchAlgorithms()
  }, [])

  const fetchCategories = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/algorithms/categories')
      const data = await response.json()
      setCategories(data.categories)
      
      // Expand first category by default
      if (data.categories.length > 0) {
        setExpandedCategories({ [data.categories[0].id]: true })
      }
    } catch (error) {
      console.error('Failed to fetch categories:', error)
    }
  }

  const fetchAlgorithms = async () => {
    setLoading(true)
    try {
      const response = await fetch('http://localhost:8000/api/algorithms')
      const data = await response.json()
      
      // Group by category
      const grouped = {}
      data.algorithms.forEach(algo => {
        if (!grouped[algo.category]) {
          grouped[algo.category] = []
        }
        grouped[algo.category].push(algo)
      })
      
      setAlgorithms(grouped)
    } catch (error) {
      console.error('Failed to fetch algorithms:', error)
    } finally {
      setLoading(false)
    }
  }

  const toggleCategory = (categoryId) => {
    setExpandedCategories(prev => ({
      ...prev,
      [categoryId]: !prev[categoryId]
    }))
  }

  const handleAlgorithmClick = async (algorithmId) => {
    // Fetch full algorithm details
    try {
      const response = await fetch(`http://localhost:8000/api/algorithms/${algorithmId}`)
      const details = await response.json()
      onAlgorithmSelect(details)
    } catch (error) {
      console.error('Failed to fetch algorithm details:', error)
    }
  }

  // Filter algorithms by search
  const filteredAlgorithms = Object.entries(algorithms).reduce((acc, [category, algos]) => {
    if (!searchQuery) {
      acc[category] = algos
    } else {
      const filtered = algos.filter(algo => 
        algo.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        algo.description.toLowerCase().includes(searchQuery.toLowerCase())
      )
      if (filtered.length > 0) {
        acc[category] = filtered
      }
    }
    return acc
  }, {})

  return (
    <div className="algorithm-sidebar">
      <div className="sidebar-header">
        <h3>Algorithms</h3>
        <input
          type="text"
          placeholder="Search algorithms..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="algorithm-search"
        />
      </div>

      {loading ? (
        <div className="sidebar-loading">Loading algorithms...</div>
      ) : (
        <div className="category-list">
          {categories.map(category => (
            <div key={category.id} className="category-section">
              <div 
                className={`category-header ${expandedCategories[category.id] ? 'expanded' : ''}`}
                onClick={() => toggleCategory(category.id)}
              >
                <span className="expand-icon">
                  {expandedCategories[category.id] ? '▼' : '▶'}
                </span>
                <span className="category-name">{category.name}</span>
                <span className="category-count">{category.count}</span>
              </div>

              {expandedCategories[category.id] && filteredAlgorithms[category.name] && (
                <div className="algorithm-list">
                  {filteredAlgorithms[category.name].map(algo => (
                    <div
                      key={algo.id}
                      className={`algorithm-item ${selectedId === algo.id ? 'selected' : ''}`}
                      onClick={() => handleAlgorithmClick(algo.id)}
                    >
                      <div className="algorithm-name">{algo.name}</div>
                      <div className="algorithm-meta">
                        <span className={`complexity-badge ${algo.complexity}`}>
                          {algo.complexity}
                        </span>
                        <span className={`speed-badge ${algo.training_time}`}>
                          {algo.training_time}
                        </span>
                        {algo.requires_gpu && (
                          <span className="gpu-badge" title="Requires GPU">
                            🎮 GPU
                          </span>
                        )}
                      </div>
                      <div className="algorithm-description">
                        {algo.description}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
```

---

### Integration with Dashboard

```javascript
// Update Dashboard.jsx to use dynamic sidebar
import AlgorithmSidebar from './components/AlgorithmSidebar'

export default function Dashboard() {
  const [selectedAlgorithm, setSelectedAlgorithm] = useState(null)
  const [hyperparameters, setHyperparameters] = useState({})

  const handleAlgorithmSelect = (algorithmDetails) => {
    setSelectedAlgorithm(algorithmDetails)
    
    // Initialize hyperparameters with defaults
    const defaultParams = {}
    algorithmDetails.hyperparameters.forEach(hp => {
      defaultParams[hp.name] = hp.default
    })
    setHyperparameters(defaultParams)
  }

  return (
    <div className="dashboard-layout">
      <AlgorithmSidebar 
        onAlgorithmSelect={handleAlgorithmSelect}
        selectedId={selectedAlgorithm?.id}
      />
      
      <div className="dashboard-main">
        {selectedAlgorithm && (
          <>
            <h2>{selectedAlgorithm.name}</h2>
            <HyperparameterControls
              hyperparameters={selectedAlgorithm.hyperparameters}
              values={hyperparameters}
              onChange={setHyperparameters}
            />
            {/* ... rest of dashboard */}
          </>
        )}
      </div>
    </div>
  )
}
```

---

## Algorithm Context Preservation

### Session State Management

```python
# backend/app/models/session.py
from sqlalchemy import Column, String, JSON, DateTime
from datetime import datetime

class AlgorithmSession(Base):
    __tablename__ = "algorithm_sessions"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=True)
    algorithm_id = Column(String, nullable=False)
    hyperparameters = Column(JSON, nullable=False)
    dataset_id = Column(String, nullable=True)
    training_status = Column(String, default="not_started")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)

@router.post("/api/sessions/create")
async def create_session(algorithm_id: str, hyperparameters: dict):
    """Create a new algorithm session"""
    session = AlgorithmSession(
        id=generate_unique_id(),
        algorithm_id=algorithm_id,
        hyperparameters=hyperparameters
    )
    db.add(session)
    db.commit()
    
    return {'session_id': session.id}

@router.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Restore a session"""
    session = db.query(AlgorithmSession).get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    
    return {
        'algorithm_id': session.algorithm_id,
        'hyperparameters': session.hyperparameters,
        'dataset_id': session.dataset_id,
        'training_status': session.training_status
    }
```

---

## Styling

```css
/* frontend/src/components/AlgorithmSidebar.css */
.algorithm-sidebar {
  width: 280px;
  height: 100%;
  background: #f8f9fa;
  border-right: 1px solid #e0e0e0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.sidebar-header {
  padding: 16px;
  border-bottom: 1px solid #e0e0e0;
  background: white;
}

.sidebar-header h3 {
  margin: 0 0 12px 0;
  font-size: 16px;
  color: #333;
}

.algorithm-search {
  width: 100%;
  padding: 8px 12px;
  border: 1px solid #ddd;
  border-radius: 6px;
  font-size: 13px;
}

.category-list {
  flex: 1;
  overflow-y: auto;
  padding: 8px;
}

.category-section {
  margin-bottom: 8px;
}

.category-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 12px;
  background: white;
  border-radius: 6px;
  cursor: pointer;
  font-weight: 600;
  font-size: 14px;
  transition: all 0.2s;
}

.category-header:hover {
  background: #e9ecef;
}

.category-header.expanded {
  background: #4682B4;
  color: white;
}

.expand-icon {
  font-size: 10px;
  opacity: 0.7;
}

.category-name {
  flex: 1;
}

.category-count {
  font-size: 12px;
  opacity: 0.7;
  background: rgba(0,0,0,0.1);
  padding: 2px 8px;
  border-radius: 10px;
}

.algorithm-list {
  margin-top: 4px;
  margin-left: 12px;
}

.algorithm-item {
  padding: 10px 12px;
  margin: 4px 0;
  background: white;
  border: 1px solid #e0e0e0;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s;
}

.algorithm-item:hover {
  border-color: #4682B4;
  box-shadow: 0 2px 8px rgba(70, 130, 180, 0.15);
}

.algorithm-item.selected {
  background: #e3f2fd;
  border-color: #4682B4;
  box-shadow: 0 2px 8px rgba(70, 130, 180, 0.25);
}

.algorithm-name {
  font-weight: 600;
  font-size: 13px;
  color: #333;
  margin-bottom: 6px;
}

.algorithm-meta {
  display: flex;
  gap: 6px;
  margin-bottom: 6px;
  flex-wrap: wrap;
}

.complexity-badge,
.speed-badge,
.gpu-badge {
  font-size: 10px;
  padding: 2px 6px;
  border-radius: 3px;
  font-weight: 500;
}

.complexity-badge.low { background: #d4edda; color: #155724; }
.complexity-badge.medium { background: #fff3cd; color: #856404; }
.complexity-badge.high { background: #f8d7da; color: #721c24; }

.speed-badge.fast { background: #d1ecf1; color: #0c5460; }
.speed-badge.medium { background: #d6d8db; color: #383d41; }
.speed-badge.slow { background: #f5c6cb; color: #721c24; }

.gpu-badge {
  background: #e7e8ea;
  color: #495057;
}

.algorithm-description {
  font-size: 11px;
  color: #666;
  line-height: 1.4;
}

.sidebar-loading {
  padding: 20px;
  text-align: center;
  color: #999;
}
```

---

## Future Enhancements

1. **Algorithm Recommendations**: Suggest algorithms based on dataset characteristics
2. **Favorites**: Star/bookmark frequently used algorithms
3. **Recent Algorithms**: Quick access to recently used algorithms
4. **Algorithm Comparison**: Side-by-side comparison of multiple algorithms
5. **Custom Algorithms**: Allow users to register custom algorithms
6. **Tags & Filters**: Filter by problem type, speed, complexity
7. **Tutorial Mode**: Guided walkthrough for beginners
8. **Performance Metrics**: Show typical accuracy ranges
9. **AutoML Integration**: Automatically select best algorithm
10. **Algorithm Versioning**: Track different versions of algorithms
## Routing and Layout (Shared Slots)

- Routes:
  - `/neural-networks`
  - `/decision-tree`
  - `/regression`
  - `/data-analysis`
  - `/performance`
- Shared slots in Layout:
  - Navbar (fixed)
  - Sidebar (left; active item reflects route)
  - CodeEditor slot (visible on model pages)
  - Content (route view renders here)
  - Footer (dark, full-width)
- Sidebar state derives from URL, not local-only state. Clicking items pushes route.

### Frontend Techstack

- React Router for route management; a top-level `Layout` wraps all routes.
- Context provides `datasetId` and `roomId` for collaboration features.
