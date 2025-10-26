# Graph Visualization Strategy - MediLytica

## Problem Statement

We need to:
1. Train ML models (backend)
2. Extract visualization data from trained models
3. Upload results to collaborative browser
4. Allow users to analyze graphs, chat, and code together in real-time

---

## Recommended Approach: **React Flow**

### Why React Flow?

✅ **Best for MediLytica because:**
- Interactive node-based graphs (perfect for neural network layers, decision trees)
- Built for React (seamless integration)
- Highly customizable node designs
- Support for dynamic graphs (update as training progresses)
- Great documentation and community
- Can handle both directed graphs (neural networks) and tree structures (decision trees)

```bash
npm install reactflow
```

### Alternative Options:
- **D3.js**: More powerful but steeper learning curve, overkill for our needs
- **Cytoscape.js**: Good for complex networks but less React-friendly
- **vis.js**: Older, less maintained

---

## Complete Workflow: Training → Visualization → Collaboration

### **Step 1: Model Training (Backend)**

```python
# backend/app/ml/trainers/neural_network.py
import torch
import torch.nn as nn

class TrainingSession:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.metrics = []
        self.snapshots = []
    
    async def train(self, websocket):
        """Train model and stream updates"""
        for epoch in range(self.config['epochs']):
            # Training loop
            loss, accuracy = self._train_epoch()
            
            # Collect metrics
            metrics = {
                'epoch': epoch,
                'loss': loss,
                'accuracy': accuracy,
                'timestamp': datetime.now().isoformat()
            }
            self.metrics.append(metrics)
            
            # Stream to frontend via WebSocket
            await websocket.send_json(metrics)
            
            # Save snapshot every N epochs
            if epoch % 10 == 0:
                snapshot = self._create_snapshot()
                self.snapshots.append(snapshot)
        
        # After training, generate visualization data
        results = self._generate_results()
        return results
```

---

### **Step 2: Extract Visualization Data (Backend)**

```python
# backend/app/ml/visualizers/graph_generator.py

class NetworkGraphGenerator:
    """Generate graph data for React Flow visualization"""
    
    @staticmethod
    def from_pytorch_model(model):
        """Extract network structure from PyTorch model"""
        nodes = []
        edges = []
        
        # Parse model layers
        for i, (name, layer) in enumerate(model.named_modules()):
            if isinstance(layer, nn.Conv2d):
                nodes.append({
                    'id': f'layer_{i}',
                    'type': 'convolution',
                    'data': {
                        'label': f'Conv2D({layer.out_channels})',
                        'params': {
                            'in_channels': layer.in_channels,
                            'out_channels': layer.out_channels,
                            'kernel_size': layer.kernel_size
                        }
                    },
                    'position': {'x': i * 150, 'y': 100}
                })
            elif isinstance(layer, nn.Linear):
                nodes.append({
                    'id': f'layer_{i}',
                    'type': 'dense',
                    'data': {
                        'label': f'Dense({layer.out_features})',
                        'params': {
                            'in_features': layer.in_features,
                            'out_features': layer.out_features
                        }
                    },
                    'position': {'x': i * 150, 'y': 100}
                })
            
            # Create edges between consecutive layers
            if i > 0:
                edges.append({
                    'id': f'edge_{i-1}_{i}',
                    'source': f'layer_{i-1}',
                    'target': f'layer_{i}',
                    'type': 'smoothstep'
                })
        
        return {'nodes': nodes, 'edges': edges}
    
    @staticmethod
    def from_sklearn_tree(tree_model):
        """Extract decision tree structure"""
        # For decision trees
        tree = tree_model.tree_
        nodes = []
        edges = []
        
        def traverse(node_id, depth=0, x_offset=0):
            if node_id == -1:
                return
            
            feature = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            
            nodes.append({
                'id': f'node_{node_id}',
                'type': 'decision',
                'data': {
                    'label': f'X[{feature}] ≤ {threshold:.2f}' if feature >= 0 else 'Leaf',
                    'samples': tree.n_node_samples[node_id],
                    'value': tree.value[node_id].tolist()
                },
                'position': {'x': x_offset, 'y': depth * 100}
            })
            
            # Recursively add children
            left = tree.children_left[node_id]
            right = tree.children_right[node_id]
            
            if left != -1:
                edges.append({
                    'id': f'edge_{node_id}_{left}',
                    'source': f'node_{node_id}',
                    'target': f'node_{left}',
                    'label': 'True'
                })
                traverse(left, depth + 1, x_offset - 100)
            
            if right != -1:
                edges.append({
                    'id': f'edge_{node_id}_{right}',
                    'source': f'node_{node_id}',
                    'target': f'node_{right}',
                    'label': 'False'
                })
                traverse(right, depth + 1, x_offset + 100)
        
        traverse(0)
        return {'nodes': nodes, 'edges': edges}

class MetricsGenerator:
    """Generate performance metrics data"""
    
    @staticmethod
    def generate_metrics(model, X_test, y_test):
        """Generate all visualization data"""
        from sklearn.metrics import confusion_matrix, roc_curve, auc
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        # ROC Curve
        roc_data = None
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            roc_data = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc': float(roc_auc)
            }
        
        # Feature Importance (if available)
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = {
                'features': [f'Feature {i}' for i in range(len(model.feature_importances_))],
                'scores': model.feature_importances_.tolist()
            }
        
        return {
            'confusion_matrix': cm,
            'roc_curve': roc_data,
            'feature_importance': feature_importance
        }
```

---

### **Step 3: Backend API Endpoint**

```python
# backend/app/api/routes/visualization.py
from fastapi import APIRouter, HTTPException
from app.ml.visualizers.graph_generator import NetworkGraphGenerator, MetricsGenerator

router = APIRouter()

@router.get("/api/viz/graph/{model_id}")
async def get_network_graph(model_id: str):
    """Get network graph visualization data"""
    # Load trained model
    model = load_model(model_id)
    
    # Generate graph data
    if is_pytorch_model(model):
        graph_data = NetworkGraphGenerator.from_pytorch_model(model)
    elif is_sklearn_tree(model):
        graph_data = NetworkGraphGenerator.from_sklearn_tree(model)
    else:
        raise HTTPException(400, "Unsupported model type")
    
    return graph_data

@router.get("/api/viz/metrics/{model_id}")
async def get_performance_metrics(model_id: str):
    """Get all performance metrics"""
    # Load model and test data
    model, X_test, y_test = load_model_and_data(model_id)
    
    # Generate metrics
    metrics = MetricsGenerator.generate_metrics(model, X_test, y_test)
    
    return metrics

@router.post("/api/results/upload")
async def upload_results(results: dict):
    """Upload training results for collaboration"""
    # Save results to database
    result_id = save_results(results)
    
    # Broadcast to all connected users via WebSocket
    await broadcast_to_room(results['room_id'], {
        'type': 'new_results',
        'result_id': result_id,
        'data': results
    })
    
    return {'result_id': result_id}
```

---

### **Step 4: Frontend Visualization (React Flow)**

```javascript
// frontend/src/components/NetworkGraph.jsx
import React, { useCallback } from 'react'
import ReactFlow, {
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
} from 'reactflow'
import 'reactflow/dist/style.css'
import './NetworkGraph.css'

// Custom node types
const ConvolutionNode = ({ data }) => {
  return (
    <div className="conv-node" style={{ background: '#F4A460' }}>
      <div className="node-label">{data.label}</div>
      <div className="node-params">
        {data.params.out_channels} filters
      </div>
    </div>
  )
}

const DenseNode = ({ data }) => {
  return (
    <div className="dense-node" style={{ background: '#4682B4' }}>
      <div className="node-label">{data.label}</div>
      <div className="node-params">
        {data.params.out_features} units
      </div>
    </div>
  )
}

const nodeTypes = {
  convolution: ConvolutionNode,
  dense: DenseNode,
}

export default function NetworkGraph({ modelId }) {
  const [nodes, setNodes, onNodesChange] = useNodesState([])
  const [edges, setEdges, onEdgesChange] = useEdgesState([])

  // Fetch graph data from backend
  React.useEffect(() => {
    fetch(`http://localhost:8000/api/viz/graph/${modelId}`)
      .then(res => res.json())
      .then(data => {
        setNodes(data.nodes)
        setEdges(data.edges)
      })
  }, [modelId])

  return (
    <div style={{ width: '100%', height: '500px' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        fitView
      >
        <Controls />
        <MiniMap />
        <Background variant="dots" gap={12} size={1} />
      </ReactFlow>
    </div>
  )
}
```

---

### **Step 5: Performance Charts (Plotly)**

```javascript
// frontend/src/components/PerformanceCharts.jsx
import React, { useState, useEffect } from 'react'
import Plot from 'react-plotly.js'

export default function PerformanceCharts({ modelId }) {
  const [metrics, setMetrics] = useState(null)

  useEffect(() => {
    fetch(`http://localhost:8000/api/viz/metrics/${modelId}`)
      .then(res => res.json())
      .then(data => setMetrics(data))
  }, [modelId])

  if (!metrics) return <div>Loading...</div>

  return (
    <div className="performance-grid">
      {/* Confusion Matrix */}
      <div className="chart-container">
        <Plot
          data={[{
            z: metrics.confusion_matrix,
            type: 'heatmap',
            colorscale: 'Blues'
          }]}
          layout={{ title: 'Confusion Matrix' }}
        />
      </div>

      {/* ROC Curve */}
      {metrics.roc_curve && (
        <div className="chart-container">
          <Plot
            data={[{
              x: metrics.roc_curve.fpr,
              y: metrics.roc_curve.tpr,
              type: 'scatter',
              mode: 'lines',
              name: `AUC = ${metrics.roc_curve.auc.toFixed(2)}`
            }]}
            layout={{ title: 'ROC Curve' }}
          />
        </div>
      )}

      {/* Feature Importance */}
      {metrics.feature_importance && (
        <div className="chart-container">
          <Plot
            data={[{
              x: metrics.feature_importance.scores,
              y: metrics.feature_importance.features,
              type: 'bar',
              orientation: 'h'
            }]}
            layout={{ title: 'Feature Importance' }}
          />
        </div>
      )}
    </div>
  )
}
```

---

### **Step 6: Real-time Collaboration Integration**

```javascript
// frontend/src/components/CollaborativeDashboard.jsx
import React, { useState, useEffect } from 'react'
import NetworkGraph from './NetworkGraph'
import PerformanceCharts from './PerformanceCharts'
import Chat from './Chat'
import CodeEditor from './CodeEditor'

export default function CollaborativeDashboard({ roomId }) {
  const [currentModel, setCurrentModel] = useState(null)
  const [ws, setWs] = useState(null)

  useEffect(() => {
    // Connect to WebSocket for real-time updates
    const websocket = new WebSocket(`ws://localhost:8000/ws/room/${roomId}`)
    
    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data)
      
      if (data.type === 'new_results') {
        // New model results uploaded, update visualization
        setCurrentModel(data.result_id)
      }
    }
    
    setWs(websocket)
    return () => websocket.close()
  }, [roomId])

  return (
    <div className="collaborative-dashboard">
      <div className="main-content">
        <NetworkGraph modelId={currentModel} />
        <PerformanceCharts modelId={currentModel} />
      </div>
      <div className="sidebar">
        <Chat roomId={roomId} ws={ws} />
        <CodeEditor roomId={roomId} ws={ws} />
      </div>
    </div>
  )
}
```

---

## Complete Flow Summary

```
1. User trains model (backend or uploads pre-trained)
   ↓
2. Backend extracts:
   - Network structure → React Flow graph data
   - Performance metrics → Plotly chart data
   ↓
3. Results saved to database with unique ID
   ↓
4. WebSocket broadcasts to all users in room
   ↓
5. Frontend fetches visualization data via REST API
   ↓
6. React Flow renders network graph
   Plotly renders performance charts
   ↓
7. Users collaborate:
   - Analyze visualizations
   - Discuss in chat
   - Modify/test code
   - All updates sync in real-time
```

---

## Data Storage Structure

```json
{
  "result_id": "model_123",
  "room_id": "team_alpha",
  "model_type": "neural_network",
  "created_at": "2025-10-25T20:30:00Z",
  "graph_data": {
    "nodes": [...],
    "edges": [...]
  },
  "metrics": {
    "confusion_matrix": [[50, 2], [3, 45]],
    "roc_curve": {...},
    "feature_importance": {...}
  },
  "training_history": {
    "epochs": [1, 2, 3, ...],
    "losses": [0.5, 0.4, 0.3, ...]
  }
}
```

---

## Recommendation Summary

✅ **Use React Flow** for network graphs
✅ **Use Plotly** (already have) for performance charts  
✅ **FastAPI** backend with WebSocket for real-time sync
✅ **Store results** in database with REST API access
✅ **Broadcast updates** via WebSocket to all room members

This gives you a complete, scalable, real-time collaborative ML visualization platform! 🚀
