# Feature Documentation: Spiral Output Visualization

## Overview
Interactive 2D scatter plot showing classification results as a spiral pattern. Uses Plotly.js for interactive visualization with zoom, pan, and hover capabilities.

---

## Current Implementation

### Frontend Component

**Location**: `Dashboard.jsx`

**Data Generation**:
```javascript
const spiralData = useMemo(() => {
  const theta = []
  const r = []
  const x = []
  const y = []
  const colors = []
  
  for (let i = 0; i < 200; i++) {
    const t = (i / 200) * 4 * Math.PI
    const radius = 1 + t / 4
    theta.push(t)
    r.push(radius)
    x.push(radius * Math.cos(t))
    y.push(radius * Math.sin(t))
    colors.push(i < 100 ? '#F4A460' : '#4682B4')
  }
  
  return { x, y, colors }
}, [])
```

**Plotly Configuration**:
```javascript
<Plot
  data={[{
    x: spiralData.x,
    y: spiralData.y,
    mode: 'markers',
    type: 'scatter',
    marker: {
      size: 8,
      color: spiralData.colors,
      line: { width: 1, color: '#fff' }
    },
    showlegend: false
  }]}
  layout={{
    autosize: true,
    paper_bgcolor: '#f5f5f5',
    plot_bgcolor: '#f5f5f5',
    margin: { l: 40, r: 40, t: 20, b: 40 },
    xaxis: { gridcolor: '#ddd', zerolinecolor: '#999', range: [-6, 6] },
    yaxis: { gridcolor: '#ddd', zerolinecolor: '#999', range: [-6, 6] }
  }}
  style={{ width: '100%', height: '100%' }}
  useResizeHandler
  config={{ displayModeBar: false }}
/>
```

---

## Backend Integration

### Dataset Generation API

**POST `/api/datasets/generate/spiral`**

```python
# backend/app/api/routes/datasets.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import numpy as np
from typing import Literal

router = APIRouter()

class SpiralDatasetConfig(BaseModel):
    n_points: int = Field(default=200, ge=10, le=10000)
    n_classes: int = Field(default=2, ge=2, le=10)
    noise: float = Field(default=0.0, ge=0.0, le=1.0)
    turns: float = Field(default=2.0, ge=0.5, le=10.0)

@router.post("/api/datasets/generate/spiral")
async def generate_spiral_dataset(config: SpiralDatasetConfig):
    """
    Generate synthetic spiral dataset for classification
    """
    try:
        data = generate_spiral_data(
            n_samples=config.n_points,
            n_classes=config.n_classes,
            noise=config.noise,
            turns=config.turns
        )
        
        # Store dataset
        dataset_id = save_dataset(data)
        
        return {
            "dataset_id": dataset_id,
            "n_points": len(data['x']),
            "n_classes": config.n_classes,
            "data": {
                "x": data['x'].tolist(),
                "y": data['y'].tolist(),
                "labels": data['labels'].tolist(),
                "colors": data['colors']
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def generate_spiral_data(n_samples=200, n_classes=2, noise=0.0, turns=2.0):
    """
    Generate spiral dataset
    
    Args:
        n_samples: Total number of points
        n_classes: Number of spiral arms
        noise: Amount of Gaussian noise to add
        turns: Number of spiral turns
    
    Returns:
        Dictionary with x, y coordinates, labels, and colors
    """
    points_per_class = n_samples // n_classes
    
    X = np.zeros((n_samples, 2))
    y = np.zeros(n_samples, dtype=int)
    colors = []
    
    # Color palette for different classes
    color_palette = ['#F4A460', '#4682B4', '#90EE90', '#FFB6C1', '#DDA0DD', 
                     '#F0E68C', '#87CEEB', '#FFA07A', '#98FB98', '#DEB887']
    
    for class_idx in range(n_classes):
        # Generate points for this class
        start_idx = class_idx * points_per_class
        end_idx = start_idx + points_per_class
        
        # Theta values for spiral
        theta = np.linspace(
            class_idx * 2 * np.pi / n_classes,
            class_idx * 2 * np.pi / n_classes + turns * 2 * np.pi,
            points_per_class
        )
        
        # Radius grows with angle
        r = np.linspace(0.1, 1, points_per_class)
        
        # Convert to Cartesian coordinates
        X[start_idx:end_idx, 0] = r * np.cos(theta)
        X[start_idx:end_idx, 1] = r * np.sin(theta)
        
        # Add noise
        if noise > 0:
            X[start_idx:end_idx] += np.random.randn(points_per_class, 2) * noise
        
        # Labels
        y[start_idx:end_idx] = class_idx
        
        # Colors
        colors.extend([color_palette[class_idx % len(color_palette)]] * points_per_class)
    
    return {
        'x': X[:, 0],
        'y': X[:, 1],
        'labels': y,
        'colors': colors
    }
```

---

### Model Prediction Visualization

**GET `/api/predictions/visualize/{model_id}`**

```python
@router.get("/api/predictions/visualize/{model_id}")
async def visualize_predictions(
    model_id: str,
    dataset_id: str,
    resolution: int = 100
):
    """
    Generate prediction visualization with decision boundaries
    
    Args:
        model_id: Trained model ID
        dataset_id: Dataset to visualize on
        resolution: Grid resolution for decision boundary
    
    Returns:
        Visualization data including points and decision boundary
    """
    # Load model and dataset
    model = load_model(model_id)
    dataset = load_dataset(dataset_id)
    
    # Get predictions
    predictions = model.predict(dataset['X'])
    
    # Generate decision boundary mesh
    x_min, x_max = dataset['X'][:, 0].min() - 1, dataset['X'][:, 0].max() + 1
    y_min, y_max = dataset['X'][:, 1].min() - 1, dataset['X'][:, 1].max() + 1
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    
    # Predict on mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    return {
        "points": {
            "x": dataset['X'][:, 0].tolist(),
            "y": dataset['X'][:, 1].tolist(),
            "true_labels": dataset['y'].tolist(),
            "predicted_labels": predictions.tolist(),
            "colors": [get_color_for_label(p) for p in predictions]
        },
        "decision_boundary": {
            "x": xx.tolist(),
            "y": yy.tolist(),
            "z": Z.tolist()
        },
        "accuracy": calculate_accuracy(dataset['y'], predictions)
    }
```

---

## Frontend Enhanced Implementation

### Dynamic Output Visualization Component

```javascript
// frontend/src/components/OutputVisualization.jsx
import React, { useState, useEffect } from 'react'
import Plot from 'react-plotly.js'

export default function OutputVisualization({ modelId, datasetId }) {
  const [vizData, setVizData] = useState(null)
  const [loading, setLoading] = useState(false)
  
  const generateVisualization = async () => {
    setLoading(true)
    
    try {
      const response = await fetch(
        `http://localhost:8000/api/predictions/visualize/${modelId}?dataset_id=${datasetId}`
      )
      const data = await response.json()
      setVizData(data)
    } catch (error) {
      console.error('Failed to generate visualization:', error)
    } finally {
      setLoading(false)
    }
  }
  
  if (loading) return <div>Generating visualization...</div>
  if (!vizData) {
    return (
      <button onClick={generateVisualization} className="generate-btn">
        Generate Visualization
      </button>
    )
  }
  
  // Prepare Plotly data
  const plotData = [
    // Decision boundary contour
    {
      type: 'contour',
      x: vizData.decision_boundary.x[0],
      y: vizData.decision_boundary.y.map(row => row[0]),
      z: vizData.decision_boundary.z,
      colorscale: [
        [0, 'rgba(244, 164, 96, 0.3)'],
        [1, 'rgba(70, 130, 180, 0.3)']
      ],
      showscale: false,
      contours: {
        coloring: 'heatmap'
      }
    },
    // Data points
    {
      x: vizData.points.x,
      y: vizData.points.y,
      mode: 'markers',
      type: 'scatter',
      marker: {
        size: 10,
        color: vizData.points.colors,
        line: {
          width: 2,
          color: vizData.points.true_labels.map((label, idx) => 
            label === vizData.points.predicted_labels[idx] ? '#fff' : '#ff0000'
          )
        }
      },
      text: vizData.points.true_labels.map((true_label, idx) => 
        `True: ${true_label}<br>Predicted: ${vizData.points.predicted_labels[idx]}`
      ),
      hoverinfo: 'text'
    }
  ]
  
  return (
    <div className="output-viz">
      <div className="viz-header">
        <h3>Classification Results</h3>
        <span className="accuracy">Accuracy: {(vizData.accuracy * 100).toFixed(2)}%</span>
      </div>
      
      <Plot
        data={plotData}
        layout={{
          autosize: true,
          paper_bgcolor: '#f5f5f5',
          plot_bgcolor: '#f5f5f5',
          margin: { l: 40, r: 40, t: 20, b: 40 },
          xaxis: { 
            gridcolor: '#ddd',
            title: 'Feature 1'
          },
          yaxis: { 
            gridcolor: '#ddd',
            title: 'Feature 2'
          },
          hovermode: 'closest'
        }}
        style={{ width: '100%', height: '100%' }}
        useResizeHandler
        config={{ 
          displayModeBar: true,
          modeBarButtonsToRemove: ['lasso2d', 'select2d']
        }}
      />
      
      <button onClick={generateVisualization} className="regenerate-btn">
        Regenerate
      </button>
    </div>
  )
}
```

---

## Multiple Dataset Types

### Backend Dataset Generator

```python
# backend/app/ml/datasets/generators.py
import numpy as np
from sklearn.datasets import make_moons, make_circles, make_blobs

class DatasetGenerator:
    """Generate various synthetic datasets for classification"""
    
    @staticmethod
    def spiral(n_samples=200, n_classes=2, noise=0.0, turns=2.0):
        """Generate spiral dataset (already implemented above)"""
        pass
    
    @staticmethod
    def moons(n_samples=200, noise=0.1):
        """Generate two interleaving half circles"""
        X, y = make_moons(n_samples=n_samples, noise=noise)
        
        colors = ['#F4A460' if label == 0 else '#4682B4' for label in y]
        
        return {
            'x': X[:, 0],
            'y': X[:, 1],
            'labels': y,
            'colors': colors
        }
    
    @staticmethod
    def circles(n_samples=200, noise=0.05, factor=0.5):
        """Generate concentric circles"""
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor)
        
        colors = ['#F4A460' if label == 0 else '#4682B4' for label in y]
        
        return {
            'x': X[:, 0],
            'y': X[:, 1],
            'labels': y,
            'colors': colors
        }
    
    @staticmethod
    def blobs(n_samples=200, n_centers=3, cluster_std=0.6):
        """Generate Gaussian blobs"""
        X, y = make_blobs(
            n_samples=n_samples,
            centers=n_centers,
            cluster_std=cluster_std
        )
        
        color_palette = ['#F4A460', '#4682B4', '#90EE90', '#FFB6C1', '#DDA0DD']
        colors = [color_palette[label % len(color_palette)] for label in y]
        
        return {
            'x': X[:, 0],
            'y': X[:, 1],
            'labels': y,
            'colors': colors
        }
```

---

## Real-time Training Visualization

### Live Decision Boundary Updates

```python
# backend/app/api/websockets/training_viz.py
from fastapi import WebSocket
import asyncio

async def stream_training_visualization(
    websocket: WebSocket,
    model,
    dataset,
    config
):
    """Stream visualization updates during training"""
    await websocket.accept()
    
    try:
        for epoch in range(config.epochs):
            # Training step
            model, metrics = train_epoch(model, dataset, config)
            
            # Generate visualization every N epochs
            if epoch % 5 == 0:
                viz_data = generate_live_visualization(model, dataset)
                
                await websocket.send_json({
                    'type': 'visualization_update',
                    'epoch': epoch,
                    'data': viz_data,
                    'metrics': metrics
                })
            
            await asyncio.sleep(0.1)  # Prevent overwhelming client
            
    except Exception as e:
        await websocket.send_json({
            'type': 'error',
            'message': str(e)
        })
    finally:
        await websocket.close()
```

---

## Frontend Animation

### Animated Training Progress

```javascript
// Animated visualization during training
const [trainingHistory, setTrainingHistory] = useState([])

useEffect(() => {
  if (!trainingWs) return
  
  trainingWs.onmessage = (event) => {
    const data = JSON.parse(event.data)
    
    if (data.type === 'visualization_update') {
      // Animate transition between old and new visualization
      setTrainingHistory(prev => [...prev, data])
      
      // Update plot with animation
      updatePlotWithAnimation(data.data)
    }
  }
}, [trainingWs])
```

---

## Export & Sharing

### Export Visualization

```python
@router.get("/api/visualizations/export/{viz_id}")
async def export_visualization(
    viz_id: str,
    format: Literal['png', 'svg', 'pdf', 'json'] = 'png'
):
    """Export visualization in various formats"""
    viz_data = load_visualization(viz_id)
    
    if format == 'json':
        return viz_data
    else:
        # Use plotly to generate static image
        fig = create_plotly_figure(viz_data)
        image_bytes = fig.to_image(format=format)
        
        return Response(
            content=image_bytes,
            media_type=f'image/{format}'
        )
```

---

## Performance Optimization

1. **Downsampling**: For large datasets (>10k points), downsample for visualization
2. **WebGL**: Use Plotly's WebGL renderer for >1000 points
3. **Lazy Loading**: Only generate visualization when requested
4. **Caching**: Cache generated visualizations

---

## Future Enhancements

1. **3D Visualizations**: Support for 3-dimensional data
2. **Interactive Editing**: Click points to relabel
3. **Comparison View**: Side-by-side before/after training
4. **Custom Datasets**: Upload CSV and auto-visualize
5. **Animation Controls**: Play/pause training visualization
## MVP Components (Missing)

- Regression visuals: scatter plot + fitted line overlay; metrics table; metric trend line chart.
- Decision tree renderer: SVG-based nodes/edges with pan/zoom and export.
- Performance tiles: ROC, confusion matrix, training curves, SHAP (static/mock acceptable).
- Chart wrapper: centralized Plotly config (theme, margins, fonts) used by all charts.

## Techstack Integration

- Frontend: Plotly via `react-plotly.js` with a `ChartWrapper` that injects theme tokens (colors, fonts from CSS variables) and default margins.
- Backend: endpoints (mockable) to provide data series for regression fit, decision boundary grids, and tree structures.
- Mock toggle: `REACT_APP_USE_MOCKS=true` switches to local JSON generators; otherwise calls backend.
