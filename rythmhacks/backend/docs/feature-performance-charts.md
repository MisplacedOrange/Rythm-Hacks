# Feature Documentation: Model Performance Charts

## Overview
Comprehensive 4-grid visualization dashboard displaying training metrics, feature importance, ROC curves, and confusion matrices. Real-time updates during training with interactive Plotly charts.

---

## Current Implementation Status

The Dashboard.jsx currently has a placeholder "Model Performance" section in a 2x2 grid layout. This document specifies the complete implementation with all 4 chart types.

---

## Frontend Implementation

### Performance Dashboard Component

```javascript
// frontend/src/components/PerformanceCharts.jsx
import React, { useState, useEffect, useMemo } from 'react'
import Plot from 'react-plotly.js'
import './PerformanceCharts.css'

export default function PerformanceCharts({ sessionId, modelType }) {
  const [metrics, setMetrics] = useState(null)
  const [loading, setLoading] = useState(true)
  const [ws, setWs] = useState(null)

  useEffect(() => {
    // Fetch initial metrics
    fetchMetrics()

    // Setup WebSocket for real-time updates
    const websocket = new WebSocket(
      `ws://localhost:8000/ws/training/${sessionId}`
    )

    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data)
      if (data.type === 'metrics_update') {
        setMetrics(data.metrics)
      }
    }

    setWs(websocket)

    return () => {
      if (websocket) websocket.close()
    }
  }, [sessionId])

  const fetchMetrics = async () => {
    setLoading(true)
    try {
      const response = await fetch(
        `http://localhost:8000/api/training/${sessionId}/metrics`
      )
      const data = await response.json()
      setMetrics(data)
    } catch (error) {
      console.error('Failed to fetch metrics:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading || !metrics) {
    return <div className="performance-loading">Loading metrics...</div>
  }

  return (
    <div className="performance-charts-grid">
      <TrainingCurves data={metrics.training_history} />
      <FeatureImportance data={metrics.feature_importance} />
      <ROCCurve data={metrics.roc_data} modelType={modelType} />
      <ConfusionMatrix data={metrics.confusion_matrix} labels={metrics.class_labels} />
    </div>
  )
}

---

## MVP Checklist (Missing)

- Metrics API contract for summary and trends (mock acceptable initially).
- Components to implement or stub:
  - MetricsTable (R², MAE, MSE, RMSE)
  - MetricsTrendChart (multi-series line over steps)
  - RocCurveChart (FPR/TPR)
  - ConfusionMatrixChart (normalized)
  - ShapBeeswarmChart (static/mock)
- Consistent loading/empty/error states across all tiles.
- Centralized chart wrapper with theme defaults (margins, fonts, colors).

## Techstack Interactions

- Frontend: `react-plotly.js` with shared `ChartWrapper` that reads CSS variables for palette and typography.
- Backend: `/metrics/summary` and `/metrics/trends` endpoints (mock-first) returning JSON shapes referenced here.
- Mock data: seed files under `models/mock/` for offline demos; controlled by `REACT_APP_USE_MOCKS`.

// 1. Training Curves (Loss & Accuracy)
function TrainingCurves({ data }) {
  if (!data) return <div className="chart-placeholder">No training data</div>

  const trainingTrace = {
    x: data.epochs,
    y: data.train_loss,
    mode: 'lines+markers',
    name: 'Training Loss',
    line: { color: '#F4A460', width: 2 },
    marker: { size: 6 }
  }

  const validationTrace = {
    x: data.epochs,
    y: data.val_loss,
    mode: 'lines+markers',
    name: 'Validation Loss',
    line: { color: '#4682B4', width: 2 },
    marker: { size: 6 }
  }

  const accuracyTrace = {
    x: data.epochs,
    y: data.train_accuracy,
    mode: 'lines+markers',
    name: 'Training Accuracy',
    line: { color: '#28a745', width: 2 },
    marker: { size: 6 },
    yaxis: 'y2'
  }

  const valAccuracyTrace = {
    x: data.epochs,
    y: data.val_accuracy,
    mode: 'lines+markers',
    name: 'Validation Accuracy',
    line: { color: '#17a2b8', width: 2 },
    marker: { size: 6 },
    yaxis: 'y2'
  }

  const layout = {
    title: 'Training Progress',
    xaxis: { title: 'Epoch' },
    yaxis: { 
      title: 'Loss',
      side: 'left'
    },
    yaxis2: {
      title: 'Accuracy',
      side: 'right',
      overlaying: 'y',
      range: [0, 1]
    },
    legend: { 
      x: 0.5, 
      y: -0.2, 
      orientation: 'h',
      xanchor: 'center'
    },
    margin: { l: 60, r: 60, t: 40, b: 60 },
    hovermode: 'x unified'
  }

  const config = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false
  }

  return (
    <div className="chart-container">
      <Plot
        data={[trainingTrace, validationTrace, accuracyTrace, valAccuracyTrace]}
        layout={layout}
        config={config}
        style={{ width: '100%', height: '100%' }}
        useResizeHandler={true}
      />
    </div>
  )
}

// 2. Feature Importance
function FeatureImportance({ data }) {
  if (!data || data.features.length === 0) {
    return <div className="chart-placeholder">Feature importance not available</div>
  }

  // Sort by importance
  const sortedData = data.features
    .map((feature, idx) => ({ feature, importance: data.importances[idx] }))
    .sort((a, b) => b.importance - a.importance)
    .slice(0, 20) // Top 20 features

  const trace = {
    x: sortedData.map(d => d.importance),
    y: sortedData.map(d => d.feature),
    type: 'bar',
    orientation: 'h',
    marker: {
      color: sortedData.map(d => d.importance),
      colorscale: [
        [0, '#f0f0f0'],
        [0.5, '#4682B4'],
        [1, '#F4A460']
      ],
      showscale: true,
      colorbar: { title: 'Importance' }
    }
  }

  const layout = {
    title: 'Feature Importance (Top 20)',
    xaxis: { title: 'Importance Score' },
    yaxis: { title: '' },
    margin: { l: 120, r: 40, t: 40, b: 60 },
    hovermode: 'closest'
  }

  const config = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false
  }

  return (
    <div className="chart-container">
      <Plot
        data={[trace]}
        layout={layout}
        config={config}
        style={{ width: '100%', height: '100%' }}
        useResizeHandler={true}
      />
    </div>
  )
}

// 3. ROC Curve
function ROCCurve({ data, modelType }) {
  if (!data || modelType === 'regression') {
    return <div className="chart-placeholder">ROC curve only for classification</div>
  }

  // Multi-class: multiple curves
  const traces = data.classes.map((className, idx) => ({
    x: data.fpr[idx],
    y: data.tpr[idx],
    mode: 'lines',
    name: `${className} (AUC = ${data.auc[idx].toFixed(3)})`,
    line: { width: 2 }
  }))

  // Diagonal reference line
  traces.push({
    x: [0, 1],
    y: [0, 1],
    mode: 'lines',
    name: 'Random (AUC = 0.5)',
    line: { dash: 'dash', color: '#999', width: 1 }
  })

  const layout = {
    title: 'ROC Curve',
    xaxis: { 
      title: 'False Positive Rate',
      range: [0, 1]
    },
    yaxis: { 
      title: 'True Positive Rate',
      range: [0, 1]
    },
    legend: { 
      x: 0.6, 
      y: 0.2
    },
    margin: { l: 60, r: 40, t: 40, b: 60 },
    hovermode: 'closest'
  }

  const config = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false
  }

  return (
    <div className="chart-container">
      <Plot
        data={traces}
        layout={layout}
        config={config}
        style={{ width: '100%', height: '100%' }}
        useResizeHandler={true}
      />
    </div>
  )
}

// 4. Confusion Matrix
function ConfusionMatrix({ data, labels }) {
  if (!data) {
    return <div className="chart-placeholder">No confusion matrix data</div>
  }

  const trace = {
    z: data,
    x: labels,
    y: labels,
    type: 'heatmap',
    colorscale: [
      [0, '#ffffff'],
      [0.5, '#4682B4'],
      [1, '#F4A460']
    ],
    showscale: true,
    colorbar: { title: 'Count' },
    hoverongaps: false,
    hovertemplate: 'Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>'
  }

  // Add text annotations
  const annotations = []
  for (let i = 0; i < data.length; i++) {
    for (let j = 0; j < data[i].length; j++) {
      annotations.push({
        x: labels[j],
        y: labels[i],
        text: String(data[i][j]),
        showarrow: false,
        font: {
          color: data[i][j] > (Math.max(...data.flat()) / 2) ? 'white' : 'black',
          size: 12
        }
      })
    }
  }

  const layout = {
    title: 'Confusion Matrix',
    xaxis: { 
      title: 'Predicted Label',
      side: 'bottom'
    },
    yaxis: { 
      title: 'True Label',
      autorange: 'reversed'
    },
    annotations: annotations,
    margin: { l: 80, r: 40, t: 40, b: 80 }
  }

  const config = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false
  }

  return (
    <div className="chart-container">
      <Plot
        data={[trace]}
        layout={layout}
        config={config}
        style={{ width: '100%', height: '100%' }}
        useResizeHandler={true}
      />
    </div>
  )
}
```

---

## Backend Integration

### Metrics Calculation Engine

```python
# backend/app/core/metrics.py
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    precision_recall_fscore_support,
    mean_squared_error,
    r2_score
)
from typing import Dict, List, Any, Optional
import torch

class MetricsCalculator:
    """Calculate and format metrics for visualization"""
    
    @staticmethod
    def calculate_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        class_labels: List[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive classification metrics
        
        Returns:
            - confusion_matrix
            - roc_data (fpr, tpr, auc for each class)
            - accuracy, precision, recall, f1
        """
        n_classes = len(np.unique(y_true))
        
        if class_labels is None:
            class_labels = [f"Class {i}" for i in range(n_classes)]
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Overall metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        # ROC curve data
        roc_data = None
        if y_pred_proba is not None:
            roc_data = MetricsCalculator._calculate_roc_multiclass(
                y_true, y_pred_proba, n_classes, class_labels
            )
        
        return {
            'confusion_matrix': cm.tolist(),
            'class_labels': class_labels,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_data': roc_data
        }
    
    @staticmethod
    def _calculate_roc_multiclass(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        n_classes: int,
        class_labels: List[str]
    ) -> Dict[str, Any]:
        """Calculate ROC curve for multi-class classification"""
        from sklearn.preprocessing import label_binarize
        
        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        fpr = []
        tpr = []
        roc_auc = []
        
        for i in range(n_classes):
            fpr_i, tpr_i, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            fpr.append(fpr_i.tolist())
            tpr.append(tpr_i.tolist())
            roc_auc.append(float(auc(fpr_i, tpr_i)))
        
        return {
            'classes': class_labels,
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc
        }
    
    @staticmethod
    def calculate_regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate regression metrics"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mae = np.mean(np.abs(y_true - y_pred))
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2)
        }
    
    @staticmethod
    def extract_feature_importance(
        model: Any,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        Extract feature importance from various model types
        
        Supports:
            - Tree-based models (sklearn)
            - Neural networks (via gradient-based importance)
            - Linear models (coefficient magnitude)
        """
        importances = None
        
        # Tree-based models
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        
        # Linear models
        elif hasattr(model, 'coef_'):
            coef = model.coef_
            if len(coef.shape) > 1:
                # Multi-class: average across classes
                importances = np.mean(np.abs(coef), axis=0)
            else:
                importances = np.abs(coef)
        
        # PyTorch models - use gradient-based importance
        elif isinstance(model, torch.nn.Module):
            importances = MetricsCalculator._neural_network_importance(model)
        
        if importances is None:
            return None
        
        # Normalize to 0-1
        importances = importances / importances.sum()
        
        return {
            'features': feature_names,
            'importances': importances.tolist()
        }
    
    @staticmethod
    def _neural_network_importance(model: torch.nn.Module) -> np.ndarray:
        """
        Calculate feature importance for neural networks
        Using first layer weights magnitude
        """
        first_layer = None
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                first_layer = module
                break
        
        if first_layer is None:
            return None
        
        # Use magnitude of weights from first layer
        weights = first_layer.weight.data.cpu().numpy()
        importances = np.mean(np.abs(weights), axis=0)
        
        return importances
```

---

### REST API Endpoints

```python
# backend/app/api/routes/metrics.py
from fastapi import APIRouter, HTTPException
from backend.app.core.metrics import MetricsCalculator
from backend.app.models.training_session import TrainingSession

router = APIRouter()

@router.get("/api/training/{session_id}/metrics")
async def get_training_metrics(session_id: str):
    """
    Get comprehensive metrics for a training session
    
    Returns all 4 chart datasets:
        - training_history (loss/accuracy curves)
        - feature_importance
        - roc_data
        - confusion_matrix
    """
    session = db.query(TrainingSession).get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    
    # Load model and data
    model = load_model(session.model_path)
    X_test, y_test = load_test_data(session.dataset_id)
    
    # Get predictions
    if session.model_type == 'classification':
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        metrics = MetricsCalculator.calculate_classification_metrics(
            y_test, y_pred, y_pred_proba,
            class_labels=session.class_labels
        )
    else:
        y_pred = model.predict(X_test)
        metrics = MetricsCalculator.calculate_regression_metrics(y_test, y_pred)
    
    # Feature importance
    feature_importance = MetricsCalculator.extract_feature_importance(
        model, session.feature_names
    )
    
    # Training history
    training_history = session.training_history  # Stored during training
    
    return {
        'training_history': training_history,
        'feature_importance': feature_importance,
        'roc_data': metrics.get('roc_data'),
        'confusion_matrix': metrics.get('confusion_matrix'),
        'class_labels': metrics.get('class_labels'),
        'overall_metrics': {
            'accuracy': metrics.get('accuracy'),
            'precision': metrics.get('precision'),
            'recall': metrics.get('recall'),
            'f1_score': metrics.get('f1_score'),
            'mse': metrics.get('mse'),
            'r2': metrics.get('r2')
        }
    }

@router.get("/api/training/{session_id}/export-metrics")
async def export_metrics(session_id: str, format: str = 'json'):
    """Export metrics to JSON/CSV"""
    metrics = await get_training_metrics(session_id)
    
    if format == 'csv':
        # Convert to CSV
        import pandas as pd
        df = pd.DataFrame([metrics['overall_metrics']])
        csv = df.to_csv(index=False)
        
        from fastapi.responses import Response
        return Response(
            content=csv,
            media_type='text/csv',
            headers={'Content-Disposition': f'attachment; filename=metrics_{session_id}.csv'}
        )
    
    return metrics
```

---

### Real-time WebSocket Updates

```python
# During training, emit metrics updates
@router.post("/api/training/{session_id}/start")
async def start_training(session_id: str):
    """Start training with real-time metrics streaming"""
    
    async def train_and_stream():
        for epoch in range(num_epochs):
            # Training step
            train_loss, train_acc = train_one_epoch(model, train_loader)
            val_loss, val_acc = validate(model, val_loader)
            
            # Update metrics
            metrics = {
                'type': 'metrics_update',
                'metrics': {
                    'training_history': {
                        'epochs': list(range(epoch + 1)),
                        'train_loss': train_losses,
                        'val_loss': val_losses,
                        'train_accuracy': train_accs,
                        'val_accuracy': val_accs
                    }
                }
            }
            
            # Broadcast via WebSocket
            await training_manager.broadcast(session_id, metrics)
    
    # Run in background
    background_tasks.add_task(train_and_stream)
    
    return {'status': 'training_started'}
```

---

## Styling

```css
/* frontend/src/components/PerformanceCharts.css */
.performance-charts-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  grid-template-rows: 1fr 1fr;
  gap: 16px;
  height: 100%;
  padding: 16px;
  background: white;
  border-radius: 8px;
}

.chart-container {
  background: white;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  padding: 12px;
  overflow: hidden;
}

.chart-placeholder {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #999;
  font-size: 14px;
  background: #f8f9fa;
  border-radius: 8px;
}

.performance-loading {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  font-size: 16px;
  color: #666;
}
```

---

## Integration with Dashboard

```javascript
// Update Dashboard.jsx
import PerformanceCharts from './components/PerformanceCharts'

// In the 2x2 grid section, replace placeholder with:
<div className="performance-section">
  <PerformanceCharts 
    sessionId={currentSessionId}
    modelType="classification"
  />
</div>
```

---

## Future Enhancements

1. **Learning Rate Schedules**: Visualize LR changes
2. **Gradient Statistics**: Track gradient norms
3. **Precision-Recall Curve**: Alternative to ROC
4. **Calibration Plots**: Model calibration assessment
5. **Cross-Validation Results**: K-fold CV visualization
6. **Ensemble Voting**: Show individual model contributions
7. **Hyperparameter Impact**: Correlate params with performance
8. **A/B Testing**: Compare multiple model runs
9. **Statistical Tests**: Significance testing between models
10. **Export to Reports**: Generate PDF/HTML reports
