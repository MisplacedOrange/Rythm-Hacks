import React, { useState, useEffect } from 'react'
import ChartWrapper from './ChartWrapper'
import TestDataUpload from './TestDataUpload'
import './PerformanceCharts.css'

export default function PerformanceCharts({ modelId, modelType = 'classifier', activeCategory }) {
  const [metrics, setMetrics] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!modelId) {
      setLoading(false)
      return
    }

    fetchMetrics()
  }, [modelId])

  const fetchMetrics = async () => {
    setLoading(true)
    setError(null)

    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'
      const response = await fetch(`${apiUrl}/api/models/${modelId}/metrics`)

      if (!response.ok) {
        if (response.status === 400) {
          // No cached metrics - need to provide test data
          setError('NO_METRICS')
        } else {
          throw new Error(`Failed to fetch metrics: ${response.statusText}`)
        }
      } else {
        const data = await response.json()
        setMetrics(data)
      }
    } catch (err) {
      setError(err.message)
      console.error('Error fetching metrics:', err)
    } finally {
      setLoading(false)
    }
  }

  if (!modelId) {
    return (
      <div className="performance-placeholder">
        <p>Upload a model to view performance metrics</p>
      </div>
    )
  }

  if (loading) {
    return (
      <div className="performance-loading">
        <div className="loading-spinner"></div>
        <p>Loading metrics...</p>
      </div>
    )
  }

  if (error) {
    if (error === 'NO_METRICS') {
      return (
        <div className="performance-placeholder">
          <TestDataUpload 
            modelId={modelId} 
            onMetricsCalculated={(metrics) => {
              setMetrics(metrics)
              setError(null)
            }}
          />
        </div>
      )
    }
    
    return (
      <div className="performance-error">
        <p>Error: {error}</p>
        <button onClick={fetchMetrics}>Retry</button>
      </div>
    )
  }

  if (!metrics) {
    return (
      <div className="performance-placeholder">
        <p>No metrics available</p>
      </div>
    )
  }

  const isClassifier = metrics.model_type === 'classifier'

  // Determine which charts to show based on active category
  const showFullDashboard = !activeCategory || activeCategory === 'Neural Networks'
  const showConfusionMatrix = isClassifier && (showFullDashboard || activeCategory === 'Decision Tree')
  const showFeatureImportance = showFullDashboard || activeCategory === 'Decision Tree' || activeCategory === 'Regression'
  const showROC = isClassifier && showFullDashboard

  return (
    <div className="performance-charts-grid">
      {/* Always show metrics table */}
      <MetricsTable
        metrics={metrics.overall_metrics}
        modelType={metrics.model_type}
      />

      {/* Conditional: Confusion Matrix */}
      {showConfusionMatrix && metrics.confusion_matrix ? (
        <ConfusionMatrix
          matrix={metrics.confusion_matrix}
          labels={metrics.class_labels}
        />
      ) : (
        <div className="chart-placeholder">
          {isClassifier 
            ? 'No confusion matrix data' 
            : 'Confusion matrix only for classification'}
        </div>
      )}

      {/* Conditional: Feature Importance */}
      {showFeatureImportance && metrics.feature_importance ? (
        <FeatureImportance data={metrics.feature_importance} />
      ) : (
        <div className="chart-placeholder">
          {showFeatureImportance 
            ? 'No feature importance data' 
            : 'Feature importance not shown for this category'}
        </div>
      )}

      {/* Conditional: ROC Curve */}
      {showROC && metrics.roc_data ? (
        <ROCCurve data={metrics.roc_data} />
      ) : (
        <div className="chart-placeholder">
          {isClassifier && showROC
            ? 'No ROC data' 
            : 'ROC curve only for classification (Neural Networks)'}
        </div>
      )}
    </div>
  )
}

// Sub-component: Metrics Table
function MetricsTable({ metrics, modelType }) {
  if (!metrics) return <div className="chart-placeholder">No metrics</div>

  const isClassifier = modelType === 'classifier'

  return (
    <div className="metrics-table-container">
      <h3>Performance Metrics</h3>
      <table className="metrics-table">
        <thead>
          <tr>
            <th>Metric</th>
            <th>Value</th>
          </tr>
        </thead>
        <tbody>
          {isClassifier ? (
            <>
              <tr>
                <td>Accuracy</td>
                <td>{(metrics.accuracy * 100).toFixed(2)}%</td>
              </tr>
              <tr>
                <td>Precision</td>
                <td>{(metrics.precision * 100).toFixed(2)}%</td>
              </tr>
              <tr>
                <td>Recall</td>
                <td>{(metrics.recall * 100).toFixed(2)}%</td>
              </tr>
              <tr>
                <td>F1 Score</td>
                <td>{(metrics.f1_score * 100).toFixed(2)}%</td>
              </tr>
            </>
          ) : (
            <>
              <tr>
                <td>R² Score</td>
                <td>{metrics.r2?.toFixed(4) || 'N/A'}</td>
              </tr>
              <tr>
                <td>Adjusted R²</td>
                <td>{metrics.adjusted_r2?.toFixed(4) || 'N/A'}</td>
              </tr>
              <tr>
                <td>MAE</td>
                <td>{metrics.mae?.toFixed(4) || 'N/A'}</td>
              </tr>
              <tr>
                <td>MSE</td>
                <td>{metrics.mse?.toFixed(4) || 'N/A'}</td>
              </tr>
              <tr>
                <td>RMSE</td>
                <td>{metrics.rmse?.toFixed(4) || 'N/A'}</td>
              </tr>
            </>
          )}
        </tbody>
      </table>
    </div>
  )
}

// Sub-component: Confusion Matrix
function ConfusionMatrix({ matrix, labels }) {
  if (!matrix || !labels) {
    return <div className="chart-placeholder">No confusion matrix data</div>
  }

  // Create annotations for cell values
  const annotations = []
  for (let i = 0; i < matrix.length; i++) {
    for (let j = 0; j < matrix[i].length; j++) {
      const maxVal = Math.max(...matrix.flat())
      annotations.push({
        x: labels[j],
        y: labels[i],
        text: String(matrix[i][j]),
        showarrow: false,
        font: {
          color: matrix[i][j] > maxVal / 2 ? 'white' : 'black',
          size: 14,
          weight: 'bold'
        }
      })
    }
  }

  const trace = {
    z: matrix,
    x: labels,
    y: labels,
    type: 'heatmap',
    colorscale: [
      [0, '#ffffff'],
      [0.5, '#4682B4'],
      [1, '#F4A460']
    ],
    showscale: true,
    hoverongaps: false,
    hovertemplate: 'Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>'
  }

  const layout = {
    title: {
      text: 'Confusion Matrix',
      font: { size: 14 }
    },
    xaxis: {
      title: 'Predicted Label',
      side: 'bottom'
    },
    yaxis: {
      title: 'True Label',
      autorange: 'reversed'
    },
    annotations: annotations,
    margin: { l: 80, r: 40, t: 50, b: 80 },
    autosize: true
  }

  return (
    <div className="chart-container">
      <ChartWrapper
        data={[trace]}
        layout={layout}
        config={{ displayModeBar: true }}
      />
    </div>
  )
}

// Sub-component: Feature Importance
function FeatureImportance({ data }) {
  if (!data || !data.features || !data.importances) {
    return <div className="chart-placeholder">No feature importance data</div>
  }

  // Sort by importance and take top 20
  const combined = data.features.map((feat, idx) => ({
    feature: feat,
    importance: data.importances[idx]
  }))

  combined.sort((a, b) => b.importance - a.importance)
  const top20 = combined.slice(0, 20)

  const trace = {
    x: top20.map(d => d.importance),
    y: top20.map(d => d.feature),
    type: 'bar',
    orientation: 'h',
    marker: {
      color: top20.map(d => d.importance),
      colorscale: [
        [0, '#f0f0f0'],
        [0.5, '#4682B4'],
        [1, '#F4A460']
      ],
      showscale: false
    }
  }

  const layout = {
    title: {
      text: 'Feature Importance (Top 20)',
      font: { size: 14 }
    },
    xaxis: { 
      title: 'Importance Score',
      tickfont: { size: 11 }
    },
    yaxis: { 
      title: '',
      tickfont: { size: 10 }
    },
    margin: { l: 150, r: 40, t: 50, b: 70 },
    autosize: true,
    showlegend: false
  }

  return (
    <div className="chart-container">
      <ChartWrapper
        data={[trace]}
        layout={layout}
        config={{ displayModeBar: true }}
      />
    </div>
  )
}

// Sub-component: ROC Curve
function ROCCurve({ data }) {
  if (!data || !data.fpr || !data.tpr) {
    return <div className="chart-placeholder">No ROC data</div>
  }

  const traces = []

  // Add curve for each class
  for (let i = 0; i < data.classes.length; i++) {
    traces.push({
      x: data.fpr[i],
      y: data.tpr[i],
      mode: 'lines',
      name: `${data.classes[i]} (AUC = ${data.auc[i].toFixed(3)})`,
      line: { width: 2 }
    })
  }

  // Add diagonal reference line (random classifier)
  traces.push({
    x: [0, 1],
    y: [0, 1],
    mode: 'lines',
    name: 'Random (AUC = 0.5)',
    line: { dash: 'dash', color: '#999', width: 1 }
  })

  const layout = {
    title: {
      text: 'ROC Curve',
      font: { size: 14 }
    },
    xaxis: {
      title: 'False Positive Rate',
      range: [0, 1],
      tickfont: { size: 11 }
    },
    yaxis: {
      title: 'True Positive Rate',
      range: [0, 1],
      tickfont: { size: 11 }
    },
    legend: {
      x: 0.5,
      y: -0.2,
      xanchor: 'center',
      yanchor: 'top',
      orientation: 'h',
      font: { size: 10 }
    },
    margin: { l: 60, r: 40, t: 50, b: 100 },
    autosize: true
  }

  return (
    <div className="chart-container">
      <ChartWrapper
        data={traces}
        layout={layout}
        config={{ displayModeBar: true }}
      />
    </div>
  )
}
