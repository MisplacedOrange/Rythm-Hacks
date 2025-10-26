import React, { useState } from 'react'
import { useApp } from '../context/AppContext'
import ChartWrapper from './ChartWrapper'

export default function UmapProjection() {
  const { datasetId } = useApp()
  const [umapData, setUmapData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  
  // UMAP parameters
  const [nNeighbors, setNNeighbors] = useState(15)
  const [minDist, setMinDist] = useState(0.1)
  const [metric, setMetric] = useState('euclidean')

  const computeUmap = async () => {
    if (!datasetId) {
      setError('No dataset uploaded')
      return
    }

    setLoading(true)
    setError('')
    
    try {
      const url = `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/datasets/${datasetId}/umap`
      
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          n_neighbors: nNeighbors,
          min_dist: minDist,
          metric: metric
        })
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'UMAP computation failed')
      }

      const data = await response.json()
      setUmapData(data)
    } catch (err) {
      setError(err.message || 'Failed to compute UMAP projection')
      setUmapData(null)
    } finally {
      setLoading(false)
    }
  }

  // Prepare Plotly data
  const plotData = umapData ? [{
    x: umapData.embedding.map(point => point[0]),
    y: umapData.embedding.map(point => point[1]),
    mode: 'markers',
    type: 'scatter',
    marker: {
      size: 6,
      color: umapData.embedding.map((_, idx) => idx), // Color by index
      colorscale: 'Viridis',
      showscale: true,
      line: { width: 0.5, color: 'white' }
    },
    text: umapData.embedding.map((_, idx) => `Point ${idx + 1}`),
    hovertemplate: '<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>'
  }] : []

  const layout = umapData ? {
    title: {
      text: 'UMAP 2D Projection',
      font: { size: 16, weight: 600 }
    },
    xaxis: { 
      title: 'UMAP Dimension 1',
      gridcolor: '#e0e0e0'
    },
    yaxis: { 
      title: 'UMAP Dimension 2',
      gridcolor: '#e0e0e0'
    },
    hovermode: 'closest',
    showlegend: false,
    plot_bgcolor: '#fafafa',
    paper_bgcolor: 'white'
  } : {}

  return (
    <div className="umap-section">
      <h3 className="umap-title">UMAP Dimensionality Reduction</h3>
      
      <div className="umap-controls">
        <div className="param-group">
          <label htmlFor="nNeighbors">
            Neighbors: {nNeighbors}
            <span className="param-hint"> (5-50)</span>
          </label>
          <input
            id="nNeighbors"
            type="range"
            min="5"
            max="50"
            step="5"
            value={nNeighbors}
            onChange={(e) => setNNeighbors(Number(e.target.value))}
            disabled={loading}
          />
        </div>

        <div className="param-group">
          <label htmlFor="minDist">
            Min Distance: {minDist.toFixed(2)}
            <span className="param-hint"> (0.0-1.0)</span>
          </label>
          <input
            id="minDist"
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={minDist}
            onChange={(e) => setMinDist(Number(e.target.value))}
            disabled={loading}
          />
        </div>

        <div className="param-group">
          <label htmlFor="metric">Distance Metric</label>
          <select
            id="metric"
            value={metric}
            onChange={(e) => setMetric(e.target.value)}
            disabled={loading}
          >
            <option value="euclidean">Euclidean</option>
            <option value="manhattan">Manhattan</option>
            <option value="cosine">Cosine</option>
          </select>
        </div>

        <button 
          className="compute-btn"
          onClick={computeUmap}
          disabled={loading || !datasetId}
        >
          {loading ? 'Computing...' : 'Compute UMAP'}
        </button>
      </div>

      {error && (
        <div className="umap-error">
          <strong>Error:</strong> {error}
        </div>
      )}

      {umapData && (
        <>
          <div className="umap-info">
            <p><strong>Features used:</strong> {umapData.feature_names.join(', ')}</p>
            <p><strong>Data points:</strong> {umapData.n_samples} samples, {umapData.n_features} numeric features</p>
          </div>
          
          <div className="umap-plot">
            <ChartWrapper
              data={plotData}
              layout={layout}
              style={{ width: '100%', height: '500px' }}
            />
          </div>
        </>
      )}

      {!umapData && !loading && !error && (
        <div className="umap-placeholder">
          <p>Adjust parameters above and click "Compute UMAP" to visualize your dataset in 2D space</p>
          <p className="umap-hint">
            UMAP will automatically use all numeric columns from your uploaded CSV file
          </p>
        </div>
      )}

      <style>{`
        .umap-section {
          background: white;
          padding: 1.5rem;
          border-radius: 8px;
          margin-top: 2rem;
          margin-bottom: 2rem;
        }

        .umap-title {
          font-size: 1rem;
          font-weight: 600;
          color: #333;
          margin-bottom: 1rem;
        }

        .umap-controls {
          display: flex;
          gap: 1.5rem;
          margin-bottom: 1.5rem;
          padding: 1rem;
          background: #f8f9fa;
          border-radius: 6px;
          flex-wrap: wrap;
          align-items: flex-end;
        }

        .param-group {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
          flex: 1;
          min-width: 150px;
        }

        .param-group label {
          font-size: 0.85rem;
          color: #666;
          font-weight: 500;
        }

        .param-hint {
          font-size: 0.75rem;
          color: #999;
          font-weight: 400;
        }

        .param-group input[type="range"] {
          width: 100%;
          height: 6px;
          border-radius: 3px;
          background: #ddd;
          outline: none;
          cursor: pointer;
        }

        .param-group select {
          padding: 0.5rem;
          border: 1px solid #ddd;
          border-radius: 4px;
          background: white;
          cursor: pointer;
          font-size: 0.9rem;
        }

        .compute-btn {
          padding: 0.6rem 1.5rem;
          background: #000;
          color: white;
          border: none;
          border-radius: 4px;
          font-size: 0.9rem;
          font-weight: 600;
          cursor: pointer;
          transition: background 0.2s;
          white-space: nowrap;
        }

        .compute-btn:hover:not(:disabled) {
          background: #333;
        }

        .compute-btn:disabled {
          background: #999;
          cursor: not-allowed;
        }

        .umap-error {
          padding: 1rem;
          background: #fee;
          border: 1px solid #fcc;
          border-radius: 4px;
          color: #c00;
          margin-bottom: 1rem;
        }

        .umap-info {
          padding: 1rem;
          background: #e8f4f8;
          border-radius: 6px;
          margin-bottom: 1rem;
        }

        .umap-info p {
          margin: 0.25rem 0;
          font-size: 0.9rem;
          color: #333;
        }

        .umap-plot {
          min-height: 500px;
          background: white;
          border-radius: 6px;
          overflow: hidden;
        }

        .umap-placeholder {
          padding: 3rem 2rem;
          text-align: center;
          color: #666;
          background: #f8f9fa;
          border-radius: 6px;
          border: 2px dashed #ddd;
        }

        .umap-placeholder p {
          margin: 0.5rem 0;
          font-size: 1rem;
        }

        .umap-hint {
          font-size: 0.85rem;
          color: #999;
        }
      `}</style>
    </div>
  )
}
