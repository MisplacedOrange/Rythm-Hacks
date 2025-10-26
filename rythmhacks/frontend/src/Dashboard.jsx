import React, { useState, useMemo } from 'react'
import ChartWrapper from './components/ChartWrapper'
import './Dashboard.css'
import Chat from './components/Chat'
import Upload from './components/Upload'
import DataTable from './components/DataTable'
import RegressionPanel from './components/RegressionPanel'
import DecisionTreePanel from './components/DecisionTreePanel'
import ModelUpload from './components/ModelUpload'
import PerformanceCharts from './components/PerformanceCharts'

export default function Dashboard() {
  const [activeCategory, setActiveCategory] = useState('Neural Networks')
  const [activeTab, setActiveTab] = useState('dashboard')
  const [layers, setLayers] = useState([7, 7, 7, 7, 7, 7])
  const [uploadedModelId, setUploadedModelId] = useState(null)
  const [uploadedModelInfo, setUploadedModelInfo] = useState(null)

  const categories = [
    'Neural Networks',
    'Decision Tree',
    'Regression',
    'Data Analysis'
  ]

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

  const updateLayer = (index, delta) => {
    setLayers(prev => {
      const newLayers = [...prev]
      newLayers[index] = Math.max(1, Math.min(15, newLayers[index] + delta))
      return newLayers
    })
  }

  const handleModelUploadSuccess = (modelInfo) => {
    setUploadedModelId(modelInfo.model_id)
    setUploadedModelInfo(modelInfo)
  }

  return (
    <div className="dashboard-page">
      <div className="dashboard-layout">
        {/* Sidebar */}
        {activeTab === 'dashboard' && (
          <aside className="dashboard-sidebar">
            <h2 className="sidebar-title">Algorithms</h2>
            <nav className="sidebar-nav">
              {categories.map(cat => (
                <button
                  key={cat}
                  className={`sidebar-item ${activeCategory === cat ? 'active' : ''}`}
                  onClick={() => setActiveCategory(cat)}
                >
                  {cat}
                </button>
              ))}
            </nav>

            {/* Current Model Info in Sidebar */}
            {uploadedModelInfo && (
              <div className="sidebar-section">
                <h3 className="sidebar-subtitle">Current Model</h3>
                <div className="uploaded-model-info">
                  <p className="model-info-detail">{uploadedModelInfo.filename}</p>
                  <p className="model-info-meta">
                    {uploadedModelInfo.framework} • {uploadedModelInfo.model_type}
                  </p>
                  <p className="model-info-meta">
                    {uploadedModelInfo.file_size_mb}MB
                  </p>
                </div>
              </div>
            )}
          </aside>
        )}

        {/* Main Content */}
        <main className="dashboard-main">
          <div className="main-tab-navigation">
            <button
              className={`main-tab-button ${activeTab === 'dashboard' ? 'active' : ''}`}
              onClick={() => setActiveTab('dashboard')}
            >
              Dashboard
            </button>
            <button
              className={`main-tab-button ${activeTab === 'chat' ? 'active' : ''}`}
              onClick={() => setActiveTab('chat')}
            >
              Chat
            </button>
          </div>

          {/* Hyperparameter controls removed as obsolete */}

          {activeTab === 'dashboard' && (
            <>
              {/* Model Upload Section - Top of Dashboard */}
              <div className="upload-section">
                <h3 className="section-title">Upload Trained Model</h3>
                <ModelUpload onUploadSuccess={handleModelUploadSuccess} />
              </div>

              {activeCategory === 'Data Analysis' && (
                <div style={{ display: 'grid', gap: '1rem', marginBottom: '1rem' }}>
                  <Upload />
                  <DataTable height={260} />
                </div>
              )}

          {/* Neural Network and Output */}
          {activeCategory === 'Neural Networks' && (
            <div className="visualization-grid">
              <div className="viz-section">
                <h3 className="viz-title">Convolutional Layers</h3>
                <div className="neural-network">
                  {/* Input layer */}
                  <div className="network-column">
                    <div className="network-node input-node">X₁</div>
                    <div className="network-node input-node">X₂</div>
                    <div className="network-node input-node">X₃</div>
                    <div className="network-node input-node">X₄</div>
                    <div className="network-node input-node">X₅</div>
                    <div className="network-label">sin(X₁)</div>
                    <div className="network-label">sin(X₂)</div>
                  </div>

                  {/* Hidden layers */}
                  {layers.map((neurons, layerIdx) => (
                    <div key={layerIdx} className="network-column">
                      {Array.from({ length: neurons }).map((_, nodeIdx) => (
                        <div key={nodeIdx} className="network-node hidden-node"></div>
                      ))}
                      <div className="layer-controls">
                        <button onClick={() => updateLayer(layerIdx, 1)}>+</button>
                        <button onClick={() => updateLayer(layerIdx, -1)}>−</button>
                      </div>
                      <div className="layer-label">{neurons} neurons</div>
                    </div>
                  ))}

                  {/* Output layer */}
                  <div className="network-column">
                    {Array.from({ length: 7 }).map((_, i) => (
                      <div key={i} className="network-node output-node"></div>
                    ))}
                    <div className="layer-label">7 neurons</div>
                  </div>
                </div>
                <div className="layers-label">Layers</div>
              </div>

              <div className="viz-section">
                <h3 className="viz-title">Output</h3>
                <div className="output-plot">
                  <ChartWrapper
                    data={[{
                      x: spiralData.x,
                      y: spiralData.y,
                      mode: 'markers',
                      type: 'scatter',
                      marker: { size: 8, color: spiralData.colors, line: { width: 1, color: '#fff' } },
                      showlegend: false
                    }]}
                    layout={{ xaxis: { range: [-6, 6] }, yaxis: { range: [-6, 6] } }}
                    style={{ width: '100%', height: '100%' }}
                  />
                </div>
                <button className="generate-btn">Generate</button>
              </div>
            </div>
          )}

          {/* Category-specific panels */}
          {activeCategory === 'Regression' && (
            <div className="viz-section" style={{ marginBottom: '2rem' }}>
              <RegressionPanel />
            </div>
          )}

          {activeCategory === 'Decision Tree' && (
            <div className="viz-section" style={{ marginBottom: '2rem' }}>
              <DecisionTreePanel />
            </div>
          )}

          {/* Model Performance Section */}
          <div className="bottom-grid">
            <div className="performance-section">
              <h2 className="performance-title">&gt; Model Performance</h2>
              <div className="performance-content">
                {uploadedModelId ? (
                  <PerformanceCharts 
                    modelId={uploadedModelId} 
                    modelType={uploadedModelInfo?.model_type || 'classifier'}
                    activeCategory={activeCategory}
                  />
                ) : (
                  <div className="performance-placeholder">
                    <p>Upload a trained model to view performance metrics</p>
                    <p className="placeholder-hint">
                      Supported formats: .pkl, .joblib (scikit-learn), .h5 (Keras), .pt (PyTorch)
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
            </>
          )}

          {/* Chat Tab */}
          {activeTab === 'chat' && (
            <div className="chat-view">
              <Chat />
            </div>
          )}
        </main>
      </div>
    </div>
  )
}
