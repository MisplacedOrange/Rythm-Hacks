import React, { useState, useMemo } from 'react'
import Plot from 'react-plotly.js'
import './Dashboard.css'
import Chat from './components/Chat'

export default function Dashboard() {
  const [activeCategory, setActiveCategory] = useState('Neural Networks')
  const [epoch, setEpoch] = useState(333)
  const [learningRate, setLearningRate] = useState(0.03)
  const [activation, setActivation] = useState('ReLU')
  const [regularization, setRegularization] = useState('L2')
  const [regRate, setRegRate] = useState(0.001)
  const [problemType, setProblemType] = useState('Classification')
  const [layers, setLayers] = useState([7, 7, 7, 7, 7, 7])

  const categories = [
    'Neural Networks',
    'Decision Tree',
    'Regression',
    'Data Analysis',
    'Model Performance'
  ]

  // Generate spiral data for visualization
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

  return (
    <div className="dashboard-page">
      <div className="dashboard-layout">
        {/* Sidebar */}
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
        </aside>

        {/* Main Content */}
        <main className="dashboard-main">
          <div className="dashboard-header">
            <h1 className="dashboard-title">Dashboard</h1>
            <p className="dashboard-subtitle">Browse different visualization methods</p>
          </div>

          {/* Hyperparameters */}
          <div className="hyperparameters">
            <div className="param-group">
              <label>Epoch</label>
              <div className="param-value">{epoch.toLocaleString()}</div>
            </div>
            <div className="param-group">
              <label>Learning rate</label>
              <select value={learningRate} onChange={e => setLearningRate(parseFloat(e.target.value))}>
                <option value={0.01}>0.01</option>
                <option value={0.03}>0.03</option>
                <option value={0.1}>0.1</option>
              </select>
            </div>
            <div className="param-group">
              <label>Activation</label>
              <select value={activation} onChange={e => setActivation(e.target.value)}>
                <option value="ReLU">ReLU</option>
                <option value="Sigmoid">Sigmoid</option>
                <option value="Tanh">Tanh</option>
              </select>
            </div>
            <div className="param-group">
              <label>Regularization</label>
              <select value={regularization} onChange={e => setRegularization(e.target.value)}>
                <option value="L1">L1</option>
                <option value="L2">L2</option>
                <option value="None">None</option>
              </select>
            </div>
            <div className="param-group">
              <label>Regularization rate</label>
              <select value={regRate} onChange={e => setRegRate(parseFloat(e.target.value))}>
                <option value={0.0001}>0.0001</option>
                <option value={0.001}>0.001</option>
                <option value={0.01}>0.01</option>
              </select>
            </div>
            <div className="param-group">
              <label>Problem type</label>
              <select value={problemType} onChange={e => setProblemType(e.target.value)}>
                <option value="Classification">Classification</option>
                <option value="Regression">Regression</option>
              </select>
            </div>
          </div>

          {/* Neural Network and Output */}
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
                <Plot
                  data={[
                    {
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
                    }
                  ]}
                  layout={{
                    autosize: true,
                    paper_bgcolor: '#f5f5f5',
                    plot_bgcolor: '#f5f5f5',
                    margin: { l: 40, r: 40, t: 20, b: 40 },
                    xaxis: { 
                      gridcolor: '#ddd', 
                      zerolinecolor: '#999',
                      range: [-6, 6]
                    },
                    yaxis: { 
                      gridcolor: '#ddd', 
                      zerolinecolor: '#999',
                      range: [-6, 6]
                    },
                  }}
                  style={{ width: '100%', height: '100%' }}
                  useResizeHandler
                  config={{ displayModeBar: false }}
                />
              </div>
              <button className="generate-btn">Generate</button>
            </div>
          </div>

          {/* Model Performance and Chat Section */}
          <div className="bottom-grid">
            <div className="performance-section">
              <h2 className="performance-title">&gt; Model Performance</h2>
              <div className="performance-content">
                {/* Add performance metrics here */}
              </div>
            </div>
            
            <div className="chat-section">
              <Chat />
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}
