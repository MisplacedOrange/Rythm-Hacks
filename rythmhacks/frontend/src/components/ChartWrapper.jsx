import React from 'react'
import Plot from 'react-plotly.js'

export default function ChartWrapper({ data, layout = {}, config = {}, style }) {
  const baseLayout = {
    paper_bgcolor: getCssVar('--bg-secondary', '#ffffff'),
    plot_bgcolor: '#ffffff',
    font: { family: getCssVar('--font-sans', 'Inter, sans-serif'), color: getCssVar('--text-secondary', '#333') },
    margin: { l: 60, r: 40, t: 50, b: 60 },
    autosize: true,
    colorway: [
      getCssVar('--series-blue', '#2D9CDB'),
      getCssVar('--series-green', '#27AE60'),
      getCssVar('--series-orange', '#F2994A'),
      getCssVar('--series-purple', '#9B51E0'),
      getCssVar('--fit-red', '#E53935')
    ],
  }
  const baseConfig = { displayModeBar: false, responsive: true }
  return (
    <Plot 
      data={data} 
      layout={{ ...baseLayout, ...layout }} 
      config={{ ...baseConfig, ...config }} 
      style={style || { width: '100%', height: '100%' }} 
      useResizeHandler 
    />
  )
}

function getCssVar(name, fallback) {
  if (typeof window === 'undefined') return fallback
  const v = getComputedStyle(document.documentElement).getPropertyValue(name)
  return v?.trim() || fallback
}

