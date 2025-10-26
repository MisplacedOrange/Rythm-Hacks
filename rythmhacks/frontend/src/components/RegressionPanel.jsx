import React, { useMemo } from 'react'
import ChartWrapper from './ChartWrapper'

function mockRegressionData(n = 80) {
  const x = [], y = []
  const m = 1.2, b = 0.5
  for (let i = 0; i < n; i++) {
    const xv = i / 8 + Math.random()
    const noise = (Math.random() - 0.5) * 2
    x.push(xv)
    y.push(m * xv + b + noise)
  }
  const lineX = [Math.min(...x), Math.max(...x)]
  const lineY = lineX.map(v => m * v + b)
  return { x, y, lineX, lineY, metrics: { r2: 0.89, mae: 0.52, mse: 0.41, rmse: 0.64 } }
}

export default function RegressionPanel() {
  const data = useMemo(() => mockRegressionData(), [])

  return (
    <div className="regression-panel" style={{ display: 'grid', gridTemplateColumns: '1.2fr 0.8fr', gap: '1rem', maxHeight: '400px' }}>
      <div className="reg-chart chart-container" style={{ background:'#fff', border:'1px solid var(--border-subtle)', borderRadius:'8px', padding:'12px' }}>
        <ChartWrapper
            data={[
              { x: data.x, y: data.y, mode:'markers', type:'scatter', name:'Data' },
              { x: data.lineX, y: data.lineY, mode:'lines', type:'scatter', name:'Linear Regression', line:{ color:'var(--fit-red, #E53935)', width:3 } }
            ]}
            layout={{ xaxis:{ title:'x' }, yaxis:{ title:'y' } }}
            style={{ width:'100%', height: '100%' }}
          />
      </div>
      <div className="reg-metrics" style={{ background:'#fff', border:'1px solid var(--border-subtle)', borderRadius:'8px', padding:'12px' }}>
        <h3 style={{ marginBottom:'8px' }}>Metrics</h3>
        <table style={{ width:'100%', borderCollapse:'collapse' }}>
          <tbody>
            {Object.entries(data.metrics).map(([k,v]) => (
              <tr key={k}>
                <td style={{ padding:'8px', borderBottom:'1px solid var(--border-subtle)', fontWeight:600 }}>{k.toUpperCase()}</td>
                <td style={{ padding:'8px', borderBottom:'1px solid var(--border-subtle)', textAlign:'right' }}>{v.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

