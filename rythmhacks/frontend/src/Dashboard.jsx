import React, { useEffect, useMemo, useState } from 'react'
import Plot from 'react-plotly.js'
import './Dashboard.css'

const loadJson = async (path) => {
  const res = await fetch(path)
  if (!res.ok) throw new Error(`Failed to load ${path}`)
  return res.json()
}

export default function Dashboard() {
  const [cycles, setCycles] = useState(null)
  const [triage, setTriage] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false
    Promise.all([
      loadJson('/data/cycles.json').catch(() => null),
      loadJson('/data/triage.json').catch(() => null),
    ])
      .then(([c, t]) => {
        if (cancelled) return
        setCycles(c)
        setTriage(t)
      })
      .catch((e) => !cancelled && setError(e.message))
    return () => {
      cancelled = true
    }
  }, [])

  // Fallback demo data if files missing
  const cyclesData = useMemo(() => {
    if (cycles) return cycles
    const x = Array.from({ length: 30 }, (_, i) => i + 1)
    const scope = x.map((d) => 20 + 0.5 * d + Math.sin(d / 3) * 2)
    const started = x.map((d) => 10 + 0.6 * d + Math.sin(d / 2) * 1.5)
    const completed = x.map((d) => 5 + 0.55 * d + Math.sin(d / 2.5))
    return { x, scope, started, completed, title: 'Cycle 55' }
  }, [cycles])

  const triageData = useMemo(() => {
    if (triage) return triage
    // simple fallback counts
    return {
      categories: ['Bug', 'Feature', 'Incident'],
      counts: [18, 9, 4],
      title: 'Triage'
    }
  }, [triage])

  return (
    <div className="dashboard-page">
      <div className="dashboard-container">
        <div className="dash-grid">
          <section className="dash-card">
            <header className="dash-header">
              <h2 className="dash-title">Build momentum with Cycles</h2>
              <p className="dash-sub">Create healthy routines and focus your team on what work should happen next.</p>
            </header>
            <div className="dash-plot-wrapper">
              <Plot
                data={[
                  {
                    x: cyclesData.x,
                    y: cyclesData.scope,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Scope',
                    line: { color: '#9aa3af', width: 2 },
                  },
                  {
                    x: cyclesData.x,
                    y: cyclesData.started,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Started',
                    line: { color: '#f59e0b', width: 2, dash: 'dot' },
                  },
                  {
                    x: cyclesData.x,
                    y: cyclesData.completed,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Completed',
                    line: { color: '#6366f1', width: 3 },
                  },
                ]}
                layout={{
                  autosize: true,
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  plot_bgcolor: 'rgba(0,0,0,0)',
                  margin: { l: 40, r: 10, t: 10, b: 30 },
                  legend: { orientation: 'h', x: 0, y: 1.2, font: { color: '#a0a0a0' } },
                  xaxis: { gridcolor: '#2a2a2a', color: '#a0a0a0' },
                  yaxis: { gridcolor: '#2a2a2a', color: '#a0a0a0' },
                }}
                style={{ width: '100%', height: '100%' }}
                useResizeHandler
              />
            </div>
          </section>

          <section className="dash-card">
            <header className="dash-header">
              <h2 className="dash-title">Manage incoming work with Triage</h2>
              <p className="dash-sub">Review and assign incoming bug reports, feature requests, and other unplanned work.</p>
            </header>
            <div className="dash-plot-wrapper">
              <Plot
                data={[
                  {
                    x: triageData.categories,
                    y: triageData.counts,
                    type: 'bar',
                    marker: { color: ['#ef4444', '#22c55e', '#f59e0b'] },
                  },
                ]}
                layout={{
                  autosize: true,
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  plot_bgcolor: 'rgba(0,0,0,0)',
                  margin: { l: 40, r: 10, t: 10, b: 40 },
                  xaxis: { gridcolor: '#2a2a2a', color: '#a0a0a0' },
                  yaxis: { gridcolor: '#2a2a2a', color: '#a0a0a0' },
                }}
                style={{ width: '100%', height: '100%' }}
                useResizeHandler
              />
            </div>
          </section>
        </div>

        {error && <div className="dash-error">{error}</div>}
      </div>
    </div>
  )
}
