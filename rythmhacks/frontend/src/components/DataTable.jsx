import React from 'react'
import { useApp } from '../context/AppContext'

export default function DataTable({ rows: rowsProp, schema: schemaProp, height = 300 }) {
  const { preview, schema } = useApp()
  const cols = schemaProp || schema || []
  const rows = rowsProp || preview || []

  if (!cols.length || !rows.length) {
    return <div style={{ padding: '1rem', color: 'var(--text-tertiary)' }}>No preview data</div>
  }

  return (
    <div className="table-wrap" style={{ maxHeight: height, overflow: 'auto', border: '1px solid var(--border-subtle)', borderRadius: '8px', background: '#fff' }}>
      <table className="mini-table">
        <thead>
          <tr>
            {cols.map((c) => (
              <th key={c.name}>{c.name}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              {cols.map((c, j) => (
                <td key={c.name + j}>{Array.isArray(r) ? r[j] : r[c.name]}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <style>{`
        .mini-table { width: 100%; border-collapse: collapse; }
        .mini-table th { position: sticky; top: 0; background: #fafafa; text-align: left; padding: 8px; font-weight: 600; border-bottom: 1px solid var(--border-subtle); }
        .mini-table td { padding: 8px; border-bottom: 1px solid var(--border-subtle); font-size: var(--text-sm); color: var(--text-secondary); }
      `}</style>
    </div>
  )
}

