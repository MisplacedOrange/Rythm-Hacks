import React, { useCallback, useRef, useState } from 'react'
import { useApp } from '../context/AppContext'

function simpleCsvParse(text, maxRows = 200) {
  const lines = text.split(/\r?\n/).filter(Boolean)
  if (lines.length === 0) return { schema: [], rows: [] }
  const header = lines[0].split(',')
  const rows = []
  for (let i = 1; i < Math.min(lines.length, maxRows + 1); i++) {
    rows.push(lines[i].split(','))
  }
  const schema = header.map((name) => ({ name, type: 'string' }))
  return { schema, rows }
}

export default function Upload() {
  const { useMocks, setDatasetId, setPreview, setSchema, setRecentDatasets } = useApp()
  const inputRef = useRef(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const onFiles = useCallback(async (files) => {
    const file = files && files[0]
    if (!file) return
    if (!file.name.endsWith('.csv')) {
      setError('Please upload a .csv file')
      return
    }
    if (file.size > 25 * 1024 * 1024) {
      setError('File too large (25MB max)')
      return
    }
    setError('')
    setBusy(true)
    try {
      if (useMocks) {
        const text = await file.text()
        const { schema, rows } = simpleCsvParse(text)
        const id = `ds_${Date.now()}`
        setDatasetId(id)
        setSchema(schema)
        setPreview(rows)
        setRecentDatasets((prev) => [{ id, name: file.name, at: Date.now() }, ...(prev || [])].slice(0, 5))
      } else {
        const url = (import.meta.env.VITE_API_URL || 'http://localhost:8000') + '/datasets/upload'
        const form = new FormData()
        form.append('file', file)
        const res = await fetch(url, { method: 'POST', body: form })
        if (!res.ok) throw new Error('Upload failed')
        const data = await res.json()
        setDatasetId(data.datasetId)
        setSchema(data.schema || [])
        setPreview(data.preview || [])
        setRecentDatasets((prev) => [{ id: data.datasetId, name: file.name, at: Date.now() }, ...(prev || [])].slice(0, 5))
      }
    } catch (e) {
      setError(e.message || 'Upload error')
    } finally {
      setBusy(false)
    }
  }, [useMocks])

  const onInput = (e) => onFiles(e.target.files)
  const onDrop = (e) => { e.preventDefault(); onFiles(e.dataTransfer.files) }
  const onDragOver = (e) => e.preventDefault()

  return (
    <div className="upload" onDrop={onDrop} onDragOver={onDragOver}>
      <div className="upload-box">
        <p><strong>Upload CSV</strong> (drag & drop or choose)</p>
        <button disabled={busy} onClick={() => inputRef.current?.click()}>
          {busy ? 'Uploading…' : 'Choose File'}
        </button>
        <input ref={inputRef} type="file" accept=".csv" style={{ display: 'none' }} onChange={onInput} />
        {error && <div className="upload-error">{error}</div>}
      </div>
      <style>{`
        .upload { background:#fff; border:1px solid var(--border-subtle); border-radius: var(--radius-md); padding: var(--space-xl); }
        .upload-box { text-align:center; color: var(--text-secondary); }
        .upload-box button { margin-top: var(--space-md); padding: 0.5rem 1rem; background:#000; color:#fff; border:none; border-radius:4px; cursor:pointer; }
        .upload-error { color:#b00020; margin-top: var(--space-sm); }
      `}</style>
    </div>
  )
}

