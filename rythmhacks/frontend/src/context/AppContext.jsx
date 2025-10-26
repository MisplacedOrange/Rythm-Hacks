import React, { createContext, useContext, useMemo, useState, useEffect } from 'react'

// Vite uses VITE_ prefix for env vars
const useMocksEnv = typeof import.meta !== 'undefined' ? (import.meta.env.VITE_USE_MOCKS === 'true' || import.meta.env.VITE_USE_MOCKS === true) : true

const AppContext = createContext(null)

export function AppProvider({ children }) {
  const [datasetId, setDatasetId] = useState(null)
  const [recentDatasets, setRecentDatasets] = useState([])
  const [preview, setPreview] = useState([]) // array of arrays
  const [schema, setSchema] = useState([]) // [{name, type}]

  // hydrate from localStorage
  useEffect(() => {
    try {
      const stored = JSON.parse(localStorage.getItem('rh_recentDatasets') || '[]')
      setRecentDatasets(stored)
    } catch {}
  }, [])

  useEffect(() => {
    try {
      localStorage.setItem('rh_recentDatasets', JSON.stringify(recentDatasets || []))
    } catch {}
  }, [recentDatasets])

  const roomId = useMemo(() => {
    const route = typeof window !== 'undefined' ? window.location.pathname.replace(/^\//, '') || 'dashboard' : 'dashboard'
    return datasetId ? `${route}:${datasetId}` : route
  }, [datasetId])

  const value = useMemo(() => ({
    useMocks: useMocksEnv,
    datasetId,
    setDatasetId,
    recentDatasets,
    setRecentDatasets,
    preview,
    setPreview,
    schema,
    setSchema,
    roomId,
  }), [datasetId, recentDatasets, preview, schema])

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>
}

export function useApp() {
  const ctx = useContext(AppContext)
  if (!ctx) throw new Error('useApp must be used within AppProvider')
  return ctx
}

export default AppContext

