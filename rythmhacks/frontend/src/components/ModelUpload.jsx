import React, { useCallback, useRef, useState } from 'react'
import './ModelUpload.css'

const SUPPORTED_FORMATS = ['.pkl', '.joblib', '.h5', '.pt', '.pth']
const MAX_SIZE = 100 * 1024 * 1024 // 100MB

export default function ModelUpload({ onUploadSuccess }) {
  const inputRef = useRef(null)
  const [error, setError] = useState('')
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploadedModel, setUploadedModel] = useState(null)

  const validateFile = (file) => {
    if (!file) return 'No file selected'
    
    const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase()
    if (!SUPPORTED_FORMATS.includes(ext)) {
      return `Unsupported format. Supported: ${SUPPORTED_FORMATS.join(', ')}`
    }
    
    if (file.size > MAX_SIZE) {
      return `File too large. Maximum size: 100MB`
    }
    
    return null
  }

  const uploadFile = async (file) => {
    setUploading(true)
    setUploadProgress(0)
    setError('')

    const formData = new FormData()
    formData.append('file', file)

    try {
      const xhr = new XMLHttpRequest()

      // Track upload progress
      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) {
          const progress = Math.round((e.loaded / e.total) * 100)
          setUploadProgress(progress)
        }
      })

      // Handle completion
      xhr.addEventListener('load', () => {
        if (xhr.status === 200) {
          const result = JSON.parse(xhr.responseText)
          setUploadedModel(result)
          
          if (onUploadSuccess) {
            onUploadSuccess(result)
          }
        } else {
          const error = JSON.parse(xhr.responseText)
          setError(error.detail || 'Upload failed')
        }
        setUploading(false)
      })

      xhr.addEventListener('error', () => {
        setError('Network error during upload')
        setUploading(false)
      })

      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'
      xhr.open('POST', `${apiUrl}/api/models/upload`)
      xhr.send(formData)
    } catch (err) {
      setError(err.message || 'Upload failed')
      setUploading(false)
    }
  }

  const onFiles = useCallback(async (files) => {
    const file = files && files[0]
    if (!file) return

    const validationError = validateFile(file)
    if (validationError) {
      setError(validationError)
      return
    }

    await uploadFile(file)
  }, [])

  const onInput = (e) => onFiles(e.target.files)
  const onDrop = (e) => {
    e.preventDefault()
    onFiles(e.dataTransfer.files)
  }
  const onDragOver = (e) => {
    e.preventDefault()
  }
  const onDragEnter = (e) => {
    e.preventDefault()
  }

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i]
  }

  return (
    <div className="model-upload">
      <div
        className="model-upload-box"
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragEnter={onDragEnter}
      >
        {uploading ? (
          <div className="upload-progress">
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{ width: `${uploadProgress}%` }}
              />
            </div>
            <p>Uploading... {uploadProgress}%</p>
          </div>
        ) : uploadedModel ? (
          <div className="upload-success">
            <div className="success-icon">✓</div>
            <p><strong>{uploadedModel.filename}</strong></p>
            <p className="model-details">
              {uploadedModel.framework} • {uploadedModel.model_type} • {uploadedModel.file_size_mb}MB
            </p>
            <button
              className="upload-another-btn"
              onClick={() => {
                setUploadedModel(null)
                setUploadProgress(0)
              }}
            >
              Upload Another Model
            </button>
          </div>
        ) : (
          <div className="upload-prompt">
            <div className="upload-icon">
              <svg viewBox="0 0 24 24" width="48" height="48">
                <path
                  fill="currentColor"
                  d="M9 16h6v-6h4l-7-7-7 7h4zm-4 2h14v2H5z"
                />
              </svg>
            </div>
            <p className="upload-title">Upload Trained Model</p>
            <p className="upload-hint">Drag and drop or click to browse</p>
            <button
              className="browse-btn"
              onClick={() => inputRef.current?.click()}
              disabled={uploading}
            >
              Browse Files
            </button>
            <p className="supported-formats">
              Supported: {SUPPORTED_FORMATS.join(', ')}
            </p>
            <p className="max-size">Maximum size: 100MB</p>
          </div>
        )}

        <input
          ref={inputRef}
          type="file"
          accept={SUPPORTED_FORMATS.join(',')}
          style={{ display: 'none' }}
          onChange={onInput}
        />

        {error && <div className="upload-error">{error}</div>}
      </div>
    </div>
  )
}
