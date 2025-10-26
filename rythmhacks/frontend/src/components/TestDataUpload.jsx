import React, { useState } from 'react'
import './TestDataUpload.css'

export default function TestDataUpload({ modelId, onMetricsCalculated }) {
  const [file, setFile] = useState(null)
  const [modelType, setModelType] = useState('classifier')
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState(null)
  const [success, setSuccess] = useState(false)

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0]
    if (selectedFile) {
      // Validate file type
      const validTypes = ['.csv', '.json']
      const fileExt = selectedFile.name.substring(selectedFile.name.lastIndexOf('.')).toLowerCase()
      
      if (!validTypes.includes(fileExt)) {
        setError(`Invalid file type. Please upload ${validTypes.join(' or ')} file.`)
        return
      }
      
      setFile(selectedFile)
      setError(null)
      setSuccess(false)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    
    if (!file) {
      setError('Please select a file')
      return
    }

    setUploading(true)
    setError(null)

    try {
      // Read file content
      const fileContent = await readFileContent(file)
      
      let testData
      if (file.name.endsWith('.json')) {
        testData = JSON.parse(fileContent)
      } else if (file.name.endsWith('.csv')) {
        testData = parseCSV(fileContent)
      }

      // Validate test data format
      if (!testData.X_test || !testData.y_test) {
        throw new Error('Test data must contain X_test and y_test')
      }

      // Send to backend
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'
      const response = await fetch(`${apiUrl}/api/models/${modelId}/metrics`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          X_test: testData.X_test,
          y_test: testData.y_test,
          feature_names: testData.feature_names || null,
          model_type: modelType
        })
      })

      if (!response.ok) {
        const errorData = await response.json()
        let errorMessage = errorData.detail || 'Failed to calculate metrics'
        
        // Check for custom model error
        if (errorMessage.includes("Can't get attribute") || errorMessage.includes("custom_models")) {
          errorMessage = "⚠️ Custom PyTorch Model Detected\n\n" +
            "Your model uses a custom architecture that needs to be defined in the backend.\n\n" +
            "To fix this:\n" +
            "1. Open backend/app/core/custom_models.py\n" +
            "2. Add your model class definition (e.g., HeartDiseaseMLP)\n" +
            "3. Restart the backend server\n\n" +
            "Original error: " + errorMessage
        }
        
        throw new Error(errorMessage)
      }

      const metrics = await response.json()
      setSuccess(true)
      
      // Notify parent component
      if (onMetricsCalculated) {
        onMetricsCalculated(metrics)
      }
    } catch (err) {
      setError(err.message)
      console.error('Error uploading test data:', err)
    } finally {
      setUploading(false)
    }
  }

  const readFileContent = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = (e) => resolve(e.target.result)
      reader.onerror = (e) => reject(e)
      reader.readAsText(file)
    })
  }

  const parseCSV = (csvText) => {
    // Simple CSV parser - expects format:
    // First row: feature names + 'target'
    // Subsequent rows: feature values + target value
    const lines = csvText.trim().split('\n')
    const headers = lines[0].split(',').map(h => h.trim())
    
    const X_test = []
    const y_test = []
    
    for (let i = 1; i < lines.length; i++) {
      const values = lines[i].split(',').map(v => parseFloat(v.trim()))
      X_test.push(values.slice(0, -1)) // All except last column
      y_test.push(values[values.length - 1]) // Last column is target
    }
    
    return {
      X_test,
      y_test,
      feature_names: headers.slice(0, -1)
    }
  }

  if (success) {
    return (
      <div className="test-data-success">
        <div className="success-icon">✓</div>
        <h3>Metrics Calculated Successfully!</h3>
        <p>Performance metrics are now available below.</p>
      </div>
    )
  }

  return (
    <div className="test-data-upload">
      <h3>📊 Calculate Performance Metrics</h3>
      <p>Upload your test data to calculate model performance metrics.</p>

      <form onSubmit={handleSubmit} className="test-data-form">
        <div className="form-group">
          <label>Test Data File (.json or .csv)</label>
          <input
            type="file"
            accept=".json,.csv"
            onChange={handleFileChange}
            disabled={uploading}
          />
          {file && <p className="file-info">Selected: {file.name}</p>}
        </div>

        <div className="form-group">
          <label>Model Type</label>
          <select
            value={modelType}
            onChange={(e) => setModelType(e.target.value)}
            disabled={uploading}
          >
            <option value="classifier">Classifier</option>
            <option value="regressor">Regressor</option>
          </select>
        </div>

        {error && <div className="error-message">{error}</div>}

        <button type="submit" disabled={!file || uploading} className="submit-button">
          {uploading ? 'Calculating...' : 'Calculate Metrics'}
        </button>
      </form>

      <div className="format-info">
        <h4>Expected File Format:</h4>
        <div className="format-examples">
          <div className="format-example">
            <strong>JSON format:</strong>
            <pre>{`{
  "X_test": [[1.2, 3.4, ...], [5.6, 7.8, ...]],
  "y_test": [0, 1, 0, ...],
  "feature_names": ["feature1", "feature2", ...]
}`}</pre>
          </div>
          <div className="format-example">
            <strong>CSV format:</strong>
            <pre>{`feature1,feature2,feature3,target
1.2,3.4,5.6,0
7.8,9.0,1.2,1
...`}</pre>
          </div>
        </div>
      </div>
    </div>
  )
}
