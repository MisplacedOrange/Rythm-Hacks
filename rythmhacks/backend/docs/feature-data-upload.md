# Feature Documentation: Data Upload & Import

## Overview
Multi-format data upload system supporting CSV, Parquet, Excel, and JSON files. Includes drag-and-drop interface, data validation, preview, automatic type detection, and preprocessing pipeline configuration.

---

## Frontend Implementation

### Upload Component

```javascript
// frontend/src/components/DataUpload.jsx
import React, { useState, useRef } from 'react'
import './DataUpload.css'

export default function DataUpload({ onUploadComplete }) {
  const [dragActive, setDragActive] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploadedFile, setUploadedFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const fileInputRef = useRef(null)

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }

  const handleChange = (e) => {
    e.preventDefault()
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0])
    }
  }

  const handleFile = async (file) => {
    // Validate file
    const validTypes = [
      'text/csv',
      'application/vnd.ms-excel',
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      'application/json',
      'application/octet-stream' // For .parquet
    ]
    
    const validExtensions = ['.csv', '.xlsx', '.xls', '.json', '.parquet']
    const fileExtension = file.name.substring(file.name.lastIndexOf('.')).toLowerCase()
    
    if (!validExtensions.includes(fileExtension)) {
      alert('Invalid file type. Supported: CSV, Excel, JSON, Parquet')
      return
    }

    // Check file size (max 500MB)
    if (file.size > 500 * 1024 * 1024) {
      alert('File too large. Maximum size: 500MB')
      return
    }

    setUploadedFile(file)
    await uploadFile(file)
  }

  const uploadFile = async (file) => {
    setUploading(true)
    setUploadProgress(0)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const xhr = new XMLHttpRequest()

      // Track upload progress
      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) {
          const progress = (e.loaded / e.total) * 100
          setUploadProgress(progress)
        }
      })

      // Handle completion
      xhr.addEventListener('load', async () => {
        if (xhr.status === 200) {
          const result = JSON.parse(xhr.responseText)
          
          // Fetch preview
          await fetchPreview(result.dataset_id)
          
          if (onUploadComplete) {
            onUploadComplete(result)
          }
        } else {
          alert('Upload failed: ' + xhr.statusText)
        }
        setUploading(false)
      })

      xhr.addEventListener('error', () => {
        alert('Upload failed')
        setUploading(false)
      })

      xhr.open('POST', 'http://localhost:8000/api/data/upload')
      xhr.send(formData)
    } catch (error) {
      console.error('Upload error:', error)
      alert('Upload failed: ' + error.message)
      setUploading(false)
    }
  }

  const fetchPreview = async (datasetId) => {
    try {
      const response = await fetch(
        `http://localhost:8000/api/data/preview/${datasetId}?page=0&page_size=10`
      )
      const data = await response.json()
      setPreview(data)
    } catch (error) {
      console.error('Failed to fetch preview:', error)
    }
  }

  const onButtonClick = () => {
    fileInputRef.current.click()
  }

  return (
    <div className="data-upload-container">
      <div className="upload-section">
        <h2>Upload Dataset</h2>
        
        <div
          className={`drop-zone ${dragActive ? 'active' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          {uploading ? (
            <div className="upload-progress">
              <div className="progress-bar">
                <div 
                  className="progress-fill" 
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
              <p>Uploading... {Math.round(uploadProgress)}%</p>
            </div>
          ) : uploadedFile ? (
            <div className="upload-success">
              <span className="checkmark">✓</span>
              <p>Uploaded: {uploadedFile.name}</p>
              <p className="file-size">{formatBytes(uploadedFile.size)}</p>
              <button onClick={() => { setUploadedFile(null); setPreview(null) }}>
                Upload Another
              </button>
            </div>
          ) : (
            <div className="upload-prompt">
              <svg className="upload-icon" viewBox="0 0 24 24" width="48" height="48">
                <path fill="currentColor" d="M9 16h6v-6h4l-7-7-7 7h4zm-4 2h14v2H5z"/>
              </svg>
              <p>Drag and drop your dataset here</p>
              <p className="upload-hint">or</p>
              <button onClick={onButtonClick} className="browse-btn">
                Browse Files
              </button>
              <p className="supported-formats">
                Supported: CSV, Excel (.xlsx, .xls), JSON, Parquet
              </p>
              <p className="max-size">Maximum size: 500MB</p>
            </div>
          )}
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,.xlsx,.xls,.json,.parquet"
          onChange={handleChange}
          style={{ display: 'none' }}
        />
      </div>

      {preview && (
        <div className="preview-section">
          <h3>Dataset Preview</h3>
          <div className="preview-stats">
            <span><strong>Rows:</strong> {preview.total_rows}</span>
            <span><strong>Columns:</strong> {preview.columns.length}</span>
          </div>
          
          <div className="preview-table-wrapper">
            <table className="preview-table">
              <thead>
                <tr>
                  {preview.columns.map(col => (
                    <th key={col.name}>
                      {col.name}
                      <div className="col-type">{col.dtype}</div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {preview.rows.map((row, idx) => (
                  <tr key={idx}>
                    {preview.columns.map(col => (
                      <td key={col.name}>{formatCell(row[col.name])}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="preview-actions">
            <button onClick={() => configurePreprocessing(preview)}>
              Configure Preprocessing
            </button>
            <button onClick={() => navigateToDataTable(preview)}>
              View Full Dataset
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

const formatBytes = (bytes) => {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i]
}

const formatCell = (value) => {
  if (value === null || value === undefined) {
    return <span className="null-value">NULL</span>
  }
  if (typeof value === 'number') {
    return value.toFixed(4)
  }
  return String(value)
}
```

---

## Backend Implementation

### Upload Endpoint with Validation

```python
# backend/app/api/routes/upload.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, Any
import hashlib
from datetime import datetime

router = APIRouter()

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

@router.post("/api/data/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload and validate dataset file
    
    Supports: CSV, Excel, JSON, Parquet
    
    Returns:
        - dataset_id
        - metadata (rows, columns, dtypes)
        - validation results
    """
    try:
        # Validate file size
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(413, "File too large. Maximum 500MB")
        
        # Determine file type
        file_ext = Path(file.filename).suffix.lower()
        
        # Read into DataFrame
        if file_ext == '.csv':
            df = pd.read_csv(io.BytesIO(contents))
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(io.BytesIO(contents))
        elif file_ext == '.json':
            df = pd.read_json(io.BytesIO(contents))
        elif file_ext == '.parquet':
            df = pd.read_parquet(io.BytesIO(contents))
        else:
            raise HTTPException(400, f"Unsupported file type: {file_ext}")
        
        # Validate DataFrame
        validation_results = validate_dataset(df)
        
        if not validation_results['valid']:
            return JSONResponse(
                status_code=400,
                content={
                    'error': 'Dataset validation failed',
                    'issues': validation_results['issues']
                }
            )
        
        # Generate dataset ID
        dataset_id = generate_dataset_id(file.filename)
        
        # Save as Parquet for efficient storage
        parquet_path = UPLOAD_DIR / f"{dataset_id}.parquet"
        df.to_parquet(parquet_path, index=False)
        
        # Extract metadata
        metadata = extract_metadata(df, file.filename, dataset_id)
        
        # Save metadata
        metadata_path = UPLOAD_DIR / f"{dataset_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Perform automatic type detection and suggestions
        type_suggestions = suggest_column_types(df)
        
        return {
            'dataset_id': dataset_id,
            'metadata': metadata,
            'validation': validation_results,
            'type_suggestions': type_suggestions,
            'message': 'Upload successful'
        }
    
    except pd.errors.EmptyDataError:
        raise HTTPException(400, "Empty dataset")
    except pd.errors.ParserError as e:
        raise HTTPException(400, f"Failed to parse file: {str(e)}")
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {str(e)}")

def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate uploaded dataset
    
    Checks:
        - Not empty
        - Reasonable size
        - Column names valid
        - No completely null columns
        - Encoding issues
    """
    issues = []
    
    # Check not empty
    if len(df) == 0:
        issues.append("Dataset is empty (0 rows)")
    
    if len(df.columns) == 0:
        issues.append("Dataset has no columns")
    
    # Check size limits
    if len(df) > 10_000_000:
        issues.append(f"Dataset very large ({len(df)} rows). May be slow to process")
    
    # Check column names
    for col in df.columns:
        if not isinstance(col, str) or col.strip() == '':
            issues.append(f"Invalid column name: '{col}'")
        if col.startswith('Unnamed'):
            issues.append(f"Unnamed column detected: '{col}'")
    
    # Check for completely null columns
    null_cols = df.columns[df.isnull().all()].tolist()
    if null_cols:
        issues.append(f"Completely null columns: {null_cols}")
    
    # Check for duplicate column names
    dup_cols = df.columns[df.columns.duplicated()].tolist()
    if dup_cols:
        issues.append(f"Duplicate column names: {dup_cols}")
    
    # Check memory usage
    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    if memory_mb > 1000:
        issues.append(f"High memory usage: {memory_mb:.1f} MB")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': []
    }

def extract_metadata(df: pd.DataFrame, filename: str, dataset_id: str) -> Dict[str, Any]:
    """Extract comprehensive dataset metadata"""
    return {
        'id': dataset_id,
        'filename': filename,
        'upload_date': datetime.utcnow().isoformat(),
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'column_names': df.columns.tolist(),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'null_counts': df.isnull().sum().to_dict(),
        'memory_usage': int(df.memory_usage(deep=True).sum()),
        'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        'sample_values': {
            col: df[col].dropna().head(5).tolist()
            for col in df.columns
        }
    }

def suggest_column_types(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Automatically detect and suggest column types
    
    Categories:
        - numeric (continuous)
        - categorical
        - datetime
        - text
        - binary
        - identifier (ID columns)
    """
    suggestions = {}
    
    for col in df.columns:
        col_data = df[col].dropna()
        
        if len(col_data) == 0:
            suggestions[col] = {'type': 'unknown', 'confidence': 0}
            continue
        
        # Check if numeric
        if pd.api.types.is_numeric_dtype(col_data):
            unique_ratio = col_data.nunique() / len(col_data)
            
            if unique_ratio > 0.95:
                # Likely an ID column
                suggestions[col] = {
                    'type': 'identifier',
                    'confidence': 0.9,
                    'recommendation': 'Remove from features'
                }
            elif col_data.nunique() < 20:
                # Low cardinality - categorical
                suggestions[col] = {
                    'type': 'categorical',
                    'confidence': 0.85,
                    'recommendation': 'One-hot encode'
                }
            else:
                # Continuous numeric
                suggestions[col] = {
                    'type': 'numeric',
                    'confidence': 0.95,
                    'recommendation': 'Normalize/standardize',
                    'stats': {
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std())
                    }
                }
        
        # Check if datetime
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            suggestions[col] = {
                'type': 'datetime',
                'confidence': 1.0,
                'recommendation': 'Extract features (year, month, day, etc.)'
            }
        
        # String/object type
        else:
            unique_count = col_data.nunique()
            
            if unique_count == 2:
                # Binary categorical
                suggestions[col] = {
                    'type': 'binary',
                    'confidence': 0.9,
                    'recommendation': 'Label encode',
                    'values': col_data.unique().tolist()
                }
            elif unique_count < 50:
                # Categorical
                suggestions[col] = {
                    'type': 'categorical',
                    'confidence': 0.85,
                    'recommendation': 'One-hot or label encode',
                    'cardinality': unique_count
                }
            else:
                # Text or high cardinality
                avg_length = col_data.astype(str).str.len().mean()
                
                if avg_length > 50:
                    suggestions[col] = {
                        'type': 'text',
                        'confidence': 0.8,
                        'recommendation': 'Use NLP techniques (TF-IDF, embeddings)'
                    }
                else:
                    suggestions[col] = {
                        'type': 'high_cardinality_categorical',
                        'confidence': 0.7,
                        'recommendation': 'Consider target encoding or removal'
                    }
    
    return suggestions

def generate_dataset_id(filename: str) -> str:
    """Generate unique dataset ID"""
    timestamp = datetime.utcnow().isoformat()
    hash_input = f"{filename}{timestamp}".encode()
    return hashlib.md5(hash_input).hexdigest()[:12]
```

---

## Preprocessing Configuration

```python
# backend/app/api/routes/preprocessing.py
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any

router = APIRouter()

class PreprocessingConfig(BaseModel):
    dataset_id: str
    operations: List[Dict[str, Any]]

@router.post("/api/data/preprocess")
async def configure_preprocessing(config: PreprocessingConfig):
    """
    Configure preprocessing pipeline
    
    Operations:
        - handle_missing: fillna, drop, interpolate
        - encode: one-hot, label, target
        - scale: standard, minmax, robust
        - feature_engineering: polynomial, interactions
        - outlier_removal: zscore, iqr
    """
    df = load_dataset(config.dataset_id)
    
    for operation in config.operations:
        op_type = operation['type']
        params = operation.get('params', {})
        
        if op_type == 'handle_missing':
            df = handle_missing_values(df, **params)
        elif op_type == 'encode':
            df = encode_categorical(df, **params)
        elif op_type == 'scale':
            df = scale_features(df, **params)
        # ... more operations
    
    # Save preprocessed dataset
    new_id = save_preprocessed_dataset(df, config.dataset_id)
    
    return {
        'preprocessed_dataset_id': new_id,
        'shape': df.shape,
        'columns': df.columns.tolist()
    }
```

---

## Styling

```css
/* frontend/src/components/DataUpload.css */
.data-upload-container {
  padding: 24px;
  max-width: 1200px;
  margin: 0 auto;
}

.upload-section h2 {
  margin-bottom: 20px;
  color: #333;
}

.drop-zone {
  border: 2px dashed #cbd5e0;
  border-radius: 12px;
  padding: 60px 40px;
  text-align: center;
  background: #f8f9fa;
  transition: all 0.3s;
  cursor: pointer;
}

.drop-zone.active {
  border-color: #4682B4;
  background: #e3f2fd;
}

.upload-icon {
  color: #4682B4;
  margin-bottom: 16px;
}

.upload-prompt p {
  margin: 8px 0;
  color: #666;
}

.browse-btn {
  margin: 16px 0;
  padding: 12px 32px;
  background: #4682B4;
  color: white;
  border: none;
  border-radius: 6px;
  font-size: 16px;
  cursor: pointer;
  transition: background 0.2s;
}

.browse-btn:hover {
  background: #3a6fa0;
}

.supported-formats {
  font-size: 13px;
  color: #999;
  margin-top: 12px;
}

.upload-progress {
  padding: 20px;
}

.progress-bar {
  width: 100%;
  height: 8px;
  background: #e0e0e0;
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 12px;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #4682B4, #5B9BD5);
  transition: width 0.3s;
}

.upload-success {
  color: #28a745;
}

.checkmark {
  font-size: 48px;
  display: block;
  margin-bottom: 12px;
}

.preview-section {
  margin-top: 32px;
  padding: 24px;
  background: white;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
}

.preview-table-wrapper {
  overflow-x: auto;
  margin: 16px 0;
}

.preview-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

.preview-table th {
  background: #f8f9fa;
  padding: 12px;
  text-align: left;
  border-bottom: 2px solid #dee2e6;
}

.col-type {
  font-size: 11px;
  color: #999;
  font-weight: normal;
  margin-top: 4px;
}

.preview-table td {
  padding: 10px 12px;
  border-bottom: 1px solid #f0f0f0;
}

.null-value {
  color: #999;
  font-style: italic;
}

.preview-actions {
  display: flex;
  gap: 12px;
  margin-top: 16px;
}

.preview-actions button {
  padding: 10px 20px;
  border: 1px solid #4682B4;
  background: white;
  color: #4682B4;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s;
}

.preview-actions button:hover {
  background: #4682B4;
  color: white;
}
```

---

## Future Enhancements

1. **URL Import**: Load datasets from URLs
2. **Database Connectors**: Import from SQL databases
3. **API Integration**: Fetch from REST APIs
4. **Cloud Storage**: S3, Google Cloud Storage
5. **Streaming Upload**: For very large files
6. **Auto-preprocessing**: One-click preprocessing
7. **Data Augmentation**: Generate synthetic samples
8. **Column Mapping**: Map to standard schemas
9. **Validation Rules**: Custom data validation
10. **Version Control**: Track dataset versions
## MVP Additions (Missing)

- Endpoint sketch: POST /datasets/upload → { datasetId, schema, previewRows }
- Client: uploader with drag-drop + file picker, 5–50MB size cap, CSV only (initially).
- Persist last used datasetId; show recent list.
- Error states: size exceeded, parse error, invalid delimiter; retry UX.

## Techstack Interactions

- Backend: FastAPI endpoint streams upload, infers schema (pandas), stores file under `datasets/{id}`.
- Frontend: upload client with drag-drop; after success, writes `datasetId` to context for other routes (Editor room id, charts).
- Security: limit size (e.g., 25MB), validate CSV, sanitize filenames.
