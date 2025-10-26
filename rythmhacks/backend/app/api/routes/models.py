"""
Model Upload and Management Routes
Handles uploading trained ML models and retrieving their performance metrics
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse
from pathlib import Path
from pydantic import BaseModel
import pickle
import joblib
import hashlib
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List
from app.core.model_analyzer import ModelAnalyzer

router = APIRouter(prefix="/api/models")

# Pydantic models for request bodies
class MetricsRequest(BaseModel):
    """Request body for metrics calculation"""
    X_test: List[List[float]]  # Test features
    y_test: List[float]  # Test labels
    feature_names: Optional[List[str]] = None
    model_type: str = "classifier"  # or "regressor"

# Storage directory for uploaded models
MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Metadata storage
METADATA_DIR = Path("data/models_metadata")
METADATA_DIR.mkdir(parents=True, exist_ok=True)

# File size limit: 100MB
MAX_FILE_SIZE = 100 * 1024 * 1024

# Supported model file extensions
SUPPORTED_EXTENSIONS = {
    '.pkl': 'sklearn',
    '.joblib': 'sklearn',
    '.h5': 'keras',
    '.pt': 'pytorch',
    '.pth': 'pytorch'
}


@router.post("/upload")
async def upload_model(file: UploadFile = File(...)):
    """
    Upload a trained ML model file
    
    Supported formats:
    - .pkl, .joblib (scikit-learn)
    - .h5 (Keras/TensorFlow)
    - .pt, .pth (PyTorch)
    
    Returns:
        - model_id: Unique identifier for the uploaded model
        - filename: Original filename
        - framework: Detected framework (sklearn, keras, pytorch)
        - upload_date: ISO timestamp
        - file_size: Size in bytes
    """
    try:
        # Read file contents
        contents = await file.read()
        
        # Validate file size
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        # Validate file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_ext}. Supported: {', '.join(SUPPORTED_EXTENSIONS.keys())}"
            )
        
        framework = SUPPORTED_EXTENSIONS[file_ext]
        
        # Generate unique model ID
        model_id = generate_model_id(file.filename, contents)
        
        # Save model file
        model_path = MODELS_DIR / f"{model_id}{file_ext}"
        with open(model_path, 'wb') as f:
            f.write(contents)
        
        # Basic model validation
        model_info = validate_model(model_path, framework)
        
        # Create metadata
        metadata = {
            'model_id': model_id,
            'filename': file.filename,
            'framework': framework,
            'file_extension': file_ext,
            'upload_date': datetime.utcnow().isoformat(),
            'file_size': len(contents),
            'file_size_mb': round(len(contents) / (1024 * 1024), 2),
            'model_path': str(model_path),
            'model_info': model_info
        }
        
        # Save metadata
        metadata_path = METADATA_DIR / f"{model_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            'model_id': model_id,
            'filename': file.filename,
            'framework': framework,
            'upload_date': metadata['upload_date'],
            'file_size': len(contents),
            'file_size_mb': metadata['file_size_mb'],
            'model_type': model_info.get('type', 'unknown'),
            'message': 'Model uploaded successfully'
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload model: {str(e)}"
        )


@router.get("/{model_id}")
async def get_model_info(model_id: str):
    """Get metadata for an uploaded model"""
    metadata_path = METADATA_DIR / f"{model_id}.json"
    
    if not metadata_path.exists():
        raise HTTPException(404, "Model not found")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return metadata


@router.get("/")
async def list_models():
    """List all uploaded models"""
    models = []
    
    for metadata_file in METADATA_DIR.glob("*.json"):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            models.append({
                'model_id': metadata['model_id'],
                'filename': metadata['filename'],
                'framework': metadata['framework'],
                'upload_date': metadata['upload_date'],
                'file_size_mb': metadata['file_size_mb'],
                'model_type': metadata.get('model_info', {}).get('type', 'unknown')
            })
    
    # Sort by upload date (newest first)
    models.sort(key=lambda x: x['upload_date'], reverse=True)
    
    return {'models': models, 'total': len(models)}


@router.post("/{model_id}/metrics")
async def calculate_metrics(model_id: str, request: MetricsRequest):
    """
    Calculate comprehensive metrics for uploaded model
    
    Requires test data (X_test, y_test) to evaluate model
    
    Returns:
        - overall_metrics: accuracy, f1, precision, recall (classification)
                          or mse, rmse, r2 (regression)
        - confusion_matrix: 2D array (classification only)
        - roc_data: ROC curves (classification only)
        - feature_importance: Feature importance scores
    """
    # Get model metadata
    metadata_path = METADATA_DIR / f"{model_id}.json"
    
    if not metadata_path.exists():
        raise HTTPException(404, "Model not found")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    model_path = Path(metadata['model_path'])
    framework = metadata['framework']
    
    if not model_path.exists():
        raise HTTPException(404, "Model file not found")
    
    try:
        # Convert test data to numpy arrays
        X_test = np.array(request.X_test)
        y_test = np.array(request.y_test)
        
        # Analyze model
        metrics = ModelAnalyzer.analyze_model(
            model_path=model_path,
            framework=framework,
            X_test=X_test,
            y_test=y_test,
            feature_names=request.feature_names,
            model_type=request.model_type
        )
        
        # Cache metrics in metadata
        metadata['cached_metrics'] = metrics
        metadata['metrics_updated'] = datetime.utcnow().isoformat()
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metrics
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to calculate metrics: {str(e)}"
        )


@router.get("/{model_id}/metrics")
async def get_cached_metrics(model_id: str):
    """Get previously calculated metrics from cache"""
    metadata_path = METADATA_DIR / f"{model_id}.json"
    
    if not metadata_path.exists():
        raise HTTPException(404, "Model not found")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    if 'cached_metrics' not in metadata:
        raise HTTPException(
            400,
            "No cached metrics. Calculate metrics first using POST /{model_id}/metrics"
        )
    
    return metadata['cached_metrics']


@router.delete("/{model_id}")
async def delete_model(model_id: str):
    """Delete an uploaded model and its metadata"""
    metadata_path = METADATA_DIR / f"{model_id}.json"
    
    if not metadata_path.exists():
        raise HTTPException(404, "Model not found")
    
    # Load metadata to get model path
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Delete model file
    model_path = Path(metadata['model_path'])
    if model_path.exists():
        model_path.unlink()
    
    # Delete metadata
    metadata_path.unlink()
    
    return {'message': 'Model deleted successfully', 'model_id': model_id}


def generate_model_id(filename: str, contents: bytes) -> str:
    """Generate unique model ID based on filename and content hash"""
    timestamp = datetime.utcnow().isoformat()
    hash_input = f"{filename}{timestamp}".encode() + contents[:1000]  # Use first 1KB for hash
    return hashlib.md5(hash_input).hexdigest()[:16]


def validate_model(model_path: Path, framework: str) -> Dict[str, Any]:
    """
    Validate model file and extract basic information
    
    Returns:
        - type: Model type (classifier, regressor, neural_network)
        - valid: Whether model loaded successfully
        - error: Error message if validation failed
    """
    info = {
        'valid': False,
        'type': 'unknown',
        'error': None
    }
    
    try:
        if framework == 'sklearn':
            # Try loading with pickle or joblib
            try:
                model = pickle.load(open(model_path, 'rb'))
            except:
                model = joblib.load(model_path)
            
            # Determine model type
            model_class = model.__class__.__name__
            
            if hasattr(model, 'predict_proba'):
                info['type'] = 'classifier'
            elif 'Regressor' in model_class or 'Regression' in model_class:
                info['type'] = 'regressor'
            elif 'Classifier' in model_class or 'Classification' in model_class:
                info['type'] = 'classifier'
            else:
                info['type'] = 'unknown'
            
            info['model_class'] = model_class
            info['valid'] = True
            
        elif framework == 'pytorch':
            # Basic PyTorch model validation
            try:
                import torch
                model = torch.load(model_path, map_location='cpu')
                info['type'] = 'neural_network'
                info['valid'] = True
                info['model_class'] = 'PyTorch Model'
            except ImportError:
                info['error'] = 'PyTorch not installed'
            except Exception as e:
                info['error'] = str(e)
        
        elif framework == 'keras':
            # Basic Keras model validation
            try:
                from tensorflow import keras
                model = keras.models.load_model(model_path)
                info['type'] = 'neural_network'
                info['valid'] = True
                info['model_class'] = 'Keras Model'
            except ImportError:
                info['error'] = 'TensorFlow/Keras not installed'
            except Exception as e:
                info['error'] = str(e)
    
    except Exception as e:
        info['error'] = f"Validation error: {str(e)}"
    
    return info
