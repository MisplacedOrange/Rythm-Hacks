"""
Dataset Upload API Routes
Endpoint for uploading CSV files (required by Upload.jsx frontend component)
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from pathlib import Path
import csv
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.utils.helpers import generate_unique_id, validate_file_extension
from app.core.umap_analyzer import compute_umap_projection, get_embedding_statistics

router = APIRouter(prefix="/datasets")

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB
MAX_PREVIEW_ROWS = 200


def detect_column_type(values: List[str]) -> str:
    """Detect column data type from sample values"""
    # Try to detect if numeric
    numeric_count = 0
    for val in values[:100]:  # Sample first 100 values
        if val.strip():
            try:
                float(val)
                numeric_count += 1
            except ValueError:
                pass
    
    if numeric_count / len(values) > 0.8:
        return 'numeric'
    
    # Check if categorical (low cardinality)
    unique_values = len(set(values))
    if unique_values < 20:
        return 'categorical'
    
    return 'string'


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload CSV dataset
    
    Matches the API expected by Upload.jsx component
    
    Request:
        file: CSV file upload
    
    Returns:
        datasetId: Unique dataset identifier
        schema: Column names and types
        preview: First N rows of data
    """
    try:
        # Validate file extension
        if not validate_file_extension(file.filename, ['.csv']):
            raise HTTPException(400, "Only CSV files are supported")
        
        # Read file content
        content = await file.read()
        
        # Check file size
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(400, f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB")
        
        # Decode content
        try:
            text_content = content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                text_content = content.decode('latin-1')
            except:
                raise HTTPException(400, "Unable to decode file. Please ensure it's a valid CSV")
        
        # Parse CSV
        lines = text_content.strip().split('\n')
        if len(lines) == 0:
            raise HTTPException(400, "Empty file")
        
        # Parse header
        reader = csv.reader(lines)
        rows = list(reader)
        
        if len(rows) < 1:
            raise HTTPException(400, "No data found in file")
        
        header = rows[0]
        data_rows = rows[1:MAX_PREVIEW_ROWS + 1]  # Limit preview
        
        # Build schema with type detection
        schema = []
        for col_idx, col_name in enumerate(header):
            col_values = [row[col_idx] if col_idx < len(row) else '' for row in data_rows]
            col_type = detect_column_type(col_values)
            
            schema.append({
                'name': col_name.strip(),
                'type': col_type
            })
        
        # Generate dataset ID
        dataset_id = generate_unique_id(prefix='ds')
        
        # Save file to disk
        dataset_dir = Path("data/datasets")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = dataset_dir / f"{dataset_id}.csv"
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # Save metadata
        metadata = {
            'id': dataset_id,
            'filename': file.filename,
            'uploaded_at': datetime.utcnow().isoformat(),
            'total_rows': len(rows) - 1,  # Exclude header
            'total_columns': len(header),
            'schema': schema,
            'file_path': str(file_path)
        }
        
        metadata_path = dataset_dir / f"{dataset_id}_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            'datasetId': dataset_id,
            'schema': schema,
            'preview': data_rows
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {str(e)}")


@router.get("/list")
async def list_datasets():
    """List all uploaded datasets"""
    dataset_dir = Path("data/datasets")
    if not dataset_dir.exists():
        return {'datasets': []}
    
    datasets = []
    for metadata_file in dataset_dir.glob("*_metadata.json"):
        try:
            import json
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                datasets.append({
                    'id': metadata['id'],
                    'filename': metadata['filename'],
                    'uploaded_at': metadata['uploaded_at'],
                    'rows': metadata['total_rows'],
                    'columns': metadata['total_columns']
                })
        except:
            continue
    
    # Sort by upload time (newest first)
    datasets.sort(key=lambda x: x['uploaded_at'], reverse=True)
    
    return {'datasets': datasets}


@router.get("/{dataset_id}")
async def get_dataset_info(dataset_id: str):
    """Get dataset metadata and preview"""
    metadata_path = Path(f"data/datasets/{dataset_id}_metadata.json")
    
    if not metadata_path.exists():
        raise HTTPException(404, "Dataset not found")
    
    try:
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Read preview data
        file_path = Path(metadata['file_path'])
        if file_path.exists():
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)
                preview = rows[1:MAX_PREVIEW_ROWS + 1]  # Skip header
        else:
            preview = []
        
        return {
            'datasetId': dataset_id,
            'schema': metadata['schema'],
            'preview': preview,
            'metadata': {
                'filename': metadata['filename'],
                'uploaded_at': metadata['uploaded_at'],
                'total_rows': metadata['total_rows'],
                'total_columns': metadata['total_columns']
            }
        }
    
    except Exception as e:
        raise HTTPException(500, f"Failed to load dataset: {str(e)}")


class UMAPRequest(BaseModel):
    """Request model for UMAP computation"""
    n_neighbors: Optional[int] = 15
    min_dist: Optional[float] = 0.1
    metric: Optional[str] = 'euclidean'


@router.post("/{dataset_id}/umap")
async def compute_dataset_umap(dataset_id: str, request: UMAPRequest):
    """
    Compute UMAP 2D projection for a dataset
    
    Applies UMAP dimensionality reduction to visualize high-dimensional data
    in 2D space. Only numeric columns are used for the projection.
    
    Args:
        dataset_id: Dataset identifier
        request: UMAP parameters (n_neighbors, min_dist, metric)
    
    Returns:
        embedding: Array of [x, y] coordinates for each data point
        feature_names: List of numeric features used
        n_samples: Number of data points
        n_features: Number of features
        parameters: UMAP parameters used
        statistics: Embedding bounds and statistics
    
    Raises:
        404: Dataset not found
        400: Invalid parameters or no numeric data
        500: Computation failed
    """
    # Find dataset file
    metadata_path = Path(f"data/datasets/{dataset_id}_metadata.json")
    
    if not metadata_path.exists():
        raise HTTPException(404, "Dataset not found")
    
    try:
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        file_path = Path(metadata['file_path'])
        
        if not file_path.exists():
            raise HTTPException(404, "Dataset file not found")
        
        # Compute UMAP projection
        result = compute_umap_projection(
            csv_path=str(file_path),
            n_neighbors=request.n_neighbors,
            min_dist=request.min_dist,
            metric=request.metric
        )
        
        # Calculate embedding statistics
        stats = get_embedding_statistics(result['embedding'])
        
        return {
            'embedding': result['embedding'],
            'feature_names': result['feature_names'],
            'n_samples': result['n_samples'],
            'n_features': result['n_features'],
            'parameters': result['parameters'],
            'statistics': stats
        }
    
    except ValueError as e:
        raise HTTPException(400, str(e))
    except ImportError as e:
        raise HTTPException(500, str(e))
    except Exception as e:
        raise HTTPException(500, f"UMAP computation failed: {str(e)}")
