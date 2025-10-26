"""
UMAP Analyzer
Dimensionality reduction using UMAP for dataset visualization
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler


def compute_umap_projection(
    csv_path: str,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'euclidean',
    random_state: int = 42
) -> Dict:
    """
    Compute UMAP 2D projection of a dataset
    
    Args:
        csv_path: Path to CSV file
        n_neighbors: Size of local neighborhood (5-50, default: 15)
        min_dist: Minimum distance between points (0.0-1.0, default: 0.1)
        metric: Distance metric ('euclidean', 'manhattan', 'cosine')
        random_state: Random seed for reproducibility
    
    Returns:
        dict: {
            'embedding': List of [x, y] coordinates,
            'feature_names': List of feature names used,
            'n_samples': Number of data points,
            'n_features': Number of features used,
            'parameters': Dict of UMAP parameters used
        }
    
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If no numeric columns found or invalid parameters
        ImportError: If umap-learn is not installed
    """
    try:
        import umap
    except ImportError:
        raise ImportError("umap-learn is not installed. Run: pip install umap-learn")
    
    # Validate parameters
    if not (5 <= n_neighbors <= 50):
        raise ValueError("n_neighbors must be between 5 and 50")
    if not (0.0 <= min_dist <= 1.0):
        raise ValueError("min_dist must be between 0.0 and 1.0")
    if metric not in ['euclidean', 'manhattan', 'cosine']:
        raise ValueError("metric must be 'euclidean', 'manhattan', or 'cosine'")
    
    # Load dataset
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {str(e)}")
    
    if df.empty:
        raise ValueError("Dataset is empty")
    
    # Extract numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        raise ValueError("No numeric columns found in dataset. UMAP requires numeric data.")
    
    # Remove columns with NaN or infinite values
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
    numeric_df = numeric_df.dropna(axis=1)
    
    if numeric_df.empty:
        raise ValueError("No valid numeric columns after removing NaN/Inf values")
    
    # Store feature names
    feature_names = numeric_df.columns.tolist()
    n_samples = len(numeric_df)
    n_features = len(feature_names)
    
    # Ensure enough samples for n_neighbors
    if n_samples < n_neighbors:
        n_neighbors = max(2, n_samples - 1)
        print(f"Warning: n_neighbors reduced to {n_neighbors} due to small sample size")
    
    # Standardize features (important for UMAP)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)
    
    # Apply UMAP
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric=metric,
        random_state=random_state
    )
    
    embedding = reducer.fit_transform(X_scaled)
    
    # Convert to list of [x, y] coordinates
    embedding_list = embedding.tolist()
    
    return {
        'embedding': embedding_list,
        'feature_names': feature_names,
        'n_samples': n_samples,
        'n_features': n_features,
        'parameters': {
            'n_neighbors': n_neighbors,
            'min_dist': min_dist,
            'metric': metric,
            'random_state': random_state
        }
    }


def get_embedding_statistics(embedding: List[List[float]]) -> Dict:
    """
    Calculate statistics about the UMAP embedding
    
    Args:
        embedding: List of [x, y] coordinates
    
    Returns:
        dict: Statistics including bounds, mean, std
    """
    embedding_array = np.array(embedding)
    
    return {
        'x_min': float(embedding_array[:, 0].min()),
        'x_max': float(embedding_array[:, 0].max()),
        'y_min': float(embedding_array[:, 1].min()),
        'y_max': float(embedding_array[:, 1].max()),
        'x_mean': float(embedding_array[:, 0].mean()),
        'y_mean': float(embedding_array[:, 1].mean()),
        'x_std': float(embedding_array[:, 0].std()),
        'y_std': float(embedding_array[:, 1].std())
    }
