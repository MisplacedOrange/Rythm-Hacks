"""
Utility functions for MediLytica backend
"""

import uuid
from datetime import datetime
from typing import Dict, Any


def generate_unique_id(prefix: str = "") -> str:
    """
    Generate a unique session ID
    
    Args:
        prefix: Optional prefix for the ID
    
    Returns:
        Unique string ID
    """
    unique_id = str(uuid.uuid4())
    if prefix:
        return f"{prefix}_{unique_id}"
    return unique_id


def format_timestamp() -> str:
    """Get current timestamp in ISO format"""
    return datetime.utcnow().isoformat()


def validate_file_extension(filename: str, allowed_extensions: list) -> bool:
    """
    Check if file has allowed extension
    
    Args:
        filename: Name of the file
        allowed_extensions: List of allowed extensions (e.g., ['.csv', '.xlsx'])
    
    Returns:
        True if valid, False otherwise
    """
    return any(filename.lower().endswith(ext) for ext in allowed_extensions)


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to int"""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default
