"""
Configuration management for MediLytica backend
"""

import os
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    
    # CORS
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173"
    
    # File Upload
    MAX_UPLOAD_SIZE_MB: int = 25
    MAX_PREVIEW_ROWS: int = 200
    
    # Paths
    DATA_DIR: str = "data"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins into list"""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(',')]


# Global settings instance
settings = Settings()
