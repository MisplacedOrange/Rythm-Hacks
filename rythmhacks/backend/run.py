"""
MediLytica Backend Server Startup Script
Run this script to start the backend API server
"""

import uvicorn
from pathlib import Path
import sys

# Add parent directory to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

if __name__ == "__main__":
    print("=" * 60)
    print("MediLytica Backend Server")
    print("=" * 60)
    print()
    print("ML Visualization and Collaboration Platform")
    print()
    print("Starting server...")
    print("- API Documentation: http://localhost:8000/docs")
    print("- Health Check: http://localhost:8000/health")
    print("- CORS Enabled for: http://localhost:5173")
    print()
    print("Press CTRL+C to stop the server")
    print("=" * 60)
    print()
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
