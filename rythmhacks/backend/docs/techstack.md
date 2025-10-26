# Backend Tech Stack - MediLytica

## Architecture Overview

**Data Flow:**
```
User (Browser) 
    ↕ WebSocket + REST API
Backend Server (FastAPI)
    ↕
ML Training Engine (PyTorch/scikit-learn)
    ↕
Results Storage (Database/File System)
    ↕
Real-time Broadcast (WebSocket)
    ↕
All Connected Users
```

---

## Core Technologies

### **1. Web Framework: FastAPI**
- **Why**: Modern, fast, async support, automatic API docs, WebSocket support
- **Version**: Latest stable
- **Install**: `pip install fastapi uvicorn`

**Key Features:**
- REST API endpoints for model training, data upload, configuration
- WebSocket endpoints for real-time collaboration
- Automatic data validation with Pydantic
- Built-in OpenAPI documentation

---

### **2. WebSocket Server: FastAPI WebSockets**
- **Why**: Real-time bidirectional communication for live training updates and collaboration
- **Install**: `pip install python-socketio`

**Use Cases:**
- Stream training metrics in real-time (loss, accuracy per epoch)
- Broadcast model snapshots to all connected users
- Synchronize chat messages across users
- Share code editor changes (two-user collaboration MVP)`r`n- Notify when training completes

---

### **3. Machine Learning Frameworks**

#### **PyTorch**
- **Why**: Deep learning, neural networks, CNNs
- **Install**: `pip install torch torchvision`
- **Use**: Neural network training and inference

#### **scikit-learn**
- **Why**: Traditional ML algorithms (Decision Trees, Random Forest, Regression)
- **Install**: `pip install scikit-learn`
- **Use**: Classical ML models

#### **NumPy & Pandas**
- **Install**: `pip install numpy pandas`
- **Use**: Data processing, manipulation, feature engineering

---

### **4. Data Storage**

#### **Option A: SQLite (Development/MVP)**
- **Why**: Simple, no setup, file-based
- **Use**: Store training history, user sessions, model metadata
- Built into Python

#### **Option B: PostgreSQL (Production)**
- **Why**: Robust, scalable, handles concurrent users
- **Install**: `pip install psycopg2-binary sqlalchemy`
- **Use**: Production deployment

#### **File Storage**
- **Trained Models**: Save as `.pth` (PyTorch) or `.pkl` (scikit-learn)
- **Training Results**: JSON files with metrics, visualizations data
- **Datasets**: CSV, Parquet files

---

### **5. Task Queue (Optional - For Heavy Training)**

#### **Celery + Redis**
- **Why**: Offload long-running training tasks to background workers
- **Install**: `pip install celery redis`
- **Use**: Asynchronous model training, prevents blocking API

---

### **6. Monitoring & Logging**

#### **Python Logging**
- Built-in, captures training progress, errors

#### **TensorBoard (Optional)**
- **Install**: `pip install tensorboard`
- **Use**: Visualize training metrics, can export data for frontend

---

## API Endpoints Design

### **REST API Endpoints**

```python
# Training
POST   /api/train/start          # Start model training
GET    /api/train/status/{id}    # Get training status
POST   /api/train/stop/{id}      # Stop training
GET    /api/train/results/{id}   # Get training results

# Models
GET    /api/models               # List available models
GET    /api/models/{id}          # Get model details
POST   /api/models/upload        # Upload trained model
DELETE /api/models/{id}          # Delete model

# Data
POST   /api/data/upload          # Upload dataset
GET    /api/data/preview/{id}    # Preview dataset

# Visualization
GET    /api/viz/graph/{id}       # Get network graph data
GET    /api/viz/metrics/{id}     # Get performance metrics
GET    /api/viz/confusion/{id}   # Get confusion matrix
GET    /api/viz/roc/{id}         # Get ROC curve data
```

### **WebSocket Endpoints**

```python
# Real-time Updates
WS /ws/training/{session_id}     # Stream training progress
WS /ws/chat/{room_id}            # Team chat
WS /ws/code/{room_id}            # Collaborative code editing
```

---

## Frontend + Backend Connection

### **HTTP Requests (REST API)**
```javascript
// Frontend (React)
const startTraining = async (config) => {
  const response = await fetch('http://localhost:8000/api/train/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config)
  })
  return response.json()
}
```

### **WebSocket Connection (Real-time)**
```javascript
// Frontend (React)
import { useEffect, useState } from 'react'

const useTrainingStream = (sessionId) => {
  const [metrics, setMetrics] = useState([])
  
  useEffect(() => {
    const ws = new WebSocket(`ws://localhost:8000/ws/training/${sessionId}`)
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      setMetrics(prev => [...prev, data])
    }
    
    return () => ws.close()
  }, [sessionId])
  
  return metrics
}
```

---

## Backend Implementation Structure

```
rythmhacks/backend/
├── app/
│   ├── main.py                 # FastAPI app entry point
│   ├── api/
│   │   ├── routes/
│   │   │   ├── training.py     # Training endpoints
│   │   │   ├── models.py       # Model management
│   │   │   ├── data.py         # Data upload/processing
│   │   │   └── visualization.py # Viz data endpoints
│   │   └── websockets/
│   │       ├── training.py     # Training stream
│   │       └── collaboration.py # Chat/code sync
│   ├── ml/
│   │   ├── trainers/
│   │   │   ├── neural_network.py
│   │   │   ├── decision_tree.py
│   │   │   └── regression.py
│   │   ├── evaluators/
│   │   │   └── metrics.py      # Calculate performance metrics
│   │   └── visualizers/
│   │       └── graph_generator.py # Generate network graphs
│   ├── models/                 # Database models (SQLAlchemy)
│   ├── schemas/                # Pydantic schemas
│   └── utils/
│       ├── storage.py          # File storage utilities
│       └── logger.py           # Logging setup
├── docs/
│   └── techstack.md           # This file
├── requirements.txt
└── .env
```

---

## Installation & Setup

### **1. Create Virtual Environment**
```bash
cd rythmhacks/backend
python -m venv venv
venv\Scripts\activate  # Windows
```

### **2. Install Dependencies**
```bash
pip install fastapi uvicorn websockets
pip install torch torchvision
pip install scikit-learn numpy pandas
pip install sqlalchemy
pip install python-multipart  # For file uploads
pip install pydantic-settings  # For config
```

### **3. Run Development Server**
```bash
uvicorn app.main:app --reload --port 8000
```

---

## Environment Variables (.env)

```bash
# Server
HOST=0.0.0.0
PORT=8000
CORS_ORIGINS=http://localhost:5173,http://localhost:3000

# Database
DATABASE_URL=sqlite:///./medilytica.db

# Storage
MODEL_STORAGE_PATH=./storage/models
DATA_STORAGE_PATH=./storage/data
RESULTS_STORAGE_PATH=./storage/results

# ML
MAX_TRAINING_TIME=3600  # 1 hour
DEFAULT_BATCH_SIZE=64
```

---

## Data Format for Frontend Visualization

### **Training Metrics (Streamed via WebSocket)**
```json
{
  "epoch": 5,
  "loss": 0.234,
  "accuracy": 0.89,
  "val_loss": 0.256,
  "val_accuracy": 0.87,
  "timestamp": "2025-10-25T20:30:00Z"
}
```

### **Network Graph Data**
```json
{
  "nodes": [
    { "id": "node1", "label": "Conv2D(32)", "type": "conv", "color": "#F4A460" },
    { "id": "node2", "label": "ReLU", "type": "activation", "color": "#4682B4" }
  ],
  "edges": [
    { "from": "node1", "to": "node2", "weight": 0.8 }
  ]
}
```

### **Performance Metrics**
```json
{
  "confusion_matrix": [[50, 2], [3, 45]],
  "roc_curve": {
    "fpr": [0, 0.1, 0.2, ...],
    "tpr": [0, 0.8, 0.9, ...],
    "auc": 0.95
  },
  "feature_importance": {
    "features": ["Feature1", "Feature2", "Feature3"],
    "scores": [0.8, 0.6, 0.4]
  },
  "training_history": {
    "epochs": [1, 2, 3, ...],
    "train_loss": [0.5, 0.4, 0.3, ...],
    "val_loss": [0.52, 0.42, 0.32, ...]
  }
}
```

---

## Security Considerations

1. **CORS**: Configure allowed origins
2. **Rate Limiting**: Prevent API abuse
3. **Authentication**: JWT tokens for user sessions (future)
4. **Input Validation**: Pydantic schemas validate all inputs
5. **File Upload Limits**: Max file size for datasets/models
6. **WebSocket Authentication**: Verify session tokens

---

## Scalability Strategy

### **Phase 1: MVP (Single Server)**
- FastAPI + SQLite
- Single server handles everything
- Good for demos and small teams

### **Phase 2: Production (Multi-Server)**
- PostgreSQL database
- Redis for session storage & caching
- Celery workers for training tasks
- Load balancer for API servers
- Separate WebSocket server

### **Phase 3: Cloud (Kubernetes)**
- Container orchestration
- Auto-scaling workers
- Cloud storage (S3/GCS)
- GPU instances for training

---

## Testing Strategy

```bash
# Install testing dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/
```

---

## Next Steps

1. Set up basic FastAPI server with CORS
2. Implement `/api/train/start` endpoint
3. Create WebSocket endpoint for training stream
4. Test connection with React frontend
5. Implement model training pipeline
6. Add visualization data generation
7. Deploy to development server

