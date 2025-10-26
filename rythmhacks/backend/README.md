# MediLytica Backend

ML Visualization and Collaboration Platform - Backend API Server

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Run the Server

```bash
# Simple start
python run.py

# Or using uvicorn directly
uvicorn app.main:app --reload --port 8000
```

### 3. Access the API

- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 📁 Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration
│   ├── core/                # ML engines
│   │   ├── regression.py    # Regression models
│   │   └── decision_tree.py # Decision tree models
│   ├── api/
│   │   └── routes/          # API endpoints
│   │       ├── regression.py
│   │       ├── decision_tree.py
│   │       └── datasets.py
│   ├── models/              # Database models (future)
│   └── utils/               # Helper functions
│       └── helpers.py
├── data/                    # Data storage
│   ├── regression_sessions/ # Trained regression models
│   ├── tree_sessions/       # Trained tree models
│   └── datasets/            # Uploaded datasets
├── docs/                    # Documentation
├── requirements.txt         # Python dependencies
├── run.py                   # Startup script
└── .env.example             # Environment configuration template
```

## 🔌 API Endpoints

### Regression

- `POST /api/regression/train` - Train regression model
- `POST /api/regression/predict` - Make predictions
- `GET /api/regression/types` - List available regression types

### Decision Trees

- `POST /api/decision-tree/train` - Train decision tree
- `POST /api/decision-tree/predict` - Make predictions
- `GET /api/decision-tree/{session_id}/structure` - Get tree structure
- `GET /api/decision-tree/parameters` - List available parameters

### Datasets

- `POST /datasets/upload` - Upload CSV file
- `GET /datasets/list` - List uploaded datasets
- `GET /datasets/{dataset_id}` - Get dataset info

## 🧪 Testing the API

### Using Python Requests

```python
import requests

# Train a regression model
response = requests.post('http://localhost:8000/api/regression/train', json={
    'x': [1, 2, 3, 4, 5],
    'y': [2, 4, 6, 8, 10],
    'regression_type': 'linear'
})

result = response.json()
print(f"R² Score: {result['metrics']['r2']}")
print(f"Session ID: {result['session_id']}")
```

### Using cURL

```bash
# Train decision tree
curl -X POST http://localhost:8000/api/decision-tree/train \
  -H "Content-Type: application/json" \
  -d '{
    "X": [[1, 2], [3, 4], [5, 6]],
    "y": [0, 1, 0],
    "task_type": "classification",
    "max_depth": 3
  }'
```

## 🛠️ Development

### Adding New Features

1. Create ML engine in `app/core/`
2. Create API routes in `app/api/routes/`
3. Add router to `app/main.py`
4. Update documentation

### Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

## 📊 Features Implemented

- ✅ Linear Regression
- ✅ Polynomial Regression
- ✅ Ridge & Lasso Regression
- ✅ Decision Tree Classification
- ✅ Decision Tree Regression
- ✅ CSV Dataset Upload
- ✅ Model Metrics Calculation
- ✅ Session Management

## 🔜 Coming Soon

- WebSocket support for real-time training updates
- Database integration (SQLAlchemy)
- User authentication
- Neural network training
- Model performance visualization
- Collaborative features

## 📝 Documentation

Full documentation available in `docs/` directory:

- Architecture: `docs/techstack.md`
- Feature specs: `docs/feature-*.md`
- API Reference: http://localhost:8000/docs

## 🐛 Troubleshooting

**Import errors?**
```bash
pip install -r requirements.txt
```

**Port already in use?**
```bash
# Change port in run.py or use:
uvicorn app.main:app --reload --port 8001
```

**CORS errors?**
- Check frontend URL in `app/main.py` CORS middleware
- Ensure frontend is running on port 5173

## 📄 License

See LICENSE file in root directory.
