# MediLytica - Complete Setup Guide

## 🎯 Project Overview

**MediLytica** is an ML Visualization and Collaboration Platform for healthcare data analysis.

### Architecture
- **Frontend**: React 19 + Vite 7 + Plotly (Port 5173)
- **Backend**: FastAPI + scikit-learn (Port 8000)
- **Features**: Regression analysis, Decision trees, Data upload, Real-time visualization

---

## 🚀 Installation & Setup

### Prerequisites
- **Python 3.9+** (3.11 recommended)
- **Node.js 18+** (for frontend)
- **Git** (for version control)

### Step 1: Clone & Navigate

```bash
git clone <repository-url>
cd Rythm-Hacks/rythmhacks
```

---

## 🔧 Backend Setup

### 1. Navigate to Backend
```bash
cd backend
```

### 2. Create Virtual Environment
```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Start Backend Server
```bash
python run.py
```

**Verify Backend:**
- API Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

---

## 🎨 Frontend Setup

### 1. Navigate to Frontend (New Terminal)
```bash
cd frontend
```

### 2. Install Dependencies
```bash
npm install
```

### 3. Start Frontend Dev Server
```bash
npm run dev
```

**Verify Frontend:**
- App: http://localhost:5173
- Should show MediLytica landing page

---

## ✅ Verification Checklist

### Backend Health Check
```bash
curl http://localhost:8000/health
```
Expected response:
```json
{
  "status": "healthy",
  "services": {
    "api": "operational",
    "ml_engine": "ready"
  }
}
```

### Test Regression API
```bash
curl -X POST http://localhost:8000/api/regression/train \
  -H "Content-Type: application/json" \
  -d "{\"x\": [1,2,3,4,5], \"y\": [2,4,6,8,10], \"regression_type\": \"linear\"}"
```

### Test Frontend-Backend Connection
1. Open http://localhost:5173/dashboard
2. Go to "Regression" category
3. Click "Regenerate" button
4. Should see scatter plot with regression line

---

## 📁 Project Structure

```
rythmhacks/
├── backend/                    # FastAPI Backend
│   ├── app/
│   │   ├── main.py            # FastAPI app
│   │   ├── core/              # ML engines
│   │   │   ├── regression.py
│   │   │   └── decision_tree.py
│   │   ├── api/routes/        # API endpoints
│   │   │   ├── regression.py
│   │   │   ├── decision_tree.py
│   │   │   └── datasets.py
│   │   └── utils/
│   ├── data/                  # Storage
│   ├── requirements.txt       # Python deps
│   ├── run.py                 # Startup script
│   └── README.md
│
├── frontend/                   # React Frontend
│   ├── src/
│   │   ├── App.jsx
│   │   ├── Dashboard.jsx
│   │   ├── Home.jsx
│   │   └── components/
│   │       ├── RegressionPanel.jsx
│   │       ├── DecisionTreePanel.jsx
│   │       ├── Upload.jsx
│   │       └── ...
│   ├── package.json
│   └── vite.config.js
│
└── main.py                    # Legacy (empty)
```

---

## 🔌 API Endpoints

### Regression (`/api/regression/*`)
- **POST /train** - Train linear/polynomial/ridge/lasso regression
- **POST /predict** - Make predictions
- **GET /types** - List regression types

### Decision Trees (`/api/decision-tree/*`)
- **POST /train** - Train classification/regression tree
- **POST /predict** - Make predictions
- **GET /{session_id}/structure** - Get tree structure
- **GET /parameters** - List parameters

### Datasets (`/datasets/*`)
- **POST /upload** - Upload CSV file (max 25MB)
- **GET /list** - List uploaded datasets
- **GET /{dataset_id}** - Get dataset info

---

## 🧪 Testing the Application

### 1. Upload Dataset
```bash
# Go to http://localhost:5173/dashboard
# Click "Upload CSV"
# Drop a CSV file or select from file browser
# Should see data preview
```

### 2. Train Regression Model
```bash
# In Dashboard, select "Regression" category
# Adjust parameters if needed
# Click "Regenerate"
# Should see scatter plot with fitted line and metrics
```

### 3. Train Decision Tree
```bash
# Select "Decision Tree" category
# Adjust max depth slider
# Click "Retrain"
# Should see tree visualization with nodes
```

---

## 🐛 Troubleshooting

### Backend Issues

**Import errors when running backend:**
```bash
# Ensure virtual environment is activated
# Windows:
.\venv\Scripts\Activate.ps1

# Reinstall requirements
pip install -r requirements.txt
```

**Port 8000 already in use:**
```bash
# Find and kill process on Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Or change port in backend/run.py
```

**CORS errors in browser console:**
- Ensure backend is running on port 8000
- Check CORS middleware in `backend/app/main.py`
- Frontend should be on port 5173

### Frontend Issues

**`npm install` fails:**
```bash
# Clear cache and retry
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

**Blank page or console errors:**
- Check browser console for errors
- Ensure backend is running (http://localhost:8000/health)
- Verify frontend is running on port 5173

**Components not loading:**
```bash
# Rebuild
npm run build
npm run dev
```

---

## 🚢 Deployment

### Backend Deployment (Railway/Render)

1. **Prepare for deployment:**
```bash
# Add Procfile in backend/
echo "web: uvicorn app.main:app --host 0.0.0.0 --port $PORT" > backend/Procfile
```

2. **Environment variables:**
- `PORT`: Auto-set by platform
- `CORS_ORIGINS`: https://your-frontend-domain.com

### Frontend Deployment (Vercel/Netlify)

1. **Build command:** `npm run build`
2. **Output directory:** `dist`
3. **Environment variables:**
   - `VITE_API_URL`: https://your-backend-domain.com

---

## 📊 Features Status

| Feature | Backend | Frontend | Status |
|---------|---------|----------|--------|
| Linear Regression | ✅ | ✅ | Working |
| Polynomial Regression | ✅ | ✅ | Working |
| Ridge/Lasso | ✅ | ⚠️ Partial | Needs UI |
| Decision Trees | ✅ | ✅ | Working |
| CSV Upload | ✅ | ✅ | Working |
| Data Table | ❌ | ✅ | Frontend only |
| Chat | ❌ | ✅ | Frontend only |
| Neural Networks | ❌ | ⚠️ Partial | Mock data |

Legend: ✅ Complete | ⚠️ Partial | ❌ Not implemented

---

## 📝 Development Workflow

### Making Changes

1. **Backend changes:**
   - Edit files in `backend/app/`
   - Server auto-reloads (uvicorn --reload)
   - Test at http://localhost:8000/docs

2. **Frontend changes:**
   - Edit files in `frontend/src/`
   - Vite hot-reloads automatically
   - View at http://localhost:5173

### Adding New ML Models

1. Create engine in `backend/app/core/`
2. Create API routes in `backend/app/api/routes/`
3. Register router in `backend/app/main.py`
4. Create frontend component in `frontend/src/components/`
5. Integrate in `Dashboard.jsx`

---

## 🔐 Security Notes

- Max upload size: 25MB (configurable in `.env`)
- No authentication implemented yet (planned)
- Pickle files used for model storage (fine for development)
- CORS restricted to localhost (update for production)

---

## 📚 Documentation

- **Backend API**: http://localhost:8000/docs
- **Backend Docs**: `backend/docs/`
- **Frontend Docs**: `frontend/docs/`
- **Tech Stack**: `backend/docs/techstack.md`

---

## 🤝 Contributing

1. Create feature branch
2. Make changes
3. Test locally
4. Submit pull request

---

## 📄 License

See LICENSE file in repository root.

---

## 🆘 Support

- Check documentation in `backend/docs/` and `frontend/docs/`
- Review API documentation at http://localhost:8000/docs
- Check browser console for frontend errors
- Check terminal for backend errors

---

**Last Updated:** October 25, 2025  
**Version:** 1.0.0  
**Status:** Ready for Development ✅
