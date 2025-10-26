# 🚀 Quick Start - MediLytica

## Get Running in 3 Minutes

### Step 1: Install Backend Dependencies (1 min)
```powershell
cd rythmhacks\backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Step 2: Start Backend (30 sec)
```powershell
python run.py
```
✅ Backend running at http://localhost:8000

### Step 3: Start Frontend (New Terminal - 1 min)
```powershell
cd rythmhacks\frontend
npm install  # Only needed first time
npm run dev
```
✅ Frontend running at http://localhost:5173

---

## Verify Everything Works

### 1. Check Backend Health
Open: http://localhost:8000/health

Should see:
```json
{"status": "healthy", "services": {"api": "operational", "ml_engine": "ready"}}
```

### 2. Check API Docs
Open: http://localhost:8000/docs

You'll see interactive API documentation with 10 endpoints.

### 3. Check Frontend
Open: http://localhost:5173

You should see the MediLytica landing page.

### 4. Test Integration
1. Go to: http://localhost:5173/dashboard
2. Click "Regression" in sidebar
3. Click "Regenerate" button
4. Should see scatter plot with red regression line and metrics

---

## What's Available

### ✅ Working Features
- **Linear Regression** - Train and visualize
- **Polynomial Regression** - Configurable degree
- **Ridge/Lasso Regression** - L1/L2 regularization
- **Decision Trees** - Classification & regression
- **CSV Upload** - Drag & drop datasets
- **Metrics Dashboard** - R², MAE, MSE, accuracy

### 📊 API Endpoints
- `POST /api/regression/train`
- `POST /api/decision-tree/train`
- `POST /datasets/upload`
- Plus 7 more (see /docs)

---

## Troubleshooting

**Backend won't start?**
```powershell
# Make sure virtual environment is activated
.\venv\Scripts\Activate.ps1

# Reinstall
pip install -r requirements.txt
```

**Frontend won't start?**
```powershell
# Clear and reinstall
rm -r node_modules
npm install
```

**CORS errors?**
- Ensure backend is on port 8000
- Ensure frontend is on port 5173

---

## Next Steps

1. ✅ Both servers running
2. 🔜 Test regression in dashboard
3. 🔜 Upload a CSV file
4. 🔜 Train decision tree
5. 🔜 Start building new features!

---

## Full Documentation

- **Complete Setup**: See `SETUP.md`
- **Backend Details**: See `backend/README.md`
- **Implementation Summary**: See `backend/IMPLEMENTATION_SUMMARY.md`
- **API Docs**: http://localhost:8000/docs
