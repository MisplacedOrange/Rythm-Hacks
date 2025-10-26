# 🎯 MediLytica Demo Guide

This guide will walk you through testing the complete ML model upload and visualization workflow using the pre-built demo model.

## 📋 Prerequisites

✅ Backend server running on http://localhost:8000  
✅ Frontend running on http://localhost:5173  
✅ Demo files in `rythmhacks/backend/demo/`

## 🚀 Step-by-Step Demo

### Step 1: Access the Dashboard

1. Open your browser to **http://localhost:5173**
2. Navigate to the **Dashboard** page
3. You should see the ML visualization interface

### Step 2: Upload the Demo Model

1. **Locate the upload section** in the dashboard
2. Click **"Upload Model"** or **drag and drop**
3. **Select file**: `rythmhacks/backend/demo/heart_disease_model.pkl`
4. **Wait for confirmation**: You should see:
   ```
   ✅ Model uploaded successfully!
   Model ID: [unique-id]
   Type: RandomForestClassifier
   ```

### Step 3: Upload Test Data

After the model uploads, you'll see a prompt to calculate metrics:

1. Click **"Choose CSV or JSON file"**
2. **Select**: `rythmhacks/backend/demo/heart_disease_test_data.csv`
3. **Select target column**: Choose `target` from dropdown
4. Click **"Calculate Metrics"**
5. **Wait for processing**: Should take 2-5 seconds

### Step 4: View Performance Charts

Once metrics are calculated, you'll see a **2×2 grid** with 4 visualizations:

#### 📊 Top Left: Metrics Table
- **F1 Score**: ~0.85-0.90
- **Precision**: ~0.85-0.90
- **Recall**: ~0.85-0.90
- **Accuracy**: ~0.85-0.90

#### 🔥 Top Right: Confusion Matrix
- Heatmap showing:
  - True Positives (correct predictions for class 1)
  - True Negatives (correct predictions for class 0)
  - False Positives (incorrectly predicted as 1)
  - False Negatives (incorrectly predicted as 0)

#### 📈 Bottom Left: ROC Curve
- **Class 0** curve (blue) with AUC ~0.90-0.95
- **Class 1** curve (orange) with AUC ~0.90-0.95
- Diagonal reference line (random classifier)

#### 🎯 Bottom Right: SHAP Feature Importance
- Top 13 features ranked by importance
- `feature_X` names (in real use, would be actual feature names)
- Color gradient showing positive/negative contribution

## 🎨 What You're Testing

This demo validates:

✅ **Model Upload**: File handling, validation, storage  
✅ **Metrics Calculation**: F1, precision, recall, accuracy from test data  
✅ **Confusion Matrix**: Binary classification visualization  
✅ **ROC Curves**: Multi-class AUC scores  
✅ **SHAP Analysis**: Feature importance extraction  
✅ **Chart Rendering**: Plotly.js integration  
✅ **Layout**: 2×2 grid with proper spacing (500px height)  

## ⚠️ Troubleshooting

### "No metrics available"
- **Cause**: Model uploaded but test data not provided
- **Fix**: Upload `heart_disease_test_data.csv` and select `target` column

### "Failed to calculate metrics"
- **Cause**: Wrong target column or data format mismatch
- **Fix**: Ensure you selected `target` (not feature_0, feature_1, etc.)

### Charts not displaying
- **Cause**: Browser console errors or missing Plotly
- **Fix**: Check browser console (F12), verify frontend dependencies installed

### Backend connection error
- **Cause**: Backend server not running
- **Fix**: Run `python run.py` in `rythmhacks/backend/`

## 🔄 Testing Different Scenarios

### Test Case 1: Re-upload Same Model
- Upload `heart_disease_model.pkl` again
- Should get a **new Model ID**
- Charts should update with same performance

### Test Case 2: Upload Without Test Data
- Upload model
- **Don't upload test data**
- Should see prompt: "Upload test data to calculate metrics"

### Test Case 3: Wrong Target Column
- Upload model
- Upload test data
- Select `feature_0` instead of `target`
- Should fail gracefully with error message

## 📸 Expected Screenshots

1. **Before upload**: Empty state with "Upload your model" prompt
2. **After model upload**: Model info card with model ID
3. **After metrics calculation**: Full 2×2 grid with all 4 charts
4. **Hover interactions**: Tooltips on charts showing values

## 🎓 Next Steps

After validating the demo:

1. **Try your own model**: Follow `PYTORCH_SETUP.md` for custom PyTorch models
2. **Test collaboration**: Open in 2 browsers (Phase 4 feature)
3. **Add more visualizations**: Extend `PerformanceCharts.jsx`

## 📁 Demo File Locations

```
rythmhacks/backend/demo/
├── heart_disease_model.pkl          # Pre-trained RandomForest
├── heart_disease_test_data.csv      # 200 test samples
└── README.md                         # Technical details
```

## 🐛 Known Issues

- **SHAP Warning**: "SHAP library not installed" (non-critical, install with `pip install shap`)
- **PyTorch Custom Models**: Requires class definition in `custom_models.py` (see `PYTORCH_SETUP.md`)

---

**🎉 Enjoy testing your ML visualization platform!**
