# Demo Models and Test Data

This directory contains pre-built demo models that are **guaranteed to work** with the dashboard.

## 🎯 Available Demo Models

### 1. Health Prediction Model (⭐ BEST - Shows All Features!)
- **Model**: `health_model.pkl` - RandomForest health predictor
- **Test Data**: `health_test_data.csv` - 300 samples, 12 meaningful features
- **Accuracy**: ~94%
- **Feature Importance**: ✅ YES! Bar chart with meaningful names
- **Features**: age, blood_pressure, cholesterol, heart_rate, bmi, glucose_level, exercise_hours, sleep_hours, stress_level, alcohol_consumption, smoking_status, family_history
- **Best for**: **Complete dashboard experience with feature importance visualization**

### 2. Simple sklearn Model (✅ Recommended for First Test)
- **Model**: `simple_demo_model.pkl` - RandomForest classifier
- **Test Data**: `simple_demo_test.csv` - 300 samples, 10 features
- **Accuracy**: ~85%
- **Feature Importance**: ⚠️ Generic names (feature_0, feature_1...)
- **Best for**: Quick testing, sklearn compatibility

### 3. PyTorch Model (✅ No BatchNorm Issues)
- **Model**: `pytorch_demo_model.pkl` - MLP classifier  
- **Test Data**: `pytorch_demo_test.csv` - 300 samples, 10 features
- **Accuracy**: ~77%
- **Feature Importance**: ❌ Neural networks don't have built-in feature importance
- **Best for**: Testing PyTorch integration, deep learning workflows

### 4. Heart Disease Model (Legacy)
- **Model**: `heart_disease_model.pkl` - RandomForest for heart disease
- **Test Data**: `heart_disease_test_data.csv` - 200 samples, 13 features
- **Best for**: Domain-specific example

## 🚀 Quick Start - Use the New Demo Models!

### Option 1: Use Pre-Built Models (Ready to Go!)

1. **In the Dashboard**:
   - Click "Upload Model" 
   - Select `health_model.pkl` from `rythmhacks/backend/demo/` ⭐ **RECOMMENDED**
   - Choose framework: `sklearn`
   - Click "Upload Test Data"
   - Select `health_test_data.csv` from the same directory
   - Click "Calculate Metrics"
   - 📊 See your performance charts **WITH FEATURE IMPORTANCE**!

**Alternative:** Use `simple_demo_model.pkl` for basic testing

### Option 2: Create Fresh Models

Run either script to generate new models with guaranteed compatibility:

**For sklearn RandomForest:**
```bash
cd rythmhacks/backend/demo
python create_simple_model.py
# Creates: simple_demo_model.pkl + simple_demo_test.csv
```

**For Health Model with Feature Importance:** ⭐ **BEST OPTION**
```bash
cd rythmhacks/backend/demo
python create_feature_rich_model.py
# Creates: health_model.pkl + health_test_data.csv
# Features: age, blood_pressure, cholesterol, heart_rate, etc.
```

**For PyTorch MLP:**
```bash
cd rythmhacks/backend/demo
python create_pytorch_model.py
# Creates: pytorch_demo_model.pkl + pytorch_demo_test.csv
```

## 💡 Why These New Models Work

The new demo models (`simple_demo_model.pkl` and `pytorch_demo_model.pkl`) are designed to avoid common pitfalls:

✅ **Matching Dimensions** - Model input size matches test data features exactly  
✅ **No BatchNorm Issues** - PyTorch model uses Dropout instead of BatchNorm  
✅ **Compatible Preprocessing** - Same normalization for train and test  
✅ **Proper Model Structure** - PyTorch models have `.model` attribute  
✅ **Correct File Formats** - Models saved with joblib, CSV has features + target

## 🔧 Common Error Solutions

**Error: "running_mean should contain X elements not Y"**
- ❌ Problem: Test data has different number of features than training data
- ✅ Solution: Use `simple_demo_model.pkl` + `simple_demo_test.csv` which are guaranteed to match

**Error: "'ModelName' object has no attribute 'model'"**
- ❌ Problem: Model class definition changed since training
- ✅ Solution: Use `pytorch_demo_model.pkl` which matches current code

**Error: "Can't get attribute 'ModelName'"**
- ❌ Problem: Custom model class not registered properly
- ✅ Solution: Use standard models from demo directory

## 📁 File Structure

```
demo/
├── create_simple_model.py          # Generate sklearn model
├── create_pytorch_model.py         # Generate PyTorch model
├── create_feature_rich_model.py    # ⭐ Generate health model with feature importance
├── health_model.pkl                # ⭐ NEW: Health predictor with meaningful features
├── health_test_data.csv            # ⭐ NEW: Test data (12 health features)
├── simple_demo_model.pkl           # ✅ sklearn RandomForest
├── simple_demo_test.csv            # ✅ Test data (10 features)
├── pytorch_demo_model.pkl          # ✅ PyTorch MLP
├── pytorch_demo_test.csv           # ✅ Test data (10 features)
├── heart_disease_model.pkl         # Legacy demo
├── heart_disease_test_data.csv     # Legacy test data
└── README.md                       # This file
```

## Files

### NEW: `health_model.pkl` ⭐ BEST CHOICE!
- **Type**: Scikit-learn RandomForestClassifier
- **Task**: Health risk prediction (binary classification)
- **Features**: 12 meaningful health-related features:
  - age, blood_pressure, cholesterol, heart_rate
  - bmi, glucose_level, exercise_hours, sleep_hours
  - stress_level, alcohol_consumption, smoking_status, family_history
- **Test Accuracy**: ~94%
- **Feature Importance**: ✅ YES! Shows bar chart with meaningful names
- **Format**: joblib/pickle (.pkl)
- **Status**: ✅ **Guaranteed to show feature importance chart!**

### NEW: `health_test_data.csv` ⭐
- **Samples**: 300 test samples
- **Columns**: 12 health features + `target`
- **Column Names**: age, blood_pressure, cholesterol, heart_rate, bmi, glucose_level, exercise_hours, sleep_hours, stress_level, alcohol_consumption, smoking_status, family_history, target
- **Format**: CSV with descriptive header row
- **Status**: ✅ Matches health_model.pkl perfectly!
- **Feature Importance**: ✅ Will display in dashboard with these names!

### NEW: `simple_demo_model.pkl` ⭐
- **Type**: Scikit-learn RandomForestClassifier
- **Task**: Binary classification
- **Features**: 10 numerical features
- **Test Accuracy**: ~85%
- **Format**: joblib/pickle (.pkl)
- **Status**: ✅ Guaranteed to work!

### NEW: `simple_demo_test.csv` ⭐
- **Samples**: 300 test samples
- **Columns**: `feature_0` through `feature_9` (10 features) + `target`
- **Format**: CSV with header row
- **Status**: ✅ Matches simple_demo_model.pkl perfectly!

### NEW: `pytorch_demo_model.pkl` ⭐
- **Type**: PyTorch MLP (SimplePyTorchMLP)
- **Architecture**: 10 → 32 → 16 → 2 (no BatchNorm!)
- **Test Accuracy**: ~77%
- **Format**: joblib/pickle (.pkl)
- **Status**: ✅ No BatchNorm dimension issues!

### NEW: `pytorch_demo_test.csv` ⭐
- **Samples**: 300 test samples  
- **Columns**: `feature_0` through `feature_9` (10 features) + `target`
- **Status**: ✅ Matches pytorch_demo_model.pkl perfectly!

### `heart_disease_model.pkl` (Legacy)
- **Type**: Scikit-learn RandomForestClassifier
- **Features**: 13 numerical features
- **Training samples**: 800
- **Format**: joblib/pickle (.pkl)

### `heart_disease_test_data.csv` (Legacy)
- **Samples**: 200 test samples
- **Columns**: `feature_0` through `feature_12` (13 features) + `target`

## How to Use

### Step 1: Upload the Model
1. In the dashboard, click **"Upload Model"**
2. Select `simple_demo_model.pkl` (recommended) or `pytorch_demo_model.pkl`
3. Choose framework: `sklearn` (works for both!)
4. Wait for upload confirmation

### Step 2: Upload Test Data for Metrics
1. After model upload, click **"Upload Test Data"**
2. Select the matching CSV:
   - `simple_demo_test.csv` for sklearn model
   - `pytorch_demo_test.csv` for PyTorch model
3. Click **"Calculate Metrics"**

### Step 3: View Performance Charts
The system will display:
- **Metrics Table**: Accuracy, Precision, Recall, F1 Score
- **Confusion Matrix**: Classification results
- **ROC Curve**: With AUC scores
- **Feature Importance**: Top contributing features

## Expected Performance

**simple_demo_model.pkl:**
- Accuracy: ~85%
- F1 Score: ~0.85
- AUC: ~0.90

**pytorch_demo_model.pkl:**
- Accuracy: ~77%
- F1 Score: ~0.77
- AUC: ~0.82

## 🎓 Creating Your Own Compatible Models

Use the creation scripts as templates:

1. **Ensure consistent feature counts** between training and testing
2. **For PyTorch**: Avoid BatchNorm or handle carefully
3. **Save properly**:
   ```python
   import joblib
   model.eval()  # For PyTorch
   joblib.dump(model, 'model.pkl')
   ```
4. **Create matching CSV** with same feature columns + target

See `create_simple_model.py` and `create_pytorch_model.py` for complete examples!
