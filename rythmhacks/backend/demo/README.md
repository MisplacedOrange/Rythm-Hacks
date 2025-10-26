# Demo Model and Test Data

This directory contains a pre-trained demo model that you can use to test the ML visualization features.

## Files

### `heart_disease_model.pkl`
- **Type**: Scikit-learn RandomForestClassifier
- **Task**: Binary classification (heart disease prediction)
- **Training samples**: 800
- **Features**: 13 numerical features
- **Model**: 100 decision trees
- **Format**: joblib/pickle (.pkl)

### `heart_disease_test_data.csv`
- **Samples**: 200 test samples (unseen during training)
- **Columns**: 
  - `feature_0` through `feature_12` (13 features)
  - `target` (0 or 1, the ground truth labels)

## How to Use

### Step 1: Upload the Model
1. In the dashboard, click **"Upload Model"** or drag and drop
2. Select `heart_disease_model.pkl` from this directory
3. Wait for upload confirmation

### Step 2: Upload Test Data for Metrics
1. After model upload, you'll see a prompt to upload test data
2. Click **"Choose CSV or JSON file"**
3. Select `heart_disease_test_data.csv`
4. **Important**: Select `target` as the target column
5. Click **"Calculate Metrics"**

### Step 3: View Performance Charts
The system will calculate and display:
- **Metrics Table**: F1 Score, Precision, Recall, Accuracy
- **Confusion Matrix**: True Positives, False Positives, etc.
- **ROC Curve**: With AUC score for each class
- **SHAP Feature Importance**: Top features contributing to predictions

## Expected Performance

Since this is a synthetic dataset with clear patterns:
- **Accuracy**: ~85-90%
- **F1 Score**: ~0.85-0.90
- **AUC**: ~0.90-0.95

The model performs well because it was trained on data with informative features.

## Troubleshooting

**Q: I don't see the metrics after uploading**
- Make sure you uploaded the test data CSV
- Check that you selected `target` as the target column

**Q: Can I use my own model?**
- Yes! For scikit-learn models, just save with `joblib.dump(model, 'mymodel.pkl')`
- For PyTorch models, see `PYTORCH_SETUP.md` in the backend directory

**Q: What if my test data has different features?**
- Make sure your test CSV has the same features as the model was trained on
- The target column should be named appropriately (e.g., 'target', 'label', 'y')
