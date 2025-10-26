# Custom PyTorch Model Setup Guide

## Quick Fix for "Can't get attribute 'HeartDiseaseMLP'" Error

You're seeing this error because your PyTorch model uses a custom class that the backend doesn't know about. Here's how to fix it:

---

## Step-by-Step Fix

### 1️⃣ Install PyTorch (if not already installed)

```powershell
cd backend
pip install torch==2.5.1
```

Or install all requirements:
```powershell
pip install -r requirements.txt
```

### 2️⃣ Add Your Model Class Definition

You need to tell the backend what your `HeartDiseaseMLP` class looks like.

**Find your training code** - Look for the file where you defined your model. It should have something like:

```python
class HeartDiseaseMLP(nn.Module):
    def __init__(self, ...):
        super(HeartDiseaseMLP, self).__init__()
        # Your layers
    
    def forward(self, x):
        # Your forward pass
        return x
```

**Copy the class definition** and paste it into:
```
backend/app/core/custom_models.py
```

**Example**: If your training code looks like this:

```python
# Your training script
import torch.nn as nn

class HeartDiseaseMLP(nn.Module):
    def __init__(self):
        super(HeartDiseaseMLP, self).__init__()
        self.fc1 = nn.Linear(13, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
```

**Then update `custom_models.py`** to replace the default HeartDiseaseMLP:

```python
# backend/app/core/custom_models.py

class HeartDiseaseMLP(nn.Module):
    """YOUR actual model architecture - REPLACE THE DEFAULT"""
    def __init__(self):
        super(HeartDiseaseMLP, self).__init__()
        self.fc1 = nn.Linear(13, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Make sure it's in the registry
CUSTOM_MODELS = {
    'HeartDiseaseMLP': HeartDiseaseMLP,  # ✅ This must match your class name exactly!
    # Add more models here...
}
```

### 3️⃣ Restart the Backend Server

**Stop the current backend:**
- Press `Ctrl + C` in the terminal running the backend

**Restart it:**
```powershell
cd backend
python run.py
```

You should see:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### 4️⃣ Test Your Model

1. Go back to the frontend
2. Upload your `.pkl` model file
3. Upload your test data CSV
4. Metrics should calculate successfully! ✅

---

## Common Issues & Solutions

### ❌ "PyTorch not installed"

```powershell
pip install torch==2.5.1
```

### ❌ Still getting "Can't get attribute" error

**Cause**: Class name mismatch or __init__ parameters don't match

**Solution**: 
1. Check that the class name is EXACTLY the same (case-sensitive!)
2. Make sure __init__ parameters match your training code
3. If your model had parameters like `HeartDiseaseMLP(input_size=13)`, make sure the __init__ accepts those

### ❌ "Model loading failed"

**Check**:
1. Is PyTorch installed? Run `pip show torch`
2. Did you restart the backend after editing custom_models.py?
3. Does your model file use `torch.save(model, 'file.pkl')`? (full model, not just state_dict)

---

## Alternative: Save Model Differently (For Future)

Instead of saving the full model with:
```python
torch.save(model, 'model.pkl')  # ❌ Requires class definition
```

Save only the weights:
```python
torch.save(model.state_dict(), 'model_weights.pt')  # ✅ Better
```

But this still requires the class definition in the backend to load it.

---

## Verification Checklist

- [ ] PyTorch installed (`pip show torch`)
- [ ] Class definition added to `custom_models.py`
- [ ] Class name matches EXACTLY (case-sensitive)
- [ ] Class added to `CUSTOM_MODELS` dictionary
- [ ] Backend restarted
- [ ] No errors in backend console

---

## Still Need Help?

**Check the backend terminal output** for detailed error messages. The error will tell you:
- Which class is missing
- Where it's trying to load from
- What went wrong

**Example error breakdown:**
```
Can't get attribute 'HeartDiseaseMLP' on <module '__mp_main__'...>
                    ^^^^^^^^^^^^^^^^
                    This is the class name you need to add
```

Look for that exact class name in your training code, copy the entire class definition, and paste it into `custom_models.py`.
