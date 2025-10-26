# Custom PyTorch Models

## Why This File Exists

When you pickle a PyTorch model using `torch.save(model, 'file.pkl')`, Python saves a reference to your model's class, not the actual class definition. When the backend tries to load your model, it needs access to the class definition.

## The Error You're Seeing

```
Can't get attribute 'HeartDiseaseMLP' on <module '__mp_main__'...>
```

This means your model uses a custom class (`HeartDiseaseMLP`) that doesn't exist in the backend.

---

## How to Fix It

### Step 1: Find Your Model Architecture

Locate the Python file where you defined your model. It looks something like this:

```python
import torch.nn as nn

class HeartDiseaseMLP(nn.Module):
    def __init__(self, input_size=13, hidden_sizes=[64, 32, 16], num_classes=2):
        super(HeartDiseaseMLP, self).__init__()
        # Your layers here
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        # ... more layers
    
    def forward(self, x):
        # Your forward pass
        return x
```

### Step 2: Copy Your Class to `custom_models.py`

1. Open `backend/app/core/custom_models.py`
2. Find the section with model definitions (after `HeartDiseaseMLP` example)
3. Paste your model class definition
4. Add it to the `CUSTOM_MODELS` dictionary:

```python
CUSTOM_MODELS = {
    'HeartDiseaseMLP': HeartDiseaseMLP,
    'YourModelName': YourModelName,  # Add your model here
    # ... other models
}
```

### Step 3: Restart the Backend

```bash
cd backend
python run.py
```

---

## Example: Complete Setup

**Your training code:**
```python
# train.py
import torch
import torch.nn as nn

class HeartDiseaseMLP(nn.Module):
    def __init__(self, input_size=13, hidden_sizes=[64, 32, 16], num_classes=2):
        super(HeartDiseaseMLP, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, num_classes))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Train model
model = HeartDiseaseMLP()
# ... training code ...

# Save model
torch.save(model, 'heart_disease_model.pkl')  # This creates the problem!
```

**Fix in `custom_models.py`:**
```python
# backend/app/core/custom_models.py

class HeartDiseaseMLP(nn.Module):
    def __init__(self, input_size=13, hidden_sizes=[64, 32, 16], num_classes=2):
        super(HeartDiseaseMLP, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, num_classes))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Register it
CUSTOM_MODELS = {
    'HeartDiseaseMLP': HeartDiseaseMLP,
    # Add more models as needed
}
```

---

## Better Alternative: Save State Dict Only

Instead of pickling the entire model, save only the weights:

**Recommended approach:**
```python
# Save (training side)
torch.save(model.state_dict(), 'model_weights.pt')

# Load (backend side)
model = HeartDiseaseMLP()  # Must have class definition
model.load_state_dict(torch.load('model_weights.pt'))
```

But this still requires the class definition in the backend, so you'll need to add it to `custom_models.py` anyway.

---

## Supported by Default

These model types work without modifications:
- ✅ Scikit-learn models (.pkl, .joblib)
- ✅ Keras/TensorFlow models (.h5)
- ✅ PyTorch state_dict only (.pt with just weights)
- ❌ PyTorch full models with custom classes (.pkl, .pt with architecture)

---

## Need Help?

If you're still having issues:

1. Check that your model class name in `custom_models.py` matches exactly what's in your training code
2. Make sure all imports (torch, nn, etc.) are included
3. Restart the backend after making changes
4. Check the backend console for error messages

The error usually provides the exact class name that's missing - just search for that in your training code!
