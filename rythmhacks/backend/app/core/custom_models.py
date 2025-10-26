"""
Custom Model Architectures
Add your custom model class definitions here to support loading pickled models
"""

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available - custom PyTorch models won't load")


# Example custom model architectures
# Users should add their own model classes here

if TORCH_AVAILABLE:
    class HeartDiseaseMLP(nn.Module):
        """
        Example MLP for Heart Disease Classification
        Add your actual model architecture here
        """
        def __init__(self, input_size=13, hidden_sizes=[64, 32, 16], num_classes=2):
            super(HeartDiseaseMLP, self).__init__()
            
            # Build layers dynamically
            layers = []
            prev_size = input_size
            
            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.3))
                prev_size = hidden_size
            
            # Output layer
            layers.append(nn.Linear(prev_size, num_classes))
            
            self.model = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.model(x)


    class SimpleMLP(nn.Module):
        """
        Simple Multi-Layer Perceptron
        """
        def __init__(self, input_size, hidden_size, num_classes):
            super(SimpleMLP, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, num_classes)
        
        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            return out


    class DeepMLP(nn.Module):
        """
        Deeper Multi-Layer Perceptron with multiple hidden layers
        """
        def __init__(self, input_size, hidden_sizes=[128, 64, 32], num_classes=2, dropout=0.3):
            super(DeepMLP, self).__init__()
            
            layers = []
            prev_size = input_size
            
            for hidden_size in hidden_sizes:
                layers.extend([
                    nn.Linear(prev_size, hidden_size),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_size),
                    nn.Dropout(dropout)
                ])
                prev_size = hidden_size
            
            layers.append(nn.Linear(prev_size, num_classes))
            
            self.model = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.model(x)


    class CNN1D(nn.Module):
        """
        1D Convolutional Neural Network for sequence data
        """
        def __init__(self, input_channels=1, num_classes=2):
            super(CNN1D, self).__init__()
            self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool1d(2)
            self.fc1 = nn.Linear(64 * 3, 128)  # Adjust based on input size
            self.fc2 = nn.Linear(128, num_classes)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.5)
        
        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)  # Flatten
            x = self.dropout(self.relu(self.fc1(x)))
            x = self.fc2(x)
            return x


    class SimplePyTorchMLP(nn.Module):
        """
        Simple PyTorch MLP without BatchNorm - robust for inference
        Used by demo models to avoid dimension issues
        """
        def __init__(self, input_size=10, hidden_size=32, num_classes=2):
            super(SimplePyTorchMLP, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_size, 16),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(16, num_classes)
            )
        
        def forward(self, x):
            return self.model(x)


# Registry of available custom models
CUSTOM_MODELS = {
    'HeartDiseaseMLP': HeartDiseaseMLP if TORCH_AVAILABLE else None,
    'SimpleMLP': SimpleMLP if TORCH_AVAILABLE else None,
    'DeepMLP': DeepMLP if TORCH_AVAILABLE else None,
    'CNN1D': CNN1D if TORCH_AVAILABLE else None,
    'SimplePyTorchMLP': SimplePyTorchMLP if TORCH_AVAILABLE else None,  # Demo model
}


def get_custom_model(model_name):
    """
    Get a custom model class by name
    
    Args:
        model_name: Name of the custom model class
    
    Returns:
        Model class or None if not found
    """
    return CUSTOM_MODELS.get(model_name)


def register_custom_model(name, model_class):
    """
    Register a new custom model class
    
    Args:
        name: Name to register the model under
        model_class: The model class to register
    """
    CUSTOM_MODELS[name] = model_class
    print(f"Registered custom model: {name}")


# Instructions for users:
"""
To add your own custom PyTorch model:

1. Define your model class in this file (inheriting from nn.Module)
2. Add it to the CUSTOM_MODELS dictionary
3. Restart the backend server

Example:
    class MyCustomModel(nn.Module):
        def __init__(self, ...):
            super(MyCustomModel, self).__init__()
            # Define layers
        
        def forward(self, x):
            # Define forward pass
            return x
    
    # Register it
    CUSTOM_MODELS['MyCustomModel'] = MyCustomModel
"""
