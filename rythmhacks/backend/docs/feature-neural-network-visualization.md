# Feature Documentation: Neural Network Layer Visualization

## Overview
Interactive visualization of neural network architecture showing input, hidden, and output layers with adjustable neurons per layer. Users can modify layer configuration and see the structural changes in real-time.

---

## Current Implementation

### Frontend Components

**Location**: `Dashboard.jsx`

**State Management**:
```javascript
const [layers, setLayers] = useState([7, 7, 7, 7, 7, 7])  // 6 hidden layers

const updateLayer = (index, delta) => {
  setLayers(prev => {
    const newLayers = [...prev]
    newLayers[index] = Math.max(1, Math.min(15, newLayers[index] + delta))
    return newLayers
  })
}
```

**Visual Elements**:
- Input layer: 5 nodes (X₁-X₅) + 2 transformation labels (sin(X₁), sin(X₂))
- Hidden layers: 6 configurable layers (1-15 neurons each)
- Output layer: 7 neurons (fixed)
- +/- controls for each hidden layer
- Node styling: Color-coded by layer type

---

## Backend Integration

### API Endpoints

**POST `/api/model/architecture/configure`**

```python
# backend/app/api/routes/models.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List

router = APIRouter()

class LayerConfig(BaseModel):
    neurons: int = Field(ge=1, le=1024)
    activation: str = 'ReLU'
    dropout: float = Field(default=0.0, ge=0.0, le=0.9)

class NetworkArchitecture(BaseModel):
    input_size: int = Field(ge=1)
    hidden_layers: List[LayerConfig]
    output_size: int = Field(ge=1)
    problem_type: str = 'classification'

@router.post("/api/model/architecture/configure")
async def configure_architecture(arch: NetworkArchitecture):
    """
    Configure neural network architecture.
    Returns the built model structure and parameter count.
    """
    try:
        # Build PyTorch model from configuration
        model_info = build_model_from_config(arch)
        
        return {
            "status": "success",
            "architecture_id": model_info['id'],
            "total_params": model_info['param_count'],
            "layer_details": model_info['layers'],
            "model_summary": model_info['summary']
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/api/model/architecture/{arch_id}")
async def get_architecture(arch_id: str):
    """Get architecture configuration and visualization data"""
    arch = load_architecture(arch_id)
    
    # Generate visualization data for frontend
    viz_data = generate_network_visualization(arch)
    
    return {
        "architecture": arch,
        "visualization": viz_data,
        "parameter_count": calculate_params(arch)
    }
```

---

## Model Builder

### PyTorch Dynamic Model Creation

```python
# backend/app/ml/models/dynamic_network.py
import torch
import torch.nn as nn
from typing import List

class DynamicNeuralNetwork(nn.Module):
    """
    Dynamically built neural network based on user configuration
    """
    def __init__(self, input_size: int, hidden_configs: List[dict], output_size: int):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.activations = []
        self.dropouts = nn.ModuleList()
        
        # Build layers dynamically
        prev_size = input_size
        
        for i, config in enumerate(hidden_configs):
            # Linear layer
            layer = nn.Linear(prev_size, config['neurons'])
            self.layers.append(layer)
            
            # Activation
            activation = self._get_activation(config.get('activation', 'ReLU'))
            self.activations.append(activation)
            
            # Dropout (if specified)
            if config.get('dropout', 0) > 0:
                dropout = nn.Dropout(config['dropout'])
                self.dropouts.append(dropout)
            else:
                self.dropouts.append(None)
            
            prev_size = config['neurons']
        
        # Output layer
        self.output_layer = nn.Linear(prev_size, output_size)
        
    def forward(self, x):
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
        # Forward through hidden layers
        for i, (layer, activation, dropout) in enumerate(
            zip(self.layers, self.activations, self.dropouts)
        ):
            x = layer(x)
            x = activation(x)
            if dropout is not None:
                x = dropout(x)
        
        # Output layer
        x = self.output_layer(x)
        return x
    
    def _get_activation(self, name: str):
        """Get activation function by name"""
        activations = {
            'ReLU': nn.ReLU(),
            'Sigmoid': nn.Sigmoid(),
            'Tanh': nn.Tanh(),
            'LeakyReLU': nn.LeakyReLU(),
            'GELU': nn.GELU()
        }
        return activations.get(name, nn.ReLU())
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_layer_info(self):
        """Get detailed information about each layer"""
        info = []
        for i, layer in enumerate(self.layers):
            info.append({
                'layer_index': i,
                'type': 'Linear',
                'input_features': layer.in_features,
                'output_features': layer.out_features,
                'parameters': layer.in_features * layer.out_features + layer.out_features,
                'activation': self.activations[i].__class__.__name__
            })
        return info

def build_model_from_config(arch: NetworkArchitecture):
    """Build model from configuration"""
    hidden_configs = [
        {
            'neurons': layer.neurons,
            'activation': layer.activation,
            'dropout': layer.dropout
        }
        for layer in arch.hidden_layers
    ]
    
    model = DynamicNeuralNetwork(
        input_size=arch.input_size,
        hidden_configs=hidden_configs,
        output_size=arch.output_size
    )
    
    return {
        'id': generate_unique_id(),
        'model': model,
        'param_count': model.count_parameters(),
        'layers': model.get_layer_info(),
        'summary': str(model)
    }
```

---

## Visualization Data Generation

### Backend Visualization Generator

```python
# backend/app/ml/visualizers/network_structure.py
from typing import List, Dict

def generate_network_visualization(arch: NetworkArchitecture) -> Dict:
    """
    Generate visualization data for frontend React Flow or custom renderer
    """
    nodes = []
    edges = []
    
    x_offset = 0
    spacing_x = 150
    
    # Input layer nodes
    for i in range(arch.input_size):
        nodes.append({
            'id': f'input_{i}',
            'type': 'input',
            'data': {
                'label': f'X_{i+1}',
                'neurons': 1
            },
            'position': {'x': x_offset, 'y': i * 40},
            'style': {
                'background': 'linear-gradient(135deg, #ffa500 50%, #87ceeb 50%)',
                'border': '2px solid #999'
            }
        })
    
    x_offset += spacing_x
    
    # Hidden layers
    for layer_idx, layer in enumerate(arch.hidden_layers):
        y_start = (arch.input_size - layer.neurons) * 20
        
        for neuron_idx in range(layer.neurons):
            node_id = f'hidden_{layer_idx}_{neuron_idx}'
            nodes.append({
                'id': node_id,
                'type': 'hidden',
                'data': {
                    'label': '',
                    'layer': layer_idx,
                    'activation': layer.activation
                },
                'position': {'x': x_offset, 'y': y_start + neuron_idx * 30},
                'style': {
                    'background': '#b0d4ff',
                    'border': '2px solid #5B9BD5'
                }
            })
            
            # Connect to previous layer
            if layer_idx == 0:
                # Connect to input layer
                for i in range(arch.input_size):
                    edges.append({
                        'id': f'edge_input_{i}_to_{node_id}',
                        'source': f'input_{i}',
                        'target': node_id,
                        'type': 'smoothstep',
                        'animated': False
                    })
            else:
                # Connect to previous hidden layer
                prev_layer = arch.hidden_layers[layer_idx - 1]
                for prev_neuron in range(prev_layer.neurons):
                    edges.append({
                        'id': f'edge_h{layer_idx-1}_{prev_neuron}_to_{node_id}',
                        'source': f'hidden_{layer_idx-1}_{prev_neuron}',
                        'target': node_id,
                        'type': 'smoothstep'
                    })
        
        x_offset += spacing_x
    
    # Output layer
    last_hidden = arch.hidden_layers[-1]
    y_start = (arch.input_size - arch.output_size) * 20
    
    for i in range(arch.output_size):
        node_id = f'output_{i}'
        nodes.append({
            'id': node_id,
            'type': 'output',
            'data': {
                'label': f'Out_{i+1}',
                'neurons': 1
            },
            'position': {'x': x_offset, 'y': y_start + i * 40},
            'style': {
                'background': 'white',
                'border': '2px solid #999'
            }
        })
        
        # Connect to last hidden layer
        for neuron_idx in range(last_hidden.neurons):
            edges.append({
                'id': f'edge_h{len(arch.hidden_layers)-1}_{neuron_idx}_to_{node_id}',
                'source': f'hidden_{len(arch.hidden_layers)-1}_{neuron_idx}',
                'target': node_id,
                'type': 'smoothstep'
            })
    
    return {
        'nodes': nodes,
        'edges': edges,
        'total_params': calculate_total_params(arch)
    }

def calculate_total_params(arch: NetworkArchitecture) -> int:
    """Calculate total number of parameters"""
    total = 0
    prev_size = arch.input_size
    
    for layer in arch.hidden_layers:
        # Weight params + bias params
        total += (prev_size * layer.neurons) + layer.neurons
        prev_size = layer.neurons
    
    # Output layer
    total += (prev_size * arch.output_size) + arch.output_size
    
    return total
```

---

## Frontend Enhanced Implementation

### Layer Control Component

```javascript
// frontend/src/components/NetworkLayerControl.jsx
import React from 'react'
import './NetworkLayerControl.css'

export default function NetworkLayerControl({ layers, onUpdateLayer, onRemoveLayer, onAddLayer }) {
  return (
    <div className="layer-controls-container">
      <h3>Network Architecture</h3>
      
      <div className="input-layer-info">
        <strong>Input Layer:</strong> 5 neurons
      </div>
      
      <div className="hidden-layers">
        <h4>Hidden Layers ({layers.length})</h4>
        {layers.map((neurons, idx) => (
          <div key={idx} className="layer-control">
            <span>Layer {idx + 1}:</span>
            <button onClick={() => onUpdateLayer(idx, -1)}>−</button>
            <span className="neuron-count">{neurons}</span>
            <button onClick={() => onUpdateLayer(idx, 1)}>+</button>
            <button 
              className="remove-btn" 
              onClick={() => onRemoveLayer(idx)}
              disabled={layers.length === 1}
            >
              ×
            </button>
          </div>
        ))}
        
        <button className="add-layer-btn" onClick={onAddLayer}>
          + Add Layer
        </button>
      </div>
      
      <div className="output-layer-info">
        <strong>Output Layer:</strong> 7 neurons
      </div>
      
      <div className="param-count">
        <strong>Total Parameters:</strong> {calculateParams(layers)}
      </div>
    </div>
  )
}

function calculateParams(layers) {
  let total = 0
  let prevSize = 5 // input size
  
  layers.forEach(neurons => {
    total += (prevSize * neurons) + neurons
    prevSize = neurons
  })
  
  // Output layer
  total += (prevSize * 7) + 7
  
  return total.toLocaleString()
}
```

---

## Data Synchronization

### Syncing with Backend

```javascript
// Dashboard.jsx - Enhanced with API integration
const [layers, setLayers] = useState([7, 7, 7, 7, 7, 7])
const [architectureId, setArchitectureId] = useState(null)

const updateLayer = (index, delta) => {
  setLayers(prev => {
    const newLayers = [...prev]
    newLayers[index] = Math.max(1, Math.min(15, newLayers[index] + delta))
    return newLayers
  })
}

const syncArchitectureWithBackend = async () => {
  const architecture = {
    input_size: 5,
    hidden_layers: layers.map(neurons => ({
      neurons,
      activation: activation,  // From state
      dropout: 0.0
    })),
    output_size: 7,
    problem_type: problemType.toLowerCase()
  }
  
  try {
    const response = await fetch('http://localhost:8000/api/model/architecture/configure', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(architecture)
    })
    
    const data = await response.json()
    setArchitectureId(data.architecture_id)
    
    // Update UI with parameter count
    console.log('Total parameters:', data.total_params)
    
  } catch (error) {
    console.error('Failed to sync architecture:', error)
  }
}

// Call this when user clicks "Apply" or "Start Training"
```

---

## Storage & Persistence

### Database Schema

```python
# backend/app/models/architecture.py
from sqlalchemy import Column, String, Integer, JSON, DateTime
from datetime import datetime

class NetworkArchitectureModel(Base):
    __tablename__ = "network_architectures"
    
    id = Column(String, primary_key=True)
    input_size = Column(Integer, nullable=False)
    hidden_layers = Column(JSON, nullable=False)  # List of layer configs
    output_size = Column(Integer, nullable=False)
    problem_type = Column(String)
    total_params = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    training_runs = relationship("TrainingRun", back_populates="architecture")
```

---

## Validation Rules

### Frontend Validation

```javascript
const validateArchitecture = (layers) => {
  const errors = []
  
  if (layers.length < 1) {
    errors.push('Must have at least 1 hidden layer')
  }
  
  if (layers.length > 20) {
    errors.push('Maximum 20 hidden layers')
  }
  
  if (layers.some(n => n < 1 || n > 1024)) {
    errors.push('Each layer must have 1-1024 neurons')
  }
  
  const totalParams = calculateParams(layers)
  if (totalParams > 10000000) {  // 10M
    errors.push('Model too large (>10M parameters)')
  }
  
  return errors
}
```

---

## Testing

```python
# tests/test_architecture.py
import pytest
from app.ml.models.dynamic_network import DynamicNeuralNetwork

def test_model_creation():
    model = DynamicNeuralNetwork(
        input_size=5,
        hidden_configs=[
            {'neurons': 10, 'activation': 'ReLU'},
            {'neurons': 8, 'activation': 'ReLU'}
        ],
        output_size=7
    )
    
    assert model.count_parameters() > 0
    assert len(model.layers) == 2

def test_forward_pass():
    model = DynamicNeuralNetwork(5, [{'neurons': 10}], 7)
    x = torch.randn(32, 5)  # batch of 32
    output = model(x)
    assert output.shape == (32, 7)
```

---

## Future Enhancements

1. **Layer Types**: Add Conv2D, LSTM, Attention layers
2. **Drag-and-Drop**: Reorder layers visually
3. **Templates**: Pre-built architectures (VGG, ResNet-like)
4. **Smart Suggestions**: Recommend architecture based on dataset size
5. **Comparison**: Compare multiple architectures side-by-side
## MVP Additions (Missing)

- Add simple connector lines between columns (canvas/SVG) to convey flow.
- Performance: cap nodes per column; collapse when >12.
- Export: snapshot to PNG.
