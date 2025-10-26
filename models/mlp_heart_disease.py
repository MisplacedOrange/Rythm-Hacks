import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import sys
import joblib
from pathlib import Path
from torchview import draw_graph


# Function to ensure the output directory exists
def ensure_output_dir(dir_name='network_visualization'):
    """Create the output directory if it doesn't exist."""
    output_dir = Path(dir_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

# Define the MLP model for Heart Disease
class HeartDiseaseMLP(nn.Module):
    def __init__(self, input_size=13):
        """
        Neural Network Architecture:
        - Input Layer: 13 features (age, sex, cp, trestbps, etc.)
        - Hidden Layer 1: 32 neurons with ReLU activation, BatchNorm, and Dropout(0.5)
        - Hidden Layer 2: 16 neurons with ReLU activation, BatchNorm, and Dropout(0.5)
        - Hidden Layer 3: 8 neurons with ReLU activation and Dropout(0.5)
        - Output Layer: 2 neurons with LogSoftmax activation for binary classification
        
        The architecture gradually reduces dimensions (13 -> 32 -> 16 -> 8 -> 2)
        while using regularization techniques (BatchNorm, Dropout) to prevent overfitting.
        """
        super(HeartDiseaseMLP, self).__init__()
        # Layer 1: Input(13) -> Hidden(32)
        self.fc1 = nn.Linear(input_size, 32)
        # Layer 2: Hidden(32) -> Hidden(16)
        self.fc2 = nn.Linear(32, 16)
        # Layer 3: Hidden(16) -> Hidden(8)
        self.fc3 = nn.Linear(16, 8)
        # Layer 4: Hidden(8) -> Output(2)
        self.fc4 = nn.Linear(8, 2)  # Binary classification: 0 (no disease) or 1 (disease)

        # Regularization components
        self.dropout = nn.Dropout(0.5)  # 50% dropout rate
        self.batch_norm1 = nn.BatchNorm1d(32)  # Normalize outputs of first layer
        self.batch_norm2 = nn.BatchNorm1d(16)  # Normalize outputs of second layer
        
    def forward(self, x):
        """
        Forward pass of the network:
        1. First hidden layer:
           - Linear transformation (13 -> 32)
           - Batch normalization
           - ReLU activation
           - Dropout (50%)
           
        2. Second hidden layer:
           - Linear transformation (32 -> 16)
           - Batch normalization
           - ReLU activation
           - Dropout (50%)
           
        3. Third hidden layer:
           - Linear transformation (16 -> 8)
           - ReLU activation
           - Dropout (50%)
           
        4. Output layer:
           - Linear transformation (8 -> 2)
           - LogSoftmax activation
        """
        # Layer 1
        x = self.fc1(x)         # Linear: 13 -> 32
        x = self.batch_norm1(x) # Normalize
        x = F.relu(x)          # ReLU activation
        x = self.dropout(x)    # Dropout

        # Layer 2
        x = self.fc2(x)         # Linear: 32 -> 16
        x = self.batch_norm2(x) # Normalize
        x = F.relu(x)          # ReLU activation
        x = self.dropout(x)    # Dropout

        # Layer 3
        x = self.fc3(x)        # Linear: 16 -> 8
        x = F.relu(x)          # ReLU activation
        x = self.dropout(x)    # Dropout

        # Output Layer
        x = self.fc4(x)        # Linear: 8 -> 2
        return F.log_softmax(x, dim=1)  # LogSoftmax for binary classification

# Load and preprocess data
def load_heart_disease_data():
    print("Loading Heart Disease dataset from UCI...")
    
    try:
        from ucimlrepo import fetch_ucirepo
        
        # Fetch dataset
        heart_disease = fetch_ucirepo(id=45)
        
        # Get features and targets
        X = heart_disease.data.features
        y = heart_disease.data.targets
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Convert target to binary: 0 = no disease, 1+ = disease present
        y = (y > 0).astype(int).values.ravel()
        
        print(f"Dataset shape: {X.shape}")
        print(f"Features: {X.columns.tolist()}")
        print(f"Class distribution: {np.bincount(y)}")
        
        return X.values, y
        
    except Exception as e:
        print(f"Error loading from ucimlrepo: {e}")
        print("Please install: pip install ucimlrepo")
        return None, None

# Prepare data loaders
def prepare_data(X, y, test_size=0.2, batch_size=32):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create datasets and loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Also return the train/test label arrays for computing class weights or diagnostics
    return train_loader, test_loader, scaler, y_train, y_test

# Training function
def train(model, device, train_loader, optimizer, criterion, epoch, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    correct = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # Gradient clipping to stabilize training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        batch_size = data.size(0)
        total_loss += loss.item() * batch_size
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = 100.0 * correct / len(train_loader.dataset)

    print(f'Epoch: {epoch}, Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.2f}%')

    # Return metrics for visualization
    return avg_loss, accuracy

# Testing function
def test(model, device, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # criterion likely returns mean loss; multiply by batch size to get sum
            batch_loss = criterion(output, target)
            total_loss += batch_loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss = total_loss / len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%\n')

    # Return metrics for visualization
    return test_loss, accuracy

# Main execution
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load data
    X, y = load_heart_disease_data()
    
    if X is None or y is None:
        return
    
    # Prepare data loaders (also get labels back for class weight calculation)
    train_loader, test_loader, scaler, y_train, y_test = prepare_data(X, y, batch_size=32)
    
    # Initialize model
    input_size = X.shape[1]
    model = HeartDiseaseMLP(input_size=input_size).to(device)
    
    # Optimizer (increase weight_decay for stronger L2 regularization)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    
    # Learning rate scheduler
    # ReduceLROnPlateau may not accept the `verbose` kwarg in some torch builds.
    # Use a compatible constructor and handle logging manually if desired.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )
    
    # Training loop with class-weighted loss, gradient clipping and early stopping
    print("\nStarting training...\n")
    epochs = 1000
    best_accuracy = 0.0

    # Compute class weights to address slight imbalance
    try:
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        # Map to tensor in class order (assumes classes are 0..C-1)
        weight_tensor = torch.tensor([cw[int(c)] for c in range(len(cw))], dtype=torch.float32).to(device)
        criterion = nn.NLLLoss(weight=weight_tensor)
        print(f"Using class weights: {cw}")
    except Exception:
        criterion = nn.NLLLoss()

    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    patience_es = 100
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train(model, device, train_loader, optimizer, criterion, epoch, grad_clip=1.0)
        te_loss, te_acc = test(model, device, test_loader, criterion)

        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        test_losses.append(te_loss)
        test_accs.append(te_acc)

        # Update learning rate based on validation metric (test accuracy)
        scheduler.step(te_acc)

        # Save best model and manage early stopping
        if te_acc > best_accuracy + 1e-6:
            best_accuracy = te_acc
            epochs_no_improve = 0
            
            # Save using both torch.save and joblib
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': te_acc,
                'scaler': scaler
            }, "heart_disease_mlp_best.pth")
            
            # Save using joblib for easier deployment
            model_package = {
                'model': model,
                'model_state_dict': model.state_dict(),
                'scaler': scaler,
                'input_size': input_size,
                'accuracy': te_acc,
                'epoch': epoch,
                'device': str(device)
            }
            joblib.dump(model_package, 'heart_disease_mlp_best.pkl')
            
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience_es:
            print(f"Early stopping: no improvement for {patience_es} epochs. Stopping training.")
            break

    print(f"\nBest Test Accuracy: {best_accuracy:.2f}%")
    print("Best model saved as:")
    print("  - heart_disease_mlp_best.pth (PyTorch format)")
    print("  - heart_disease_mlp_best.pkl (Joblib format)")

    # Create neural network visualization
    try:
        print("\nStarting visualization process...")
        
        # Ensure Graphviz is in PATH
        import os
        graphviz_path = r"C:\Program Files\Graphviz\bin"
        if graphviz_path not in os.environ['PATH']:
            os.environ['PATH'] += os.pathsep + graphviz_path
        
        print(f"Using Graphviz from: {graphviz_path}")
        
        # Create visualization directory
        output_dir = ensure_output_dir()
        output_dir = Path(output_dir).resolve()
        print(f"Output directory: {output_dir}")
        
        # Set up model visualization
        batch_size = 1
        input_shape = (batch_size, input_size)
        
        print("Drawing network graph...")
        model_graph = draw_graph(
            model, 
            input_size=input_shape,
            expand_nested=False,
            hide_module_functions=True,
            hide_inner_tensors=True,
            depth=3,  # Increased depth for more detail
            graph_dir=str(output_dir)
        )
        
        # Configure output path
        output_path = output_dir / "heart_disease_mlp_architecture"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print("\nChecking Graphviz setup...")
        import subprocess
        try:
            dot_path = os.path.join(graphviz_path, "dot.exe")
            result = subprocess.run([dot_path, '-V'], 
                                 capture_output=True, 
                                 text=True, 
                                 check=True)
            print(f"Graphviz version: {result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Error running dot: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            raise
        
        print("\nRendering visualization...")
        
        # Apply custom styling to make it horizontal and prettier
        graph = model_graph.visual_graph
        
        # Set graph to horizontal layout (Left to Right)
        graph.graph_attr.update({
            'rankdir': 'LR',  # Left to Right layout
            'splines': 'ortho',  # Orthogonal edges for cleaner look
            'nodesep': '0.8',  # Horizontal spacing between nodes
            'ranksep': '1.5',  # Vertical spacing between ranks
            'bgcolor': '#f8f9fa',  # Light background
            'dpi': '300',  # High resolution
            'pad': '0.5',  # Padding around graph
        })
        
        # Style nodes for better appearance
        graph.node_attr.update({
            'shape': 'box',
            'style': 'rounded,filled',
            'fillcolor': '#e3f2fd',  # Light blue fill
            'color': '#1976d2',  # Blue border
            'fontname': 'Arial',
            'fontsize': '11',
            'penwidth': '2',
            'margin': '0.3,0.2',
        })
        
        # Style edges
        graph.edge_attr.update({
            'color': '#424242',  # Dark gray
            'penwidth': '1.5',
            'arrowsize': '0.8',
        })
        
        graph.engine = 'dot'
        graph.render(
            str(output_path),
            format='png',
            cleanup=True,
        )
        print(f'Neural network architecture visualization saved as: {output_path}.png')
        print('Visualization is now horizontal (left-to-right) with improved styling!')
        
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        print("Please ensure Graphviz is installed and in your system PATH")
    
    # Demonstrate how to load the model from joblib
    print("\n" + "="*60)
    print("LOADING MODEL EXAMPLE")
    print("="*60)
    print("\nTo load the model later, use:")
    print("```python")
    print("import joblib")
    print("import torch")
    print("")
    print("# Load the model package")
    print("model_package = joblib.load('heart_disease_mlp_best.pkl')")
    print("")
    print("# Extract components")
    print("model = model_package['model']")
    print("scaler = model_package['scaler']")
    print("input_size = model_package['input_size']")
    print("")
    print("# Set to evaluation mode")
    print("model.eval()")
    print("")
    print("# Make predictions")
    print("# X_new = ... # your new data")
    print("# X_scaled = scaler.transform(X_new)")
    print("# X_tensor = torch.FloatTensor(X_scaled)")
    print("# with torch.no_grad():")
    print("#     predictions = model(X_tensor)")
    print("#     predicted_classes = predictions.argmax(dim=1)")
    print("```")

if __name__ == '__main__':
    main()