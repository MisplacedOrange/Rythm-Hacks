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
from pathlib import Path


def _import_visualize():
    """Import the top-level visualize module even when running from inside models/.

    This inserts the repository root into sys.path as a fallback so `import visualize`
    works whether the current working directory is the repo root or the models/ folder.
    """
    try:
        import visualize
        return visualize
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parent.parent
        repo_root_str = str(repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)
        # Try import again (let any other exceptions propagate)
        import visualize
        return visualize

# Define the MLP model for Heart Disease
class HeartDiseaseMLP(nn.Module):
    def __init__(self, input_size=13):
        super(HeartDiseaseMLP, self).__init__()
        # Use a smaller network to reduce overfitting on tiny UCI dataset
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 2)  # Binary classification: 0 (no disease) or 1 (disease)

        # increase dropout and add batch norm sizes matching new layers
        self.dropout = nn.Dropout(0.5)
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.batch_norm2 = nn.BatchNorm1d(16)
        
    def forward(self, x):
        x = F.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)

        x = F.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)

        x = F.relu(self.fc3(x))
        x = self.dropout(x)

        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

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
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': te_acc,
                'scaler': scaler
            }, "heart_disease_mlp_best.pth")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience_es:
            print(f"Early stopping: no improvement for {patience_es} epochs. Stopping training.")
            break

    print(f"\nBest Test Accuracy: {best_accuracy:.2f}%")
    print("Best model saved as heart_disease_mlp_best.pth")

    # Create visualizations (saved as PNG files in the repo root)
    try:
        viz = _import_visualize()
        viz.plot_training_history(train_losses, train_accs, val_losses=test_losses, val_accs=test_accs, out_prefix='heart_disease_training')
        print('Saved training plots: heart_disease_training_loss.png, heart_disease_training_acc.png')
    except Exception as e:
        print(f'Could not create training plots: {e}')

if __name__ == '__main__':
    main()