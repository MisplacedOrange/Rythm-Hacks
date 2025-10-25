import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for scripts
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_training_history(train_losses, train_accs, val_losses=None, val_accs=None, out_prefix='training'):
    """Plot training/validation loss and accuracy and save PNG files.

    Args:
        train_losses (list[float])
        train_accs (list[float])
        val_losses (list[float] | None)
        val_accs (list[float] | None)
        out_prefix (str): filename prefix for saved figures
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label='Train Loss')
    if val_losses is not None:
        plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_loss.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accs, label='Train Acc')
    if val_accs is not None:
        plt.plot(epochs, val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy per epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_acc.png")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, labels=None, out_file='confusion_matrix.png'):
    """Plot and save a confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def plot_feature_importances(importances, feature_names=None, top_n=20, out_file='feature_importances.png'):
    """Plot top_n feature importances from a model."""
    importances = np.asarray(importances)
    indices = np.argsort(importances)[::-1][:top_n]
    values = importances[indices]

    if feature_names is None:
        feature_names = [str(i) for i in range(len(importances))]

    names = [feature_names[i] for i in indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=values, y=names, palette='viridis')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def plot_pixel_importances(importances, shape=(28, 28), out_file='pixel_importances.png'):
    """Reshape flat importances into an image (e.g., 28x28) and save a heatmap."""
    importances = np.asarray(importances)
    expected = shape[0] * shape[1]
    if importances.size != expected:
        raise ValueError(f"Importances length {importances.size} does not match shape {shape}")

    img = importances.reshape(shape)
    plt.figure(figsize=(6, 6))
    sns.heatmap(img, cmap='viridis')
    plt.title('Pixel Importances') 
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
