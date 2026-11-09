"""Script de entrenamiento para clasificación de gestos EMG"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import balanced_accuracy_score

from src.data_loader import create_data_loaders
from src.models import HybridCNNLSTM
from src.config import DEVICE, SEED, BATCH_SIZE, LEARNING_RATE, EPOCHS

MAX_GRAD_NORM = 1.0
PATIENCE_ES = 15  # early stopping patience

# Para reproducibilidad
torch.manual_seed(SEED)
np.random.seed(SEED)

def print_class_distribution(dataset):
    """Imprime distribución de clases del dataset."""
    labels = dataset.labels
    counts = np.bincount(labels, minlength=7)
    total = len(labels)
    print("Distribucion de clases:")
    for i, c in enumerate(counts):
        print(f"  Clase {i}: {c} muestras ({c/total*100:.1f}%)")

def calculate_class_weights(dataset) -> torch.Tensor:
    """Calcula pesos sqrt-inverse de clase (moderada corrección)."""
    labels = dataset.labels
    counts = np.bincount(labels, minlength=7).astype(np.float64)
    counts[counts == 0] = 1.0
    total = counts.sum()
    
    # sqrt-inverse: moderada corrección para minorías
    weights = np.sqrt(total / (len(counts) * counts))
    
    print("Class weights (sqrt-inverse):")
    for i in range(len(counts)):
        print(f"  Clase {i}: peso {weights[i]:.3f}")
    
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)

def calculate_accuracy(predictions, targets):
    """Calcula accuracy"""
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    return correct / total

def calculate_balanced_accuracy(predictions, targets):
    """Calcula balanced accuracy"""
    _, predicted = torch.max(predictions, 1)
    return balanced_accuracy_score(targets.cpu().numpy(), predicted.cpu().numpy())

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Entrena el modelo por una época con gradient clipping."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(train_loader)
    
    pbar = tqdm(train_loader, desc="Train", leave=False)
    
    for batch_idx, (signals, labels) in enumerate(pbar):
        signals, labels = signals.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{correct/total:.3f}'
        })
    
    return total_loss / num_batches, correct / total

def validate_epoch(model, val_loader, criterion, device):
    """Valida el modelo por una época. Retorna loss, accuracy, balanced_accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Val", leave=False)
        for signals, labels in pbar:
            signals, labels = signals.to(device), labels.to(device)
            outputs = model(signals)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{correct/total:.3f}'})
    
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    return total_loss / len(val_loader), correct / total, bal_acc

def save_checkpoint(model, optimizer, epoch, val_acc, is_best, checkpoint_path):
    """
    Guarda checkpoint del modelo.
    
    Args:
        is_best: Si es el mejor modelo hasta ahora
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
    }
    
    # Guardar checkpoint actual
    torch.save(checkpoint, checkpoint_path / 'last_model.pth')
    
    # Guardar mejor modelo
    if is_best:
        torch.save(checkpoint, checkpoint_path / 'best_model.pth')
        print(f"  ¡Nuevo mejor modelo guardado! Val Acc: {val_acc:.4f}")

def train_model():
    """Función principal de entrenamiento"""
    print("=== INICIANDO ENTRENAMIENTO ===")
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Grad clip: {MAX_GRAD_NORM}")
    print(f"Early stopping patience: {PATIENCE_ES}")
    print("=" * 50)
    
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Crear data loaders (shuffle + augmentation online ligera)
    print("Cargando datasets...")
    train_loader, val_loader, test_loader = create_data_loaders(batch_size=BATCH_SIZE)
    
    print_class_distribution(train_loader.dataset)
    class_weights = calculate_class_weights(train_loader.dataset)
    
    # Modelo
    print("Creando modelo...")
    model = HybridCNNLSTM().to(DEVICE)
    print(f"Modelo: {sum(p.numel() for p in model.parameters()):,} parametros")
    
    # CE con inverse-frequency class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
    )
    
    best_val_accuracy = 0.0
    epochs_no_improve = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print("\nIniciando entrenamiento...")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 40)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_bal_acc = validate_epoch(model, val_loader, criterion, DEVICE)
        
        scheduler.step(val_acc)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val BalAcc: {val_bal_acc*100:.2f}%")
        print(f"LR: {lr:.6f}")
        
        is_best = val_acc > best_val_accuracy
        if is_best:
            best_val_accuracy = val_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        save_checkpoint(model, optimizer, epoch+1, val_acc, is_best, checkpoint_dir)
        
        if epochs_no_improve >= PATIENCE_ES:
            print(f"\nEarly stopping: sin mejora en {PATIENCE_ES} epochs")
            break
    
    total_time = time.time() - start_time
    print(f"\nEntrenamiento completado en {total_time/60:.2f} minutos")
    print(f"Mejor Val Accuracy: {best_val_accuracy:.4f} ({best_val_accuracy*100:.2f}%)")
    
    # Evaluar en test set
    print("\nEvaluando mejor modelo en test set...")
    best_checkpoint = torch.load(checkpoint_dir / 'best_model.pth')
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_bal_acc = validate_epoch(model, test_loader, criterion, DEVICE)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc*100:.2f}% | Test Balanced Accuracy: {test_bal_acc*100:.2f}%")
    
    np.save(checkpoint_dir / 'training_history.npy', history)
    print(f"\nCheckpoints guardados en: {checkpoint_dir}")
    print("Entrenamiento finalizado!")

if __name__ == "__main__":
    train_model()