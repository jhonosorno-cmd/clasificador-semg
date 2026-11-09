"""
Entrenamiento del Modelo A: Detector Binario (Reposo vs Gesto Activo)
Este modelo clasifica si una ventana es reposo (0) o gesto activo (1-6 → 1)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from src.config import *
from src.models import HybridCNNLSTM
from src.data_loader import create_data_loaders


def convert_to_binary_labels(labels):
    """Convierte etiquetas 0-6 a binarias: 0→0 (reposo), 1-6→1 (gesto)"""
    return (labels > 0).astype(np.int64)


def train_epoch(model, loader, criterion, optimizer, device, max_grad_norm=1.0):
    """Entrena una época del modelo binario."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Train", leave=False)
    for signals, labels in pbar:
        # Convertir a binario
        labels = (labels > 0).long()
        
        signals = signals.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{correct/total:.3f}"})
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


def validate_epoch(model, loader, criterion, device):
    """Valida el modelo binario."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for signals, labels in tqdm(loader, desc="Val", leave=False):
            # Convertir a binario
            labels = (labels > 0).long()
            
            signals = signals.to(device)
            labels = labels.to(device)
            
            outputs = model(signals)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy, balanced_acc, all_preds, all_labels


def main():
    print("=" * 60)
    print("ENTRENAMIENTO MODELO A: DETECTOR BINARIO (REPOSO VS GESTO)")
    print("=" * 60)
    
    device = torch.device(DEVICE)
    print(f"\nDispositivo: {device}")
    
    # Cargar datos
    print("\n[1/5] Cargando datos...")
    train_loader, val_loader, test_loader = create_data_loaders(
        batch_size=BATCH_SIZE
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Crear modelo - BINARIO (2 clases)
    print("\n[2/5] Creando modelo binario...")
    model = HybridCNNLSTM(num_classes=2).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros totales: {total_params:,}")
    
    # Calcular class weights para balancear binario
    print("\nCalculando class weights binarios...")
    all_labels = []
    for _, labels in train_loader:
        binary_labels = (labels > 0).numpy()
        all_labels.extend(binary_labels)
    
    counts = np.bincount(all_labels, minlength=2).astype(np.float64)
    total = counts.sum()
    weights = total / (2 * counts)  # inverse frequency
    class_weights = torch.FloatTensor(weights).to(device)
    
    print(f"Distribución binaria train:")
    print(f"  Reposo (0): {int(counts[0])} muestras, weight: {weights[0]:.3f}")
    print(f"  Gesto (1):  {int(counts[1])} muestras, weight: {weights[1]:.3f}")
    
    # Loss y optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Early stopping
    best_val_accuracy = 0.0
    patience_counter = 0
    patience_es = 15
    
    print("\n[3/5] Iniciando entrenamiento...")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Early stopping patience: {patience_es}")
    print("-" * 60)
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, val_bal_acc, _, _ = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Scheduler
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val BalAcc: {val_bal_acc*100:.2f}%")
        print(f"LR: {current_lr:.6f}")
        
        # Early stopping
        is_best = val_acc > best_val_accuracy
        if is_best:
            best_val_accuracy = val_acc
            patience_counter = 0
            
            # Guardar mejor modelo
            checkpoint_path = CHECKPOINTS_PATH / "best_model_binary.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'val_balanced_accuracy': val_bal_acc,
            }, checkpoint_path)
            print(f"[OK] Mejor modelo guardado (Val Acc: {val_acc*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience_es:
                print(f"\nEarly stopping activado (patience {patience_es})")
                break
    
    # Cargar mejor modelo y evaluar en test
    print("\n[4/5] Evaluando mejor modelo en test set...")
    checkpoint = torch.load(CHECKPOINTS_PATH / "best_model_binary.pth", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_bal_acc, test_preds, test_labels = validate_epoch(
        model, test_loader, criterion, device
    )
    
    print(f"\n{'='*60}")
    print(f"RESULTADOS FINALES - MODELO A (DETECTOR BINARIO)")
    print(f"{'='*60}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test Balanced Accuracy: {test_bal_acc*100:.2f}%")
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    print(f"\nConfusion Matrix:")
    print(f"                Pred Reposo  Pred Gesto")
    print(f"Real Reposo     {cm[0,0]:6d}      {cm[0,1]:6d}")
    print(f"Real Gesto      {cm[1,0]:6d}      {cm[1,1]:6d}")
    
    # Métricas por clase
    print(f"\nMétricas por clase:")
    for i in range(2):
        class_name = "Reposo" if i == 0 else "Gesto"
        class_acc = cm[i,i] / cm[i].sum() if cm[i].sum() > 0 else 0
        print(f"  {class_name}: {class_acc*100:.2f}% ({cm[i,i]}/{cm[i].sum()})")
    
    print(f"\n[5/5] Entrenamiento completado!")
    print(f"Mejor Val Accuracy: {best_val_accuracy*100:.2f}%")
    print(f"Modelo guardado en: {CHECKPOINTS_PATH / 'best_model_binary.pth'}")


if __name__ == "__main__":
    main()
