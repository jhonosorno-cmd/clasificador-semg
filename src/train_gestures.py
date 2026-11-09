"""
Entrenamiento del Modelo B: Clasificador de 6 Gestos (solo clases 1-6, sin reposo)
Este modelo clasifica entre los 6 gestos activos, asumiendo que ya se filtró el reposo.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from src.config import *
from src.models import HybridCNNLSTM
from src.data_loader import NinaProDB3Dataset
from torch.utils.data import DataLoader


def filter_gestures_only(dataset):
    """Filtra solo muestras de gestos activos (clases 1-6), remapea a 0-5"""
    gesture_mask = dataset.labels > 0
    dataset.samples = dataset.samples[gesture_mask]
    dataset.labels = dataset.labels[gesture_mask] - 1  # Remapear 1-6 → 0-5
    return dataset


def train_epoch(model, loader, criterion, optimizer, device, max_grad_norm=1.0):
    """Entrena una época del clasificador de gestos."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Train", leave=False)
    for signals, labels in pbar:
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
    """Valida el clasificador de gestos."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for signals, labels in tqdm(loader, desc="Val", leave=False):
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
    print("ENTRENAMIENTO MODELO B: CLASIFICADOR DE 6 GESTOS")
    print("=" * 60)
    
    device = torch.device(DEVICE)
    print(f"\nDispositivo: {device}")
    
    # Cargar datos y filtrar solo gestos
    print("\n[1/5] Cargando datos (solo gestos activos)...")
    train_reps = [1, 2, 3, 4]
    val_reps = [5]
    test_reps = [6]
    
    train_dataset = NinaProDB3Dataset(1, train_reps, mode='train', augment=True)
    val_dataset = NinaProDB3Dataset(1, val_reps, mode='val')
    test_dataset = NinaProDB3Dataset(1, test_reps, mode='test')
    
    # Filtrar solo gestos (clases 1-6 → 0-5)
    print("\nFiltrando solo gestos activos...")
    train_dataset = filter_gestures_only(train_dataset)
    val_dataset = filter_gestures_only(val_dataset)
    test_dataset = filter_gestures_only(test_dataset)
    
    print(f"Train: {len(train_dataset)} muestras de gestos")
    print(f"Val: {len(val_dataset)} muestras de gestos")
    print(f"Test: {len(test_dataset)} muestras de gestos")
    
    # Verificar distribución
    train_counts = np.bincount(train_dataset.labels, minlength=6)
    print(f"\nDistribución train (gestos 0-5):")
    for i in range(6):
        print(f"  Gesto {i}: {train_counts[i]} muestras")
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Crear modelo - 6 CLASES (gestos 0-5)
    print("\n[2/5] Creando modelo clasificador de 6 gestos...")
    model = HybridCNNLSTM(num_classes=6).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros totales: {total_params:,}")
    
    # Calcular class weights (sqrt-inverse para balance moderado)
    print("\nCalculando class weights...")
    counts = np.bincount(train_dataset.labels, minlength=6).astype(np.float64)
    counts[counts == 0] = 1.0
    total = counts.sum()
    weights = np.sqrt(total / (6 * counts))  # sqrt-inverse
    class_weights = torch.FloatTensor(weights).to(device)
    
    print(f"Class weights (sqrt-inverse):")
    for i in range(6):
        print(f"  Gesto {i}: {int(counts[i])} muestras, weight: {weights[i]:.3f}")
    
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
            checkpoint_path = CHECKPOINTS_PATH / "best_model_gestures.pth"
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
    checkpoint = torch.load(CHECKPOINTS_PATH / "best_model_gestures.pth", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_bal_acc, test_preds, test_labels = validate_epoch(
        model, test_loader, criterion, device
    )
    
    print(f"\n{'='*60}")
    print(f"RESULTADOS FINALES - MODELO B (CLASIFICADOR 6 GESTOS)")
    print(f"{'='*60}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test Balanced Accuracy: {test_bal_acc*100:.2f}%")
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    print(f"\nConfusion Matrix (6 gestos):")
    print("     ", end="")
    for i in range(6):
        print(f"Pred{i:1d} ", end="")
    print()
    for i in range(6):
        print(f"Real{i}: ", end="")
        for j in range(6):
            print(f"{cm[i,j]:5d} ", end="")
        print()
    
    # Métricas por clase
    print(f"\nAccuracy por gesto:")
    for i in range(6):
        class_acc = cm[i,i] / cm[i].sum() if cm[i].sum() > 0 else 0
        print(f"  Gesto {i}: {class_acc*100:.2f}% ({cm[i,i]}/{cm[i].sum()})")
    
    print(f"\n[5/5] Entrenamiento completado!")
    print(f"Mejor Val Accuracy: {best_val_accuracy*100:.2f}%")
    print(f"Modelo guardado en: {CHECKPOINTS_PATH / 'best_model_gestures.pth'}")


if __name__ == "__main__":
    main()
