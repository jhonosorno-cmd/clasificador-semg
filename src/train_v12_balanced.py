"""
Entrenamiento v12 FINAL: Cyborg + Focal Loss Moderado
Balance entre overall accuracy y balanced accuracy
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from src.config import *
from src.models_cyborg import CyborgHybrid
from src.focal_loss import FocalLoss
from src.data_loader import create_data_loaders


def calculate_class_weights(dataset) -> torch.Tensor:
    """Calcula pesos SQRT-INVERSE de clase (moderado)."""
    labels = dataset.labels
    counts = np.bincount(labels, minlength=7).astype(np.float64)
    counts[counts == 0] = 1.0
    total = counts.sum()
    
    # Sqrt-inverse frequency (moderado - balance entre overall y balanced acc)
    weights = np.sqrt(total / (len(counts) * counts))
    
    print("Class weights (sqrt-inverse - moderado):")
    for i in range(len(counts)):
        print(f"  Clase {i}: {int(counts[i])} muestras, peso {weights[i]:.3f}")
    
    return torch.FloatTensor(weights)


def train_epoch(model, loader, criterion, optimizer, device, max_grad_norm=1.0):
    """Entrena una época."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Train", leave=False)
    for signals, features, labels in pbar:
        signals = signals.to(device)
        features = features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(signals, features)
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
    """Valida el modelo."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for signals, features, labels in tqdm(loader, desc="Val", leave=False):
            signals = signals.to(device)
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(signals, features)
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
    print("ENTRENAMIENTO v12 FINAL: CYBORG + FOCAL LOSS MODERADO")
    print("Focal Loss (gamma=1.5) + Sqrt-Inverse Weights")
    print("=" * 60)
    
    device = torch.device(DEVICE)
    print(f"\nDispositivo: {device}")
    
    # Cargar datos
    print("\n[1/5] Cargando datos...")
    train_loader, val_loader, test_loader = create_data_loaders(batch_size=BATCH_SIZE)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Crear modelo Cyborg
    print("\n[2/5] Creando modelo Cyborg...")
    model = CyborgHybrid().to(device)
    model.get_model_info()
    
    # Calcular class weights (SQRT-INVERSE - moderado)
    print("\n[3/5] Calculando class weights...")
    class_weights = calculate_class_weights(train_loader.dataset).to(device)
    
    # Focal Loss con gamma=1.5 (moderado)
    print("\nUsando Focal Loss (gamma=1.5 - moderado) + Sqrt-Inverse Weights")
    criterion = FocalLoss(alpha=class_weights, gamma=1.5, reduction='mean')
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Early stopping basado en OVERALL ACCURACY (métrica del paper)
    best_val_accuracy = 0.0
    patience_counter = 0
    patience_es = 15
    
    print("\n[4/5] Iniciando entrenamiento...")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Early stopping patience: {patience_es}")
    print(f"Early stopping metric: OVERALL ACCURACY")
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
        
        # Scheduler basado en overall accuracy
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val BalAcc: {val_bal_acc*100:.2f}%")
        print(f"LR: {current_lr:.6f}")
        
        # Early stopping basado en OVERALL ACCURACY
        is_best = val_acc > best_val_accuracy
        if is_best:
            best_val_accuracy = val_acc
            patience_counter = 0
            
            # Guardar mejor modelo
            checkpoint_path = CHECKPOINTS_PATH / "best_model_v12_balanced.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'val_balanced_accuracy': val_bal_acc,
            }, checkpoint_path)
            print(f"[OK] Mejor modelo guardado (Val Acc: {val_acc*100:.2f}%, BalAcc: {val_bal_acc*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience_es:
                print(f"\nEarly stopping activado (patience {patience_es})")
                break
    
    # Cargar mejor modelo y evaluar en test
    print("\n[5/5] Evaluando mejor modelo en test set...")
    checkpoint = torch.load(CHECKPOINTS_PATH / "best_model_v12_balanced.pth", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_bal_acc, test_preds, test_labels = validate_epoch(
        model, test_loader, criterion, device
    )
    
    print(f"\n{'='*60}")
    print(f"RESULTADOS FINALES - v12 BALANCED")
    print(f"{'='*60}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test Balanced Accuracy: {test_bal_acc*100:.2f}%")
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    print(f"\nConfusion Matrix:")
    print("       ", end="")
    for i in range(7):
        print(f"Pred{i} ", end="")
    print()
    for i in range(7):
        print(f"Real{i}: ", end="")
        for j in range(7):
            print(f"{cm[i,j]:5d} ", end="")
        print()
    
    # Métricas por clase
    print(f"\nAccuracy por clase:")
    for i in range(7):
        class_acc = cm[i,i] / cm[i].sum() if cm[i].sum() > 0 else 0
        print(f"  Clase {i}: {class_acc*100:.2f}% ({cm[i,i]}/{cm[i].sum()})")
    
    # Comparación con objetivo
    print(f"\n{'='*60}")
    print(f"RESUMEN FINAL DE EXPERIMENTOS")
    print(f"{'='*60}")
    print(f"v12 Balanced:      {test_acc*100:.2f}% (BalAcc: {test_bal_acc*100:.2f}%)")
    print(f"v11 Focal (agr):   51.99% (BalAcc: 50.47%)")
    print(f"v10 Cyborg:        71.32% (BalAcc: 41.92%)")
    print(f"v7 Best CNN:       69.95%")
    print(f"Objetivo:          79.72%")
    
    if test_acc > 0.7972:
        print(f"\n*** OBJETIVO ALCANZADO! (+{(test_acc-0.7972)*100:.2f}%) ***")
    else:
        print(f"\nGap al objetivo: {(0.7972-test_acc)*100:.2f}%")
    
    print(f"\nMejor Val Accuracy: {best_val_accuracy*100:.2f}%")
    print(f"Modelo guardado en: {CHECKPOINTS_PATH / 'best_model_v12_balanced.pth'}")


if __name__ == "__main__":
    main()
