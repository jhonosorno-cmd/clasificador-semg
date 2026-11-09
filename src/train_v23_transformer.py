"""
Entrenamiento v23: CyborgTransformer (CNN + Transformer Encoder + Features Manuales).

Hipótesis: el Transformer Encoder captura dependencias no locales entre los
15 frames temporales resultantes de la CNN, mejorando sobre el BiLSTM de v10.

Misma configuración de entrenamiento que v10 (train_cyborg.py):
- sqrt-inverse class weights
- AdamW lr=5e-4, wd=1e-4
- ReduceLROnPlateau(factor=0.5, patience=5)
- Early stopping patience=15
- Gradient clipping max_norm=1.0
- Mixup augmentation alpha=0.4
- Signal augmentation (noise, scaling, channel dropout) de v15
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from src.config import DEVICE, BATCH_SIZE, LEARNING_RATE, EPOCHS, CHECKPOINTS_PATH, SEED
from src.models_v23 import CyborgTransformer
from src.data_loader import create_data_loaders

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def augment_signal(signals: torch.Tensor, training: bool = True) -> torch.Tensor:
    """
    Augmentación de señal EMG (de v15).

    Aplica: ruido gaussiano (σ=0.05), scaling aleatorio (0.9–1.1),
    channel dropout (p=10% por canal).

    Args:
        signals: (B, 1, 125, 13)
        training: sólo aplica en modo entrenamiento
    Returns:
        signals aumentados, misma forma
    """
    if not training:
        return signals

    B = signals.size(0)

    # Ruido gaussiano
    noise = torch.randn_like(signals) * 0.05
    signals = signals + noise

    # Scaling aleatorio por muestra
    scale = torch.FloatTensor(B, 1, 1, 1).uniform_(0.9, 1.1).to(signals.device)
    signals = signals * scale

    # Channel dropout (10% probabilidad por canal)
    if np.random.rand() < 0.5:
        ch_mask = (torch.rand(B, 1, 1, 13) > 0.10).float().to(signals.device)
        signals = signals * ch_mask

    return signals


def mixup_data(
    x_signal: torch.Tensor,
    x_features: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.4,
    device: str = "cpu",
) -> tuple:
    """
    Mixup Augmentation sobre señal y features.

    Args:
        x_signal:   (B, 1, 125, 13)
        x_features: (B, F)
        y:          (B,)
        alpha:      parámetro Beta
        device:     dispositivo

    Returns:
        mixed_signal, mixed_features, y_a, y_b, lam
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x_signal.size(0)).to(device)
    mixed_signal   = lam * x_signal   + (1 - lam) * x_signal[idx]
    mixed_features = lam * x_features + (1 - lam) * x_features[idx]
    return mixed_signal, mixed_features, y, y[idx], lam


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def calculate_class_weights(dataset) -> torch.Tensor:
    """Calcula pesos sqrt-inverse de clase para CrossEntropyLoss."""
    labels = dataset.labels
    counts = np.bincount(labels, minlength=7).astype(np.float64)
    counts[counts == 0] = 1.0
    total = counts.sum()
    weights = np.sqrt(total / (len(counts) * counts))
    print("Class weights (sqrt-inverse):")
    for i in range(len(counts)):
        print(f"  Clase {i}: {int(counts[i])} muestras, peso {weights[i]:.3f}")
    return torch.FloatTensor(weights)


# ---------------------------------------------------------------------------
# Train / Validate
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> tuple[float, float]:
    """Entrena una época con signal augmentation y Mixup."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for signals, features, labels in pbar:
        signals  = signals.to(device)
        features = features.to(device)
        labels   = labels.to(device)

        # Signal augmentation
        signals = augment_signal(signals, training=True)

        # Mixup
        signals, features, labels_a, labels_b, lam = mixup_data(
            signals, features, labels, alpha=0.4, device=str(device)
        )

        optimizer.zero_grad()
        outputs = model(signals, features)
        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        target = labels_a if lam >= 0.5 else labels_b
        correct += (predicted == target).sum().item()
        total   += labels_a.size(0)

        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{correct/total:.3f}"})

    return total_loss / len(loader), correct / total


def validate_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, list, list]:
    """Valida el modelo (sin augmentation)."""
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for signals, features, labels in tqdm(loader, desc="Val", leave=False):
            signals  = signals.to(device)
            features = features.to(device)
            labels   = labels.to(device)

            outputs = model(signals, features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss   = total_loss / len(loader)
    accuracy   = np.mean(np.array(all_preds) == np.array(all_labels))
    bal_acc    = balanced_accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy, bal_acc, all_preds, all_labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("ENTRENAMIENTO v23: CyborgTransformer")
    print("CNN + Transformer Encoder + Features Manuales")
    print("=" * 60)

    device = torch.device(DEVICE)
    print(f"\nDispositivo: {device}")

    # --- Datos ---
    print("\n[1/5] Cargando datos S1...")
    train_loader, val_loader, test_loader = create_data_loaders(batch_size=BATCH_SIZE)
    print(f"  Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    # --- Modelo ---
    print("\n[2/5] Creando CyborgTransformer...")
    model = CyborgTransformer(manual_features_dim=48, num_classes=7).to(device)
    model.get_model_info()

    # --- Class weights ---
    print("\n[3/5] Calculando class weights...")
    class_weights = calculate_class_weights(train_loader.dataset).to(device)

    # --- Optimización ---
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # --- Entrenamiento ---
    print("\n[4/5] Iniciando entrenamiento...")
    print(f"  Epochs:   {EPOCHS}")
    print(f"  LR:       {LEARNING_RATE}")
    print(f"  Patience: 15")
    print("-" * 60)

    best_val_acc = 0.0
    patience_counter = 0
    patience_es = 15
    checkpoint_path = CHECKPOINTS_PATH / "best_model_v23_transformer.pth"

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 40)

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_bal_acc, _, _ = validate_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | BalAcc: {val_bal_acc*100:.2f}%")
        print(f"LR: {current_lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
                "val_balanced_accuracy": val_bal_acc,
            }, checkpoint_path)
            print(f"[OK] Mejor modelo guardado (Val Acc: {val_acc*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience_es:
                print(f"\nEarly stopping activado (patience {patience_es})")
                break

    # --- Evaluación final ---
    print("\n[5/5] Evaluando en test set...")
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    _, test_acc, test_bal_acc, test_preds, test_labels_list = validate_epoch(
        model, test_loader, criterion, device
    )

    print(f"\n{'='*60}")
    print("RESULTADOS FINALES — CyborgTransformer v23")
    print(f"{'='*60}")
    print(f"Test Accuracy:          {test_acc*100:.2f}%")
    print(f"Test Balanced Accuracy: {test_bal_acc*100:.2f}%")

    cm = confusion_matrix(test_labels_list, test_preds)
    print("\nAccuracy por clase:")
    for i in range(7):
        acc_i = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0.0
        print(f"  Clase {i}: {acc_i*100:.2f}% ({cm[i,i]}/{cm[i].sum()})")

    print(f"\n{'='*60}")
    print("COMPARACIÓN")
    print(f"{'='*60}")
    print(f"  v23 CyborgTransformer:   {test_acc*100:.2f}%")
    print(f"  v10 CyborgHybrid (ref):  71.32%")
    print(f"  Ensemble 4M (máximo):    72.88%")
    print(f"  Objetivo paper:          79.72%")

    if test_acc > 0.7288:
        print(f"\n*** NUEVO MÁXIMO: {test_acc*100:.2f}% (+{(test_acc-0.7288)*100:.2f}%) ***")
    elif test_acc > 0.7132:
        print(f"\n  Supera v10 baseline: +{(test_acc-0.7132)*100:.2f}%")
    else:
        print(f"\n  Gap vs v10: {(0.7132-test_acc)*100:.2f}%")

    print(f"\nBest Val Accuracy: {best_val_acc*100:.2f}%")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
