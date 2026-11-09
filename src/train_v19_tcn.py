"""
Entrenamiento v19: TCNCyborgHybrid (S1)
Modelo: TCNCyborgHybrid (src/models_v19.py)
  - Reemplaza CNN+BiLSTM por TCN de 4 bloques residuales dilatados
  - Dilaciones [1, 2, 4, 8], kernel=7 no-causal -> RF = 361 samples
  - Misma augmentation que v15 (noise + scaling + channel dropout)
  - Mismo feature set que v10 (48D)
  - Identico pipeline de entrenamiento que v18
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from src.config import DEVICE, BATCH_SIZE, LEARNING_RATE, EPOCHS, CHECKPOINTS_PATH, SEED
from src.models_v19 import TCNCyborgHybrid
from src.data_loader import create_data_loaders
from src.train_v15_augmentation import (
    augment_signals,
    calculate_class_weights,
    validate_epoch,
)

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Train epoch
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    max_grad_norm: float = 1.0,
) -> tuple[float, float]:
    """
    Entrena una epoca con augmentation de senal EMG.

    Args:
        model: TCNCyborgHybrid a entrenar.
        loader: DataLoader de entrenamiento.
        criterion: Funcion de perdida.
        optimizer: Optimizador.
        device: Dispositivo (cpu/cuda).
        max_grad_norm: Umbral de gradient clipping.

    Returns:
        Tuple (avg_loss, accuracy).
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="Train", leave=False)
    for signals, features, labels in pbar:
        signals  = signals.to(device)
        features = features.to(device)
        labels   = labels.to(device)
        signals  = augment_signals(signals)
        optimizer.zero_grad()
        outputs = model(signals, features)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total   += labels.size(0)
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{correct/total:.3f}"})
    return total_loss / len(loader), correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entrena TCNCyborgHybrid en S1 con augmentation."""
    print("=" * 60)
    print("ENTRENAMIENTO v19: TCN CYBORG HYBRID")
    print("TCN dilated (d=[1,2,4,8]) + Manual Features (48D)")
    print("=" * 60)

    device = torch.device(DEVICE)
    print(f"\nDispositivo: {device}")
    CHECKPOINTS_PATH.mkdir(parents=True, exist_ok=True)

    # Datos S1 (features 48D, identico a v10/v15/v18)
    print("\n[1/5] Cargando datos S1...")
    train_loader, val_loader, test_loader = create_data_loaders(batch_size=BATCH_SIZE)
    print(f"  Train: {len(train_loader.dataset)} muestras")
    print(f"  Val:   {len(val_loader.dataset)} muestras")
    print(f"  Test:  {len(test_loader.dataset)} muestras")

    # Modelo TCN
    print("\n[2/5] Creando TCNCyborgHybrid...")
    model = TCNCyborgHybrid().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parametros: {total_params:,}")
    print(f"  Dilaciones TCN:   [1, 2, 4, 8]  |  Kernel: 7  |  RF: 361 samples")
    print(f"  Canales TCN:      64 -> 128 -> 256 -> 256")

    # Loss y optimizer
    print("\n[3/5] Configurando optimizer y loss...")
    class_weights = calculate_class_weights(train_loader.dataset).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    optimizer     = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler     = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # Entrenamiento
    patience_es      = 15
    best_val_acc     = 0.0
    patience_counter = 0
    checkpoint_path  = CHECKPOINTS_PATH / "best_model_v19_tcn.pth"

    print(f"\n[4/5] Iniciando entrenamiento (max {EPOCHS} epochs, patience={patience_es})...")

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
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
        print(f"LR: {current_lr:.2e}")

        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_acc,
                    "val_balanced_accuracy": val_bal_acc,
                    "architecture": "TCNCyborgHybrid",
                    "tcn_channels": [64, 128, 256, 256],
                    "dilations": [1, 2, 4, 8],
                    "kernel_size": 7,
                    "manual_features_dim": 48,
                },
                checkpoint_path,
            )
            print(f"  [OK] Checkpoint guardado (Val Acc: {val_acc*100:.2f}%)")
        else:
            patience_counter += 1
            print(f"  [--] Sin mejora ({patience_counter}/{patience_es})")
            if patience_counter >= patience_es:
                print(f"\nEarly stopping activado (patience={patience_es})")
                break

    # Evaluacion final
    print("\n[5/5] Evaluando en test set (S1)...")
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    _, test_acc, test_bal_acc, test_preds, test_labels = validate_epoch(
        model, test_loader, criterion, device
    )

    cm = confusion_matrix(test_labels, test_preds)

    print(f"\n{'='*60}")
    print("RESULTADOS FINALES - v19 TCN")
    print(f"{'='*60}")
    print(f"Test Accuracy:          {test_acc*100:.2f}%")
    print(f"Test Balanced Accuracy: {test_bal_acc*100:.2f}%")

    print("\nConfusion Matrix:")
    print("       " + "".join(f"Pred{i} " for i in range(7)))
    for i in range(7):
        print(f"Real{i}: " + "".join(f"{cm[i,j]:5d} " for j in range(7)))

    print("\nAccuracy por clase:")
    for i in range(7):
        acc_i = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0.0
        print(f"  Clase {i}: {acc_i*100:.2f}% ({cm[i,i]}/{cm[i].sum()})")

    print(f"\n{'='*60}")
    print("COMPARACION CON BASELINES")
    print(f"{'='*60}")
    print(f"v19 TCN (S1):                  {test_acc*100:.2f}% (BalAcc: {test_bal_acc*100:.2f}%)")
    print(f"v18 Attention (S1):            71.63% (BalAcc: 45.37%)")
    print(f"v10 Cyborg (S1):               71.32% (BalAcc: 41.92%)")
    print(f"Ensemble 4M (v10+v15+v17+v18): 72.78%")
    print(f"Objetivo paper:                79.72%")

    if test_acc > 0.7972:
        print(f"\n*** OBJETIVO ALCANZADO! (+{(test_acc - 0.7972)*100:.2f}%) ***")
    elif test_acc > 0.7132:
        print(f"\n*** SUPERA v10! (+{(test_acc - 0.7132)*100:.2f}%) ***")
        print(f"Gap al objetivo: {(0.7972 - test_acc)*100:.2f}%")
    else:
        print(f"\nDebajo de v10 por {(0.7132 - test_acc)*100:.2f}%")

    print(f"\nMejor Val Acc: {best_val_acc*100:.2f}%")
    print(f"Checkpoint:    {checkpoint_path}")


if __name__ == "__main__":
    main()
