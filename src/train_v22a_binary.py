"""
Entrenamiento v22a: Clasificador binario reposo vs gesto.

Etapa A de la cascada v22:
  - Entrada: señal EMG + features 48D
  - Salida: 2 clases (0=reposo, 1=gesto)
  - Modelo: CyborgHybrid(num_classes=2)

Dataset:
  Todos los windows de S1 con etiquetas remapeadas:
    clase 0 (reposo) -> 0
    clases 1-6 (gestos) -> 1
  Desbalance moderado: ~60% reposo, ~40% gestos.

Objetivo: maximizar el recall de gestos (minimizar falsos negativos
que routing incorrecto al clasificador de gestos).
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from tqdm import tqdm

from src.config import DEVICE, BATCH_SIZE, LEARNING_RATE, EPOCHS, CHECKPOINTS_PATH, SEED
from src.models_cyborg import CyborgHybrid
from src.data_loader import NinaProDB3Dataset
from src.train_v15_augmentation import augment_signals

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Dataset binario (wrapper)
# ---------------------------------------------------------------------------

class BinaryDataset(torch.utils.data.Dataset):
    """
    Wrapper sobre NinaProDB3Dataset que convierte etiquetas a binario.

    0 -> 0  (reposo)
    1-6 -> 1  (gesto)
    """

    def __init__(self, base_dataset: NinaProDB3Dataset) -> None:
        self._base = base_dataset

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int):
        signal, features, label = self._base[idx]
        binary = torch.tensor(1 if label.item() > 0 else 0, dtype=torch.long)
        return signal, features, binary


def create_binary_loaders(batch_size: int = BATCH_SIZE):
    """Crea DataLoaders binarios (rest vs gesture)."""
    train_base = NinaProDB3Dataset(
        subjects=[1], reps_to_use=[1, 2, 3, 4], mode="train", target_subject=1
    )
    val_base = NinaProDB3Dataset(
        subjects=[1], reps_to_use=[5], mode="val", target_subject=1
    )
    test_base = NinaProDB3Dataset(
        subjects=[1], reps_to_use=[6], mode="test", target_subject=1
    )

    train_ds = BinaryDataset(train_base)
    val_ds   = BinaryDataset(val_base)
    test_ds  = BinaryDataset(test_base)

    make = lambda ds, sh: torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=sh, num_workers=0, drop_last=sh
    )
    return make(train_ds, True), make(val_ds, False), make(test_ds, False)


# ---------------------------------------------------------------------------
# Class weights para binario
# ---------------------------------------------------------------------------

def binary_class_weights(base_labels: np.ndarray) -> torch.Tensor:
    """Pesos inversos para clases binarias."""
    binary = (base_labels > 0).astype(int)
    counts = np.bincount(binary, minlength=2).astype(float)
    weights = 1.0 / np.sqrt(counts)
    weights /= weights.sum()
    return torch.tensor(weights * 2, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, device, max_grad_norm=1.0):
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
        _, pred = torch.max(outputs, 1)
        correct += (pred == labels).sum().item()
        total   += labels.size(0)
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{correct/total:.3f}"})
    return total_loss / len(loader), correct / total


def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for signals, features, labels in tqdm(loader, desc="Val", leave=False):
            signals, features, labels = (
                signals.to(device), features.to(device), labels.to(device)
            )
            outputs = model(signals, features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
            total   += labels.size(0)
            all_preds.extend(pred.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    bal = balanced_accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), correct / total, bal, all_preds, all_labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("ENTRENAMIENTO v22a: CLASIFICADOR BINARIO (Reposo vs Gesto)")
    print("CyborgHybrid(num_classes=2, features=48D)")
    print("=" * 60)

    device = torch.device(DEVICE)
    print(f"\nDispositivo: {device}")
    CHECKPOINTS_PATH.mkdir(parents=True, exist_ok=True)

    print("\n[1/5] Cargando datos S1 (binario)...")
    train_loader, val_loader, test_loader = create_binary_loaders(batch_size=BATCH_SIZE)
    base_labels = train_loader.dataset._base.labels
    n_rest    = (base_labels == 0).sum()
    n_gesture = (base_labels > 0).sum()
    print(f"  Train: {len(train_loader.dataset)} muestras  (Reposo: {n_rest}, Gestos: {n_gesture})")
    print(f"  Val:   {len(val_loader.dataset)} muestras")
    print(f"  Test:  {len(test_loader.dataset)} muestras")

    print("\n[2/5] Creando CyborgHybrid binario...")
    model = CyborgHybrid(num_classes=2, manual_features_dim=48).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parametros: {total_params:,}")

    print("\n[3/5] Configurando optimizer y loss...")
    class_weights = binary_class_weights(base_labels).to(device)
    print(f"  Class weights: Reposo={class_weights[0]:.4f}  Gesto={class_weights[1]:.4f}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    patience_es      = 15
    best_val_acc     = 0.0
    patience_counter = 0
    checkpoint_path  = CHECKPOINTS_PATH / "best_model_v22a_binary.pth"

    print(f"\n[4/5] Entrenando (max {EPOCHS} epochs, patience={patience_es})...")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_bal, _, _ = validate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | BalAcc: {val_bal*100:.2f}%")
        print(f"LR: {current_lr:.2e}")

        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_accuracy": val_acc,
                "val_balanced_accuracy": val_bal,
                "num_classes": 2,
                "manual_features_dim": 48,
            }, checkpoint_path)
            print(f"  [OK] Checkpoint guardado (Val Acc: {val_acc*100:.2f}%)")
        else:
            patience_counter += 1
            print(f"  [--] Sin mejora ({patience_counter}/{patience_es})")
            if patience_counter >= patience_es:
                print(f"\nEarly stopping (patience={patience_es})")
                break

    # Test
    print("\n[5/5] Evaluando en test set...")
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    _, test_acc, test_bal, test_preds, test_labels = validate_epoch(
        model, test_loader, criterion, device
    )
    cm = confusion_matrix(test_labels, test_preds)

    print(f"\n{'='*60}")
    print("RESULTADOS v22a — Binario")
    print(f"{'='*60}")
    print(f"Test Accuracy:          {test_acc*100:.2f}%")
    print(f"Test Balanced Accuracy: {test_bal*100:.2f}%")
    print(f"\nConfusion Matrix (filas=real, cols=pred):")
    print(f"              Pred_Reposo  Pred_Gesto")
    print(f"  Real_Reposo     {cm[0,0]:5d}       {cm[0,1]:5d}   ({cm[0,0]/cm[0].sum()*100:.1f}% recall)")
    print(f"  Real_Gesto      {cm[1,0]:5d}       {cm[1,1]:5d}   ({cm[1,1]/cm[1].sum()*100:.1f}% recall)")
    print(f"\nMejor Val Acc: {best_val_acc*100:.2f}%")
    print(f"Checkpoint:    {checkpoint_path}")


if __name__ == "__main__":
    main()
