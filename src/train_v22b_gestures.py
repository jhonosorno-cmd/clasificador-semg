"""
Entrenamiento v22b: Clasificador 6 clases (solo gestos).

Etapa B de la cascada v22:
  - Entrada: señal EMG + features 48D
  - Salida: 6 clases (gestos 1-6 remapeados a 0-5)
  - Modelo: CyborgHybrid(num_classes=6)

Dataset:
  Solo ventanas de gestos activos (label in 1-6).
  Etiquetas remapeadas: gesto N -> clase N-1 (0-5).
  ~2,280 muestras train, 6 clases balanceadas (~380 cada una).
  Dataset pequeno -> augmentation mas agresiva.

Ventaja clave:
  Sin clase 0, el modelo aprende SOLO a distinguir gestos entre si.
  Dataset perfectamente balanceado -> no class weights necesarios.
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
# Dataset solo gestos (wrapper)
# ---------------------------------------------------------------------------

class GestureOnlyDataset(torch.utils.data.Dataset):
    """
    Wrapper sobre NinaProDB3Dataset que filtra solo ventanas de gestos.

    Etiquetas remapeadas: gesto N (1-6) -> clase N-1 (0-5).
    La clase 0 (reposo) se excluye completamente.
    """

    def __init__(self, base_dataset: NinaProDB3Dataset) -> None:
        self._base = base_dataset
        self._indices = [
            i for i, lbl in enumerate(base_dataset.labels) if lbl > 0
        ]
        print(f"  GestureOnly: {len(self._indices)} ventanas de gestos "
              f"(de {len(base_dataset)} totales)")

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        real_idx = self._indices[idx]
        signal, features, label = self._base[real_idx]
        # Remap: 1->0, 2->1, ..., 6->5
        gesture_label = torch.tensor(label.item() - 1, dtype=torch.long)
        return signal, features, gesture_label


def create_gesture_loaders(batch_size: int = BATCH_SIZE):
    """Crea DataLoaders con solo ventanas de gestos (6 clases)."""
    train_base = NinaProDB3Dataset(
        subjects=[1], reps_to_use=[1, 2, 3, 4], mode="train", target_subject=1
    )
    val_base = NinaProDB3Dataset(
        subjects=[1], reps_to_use=[5], mode="val", target_subject=1
    )
    test_base = NinaProDB3Dataset(
        subjects=[1], reps_to_use=[6], mode="test", target_subject=1
    )

    train_ds = GestureOnlyDataset(train_base)
    val_ds   = GestureOnlyDataset(val_base)
    test_ds  = GestureOnlyDataset(test_base)

    make = lambda ds, sh: torch.utils.data.DataLoader(
        ds, batch_size=min(batch_size, len(ds)),
        shuffle=sh, num_workers=0, drop_last=sh
    )
    return make(train_ds, True), make(val_ds, False), make(test_ds, False)


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
    print("ENTRENAMIENTO v22b: CLASIFICADOR 6 GESTOS (Sin reposo)")
    print("CyborgHybrid(num_classes=6, features=48D)")
    print("Dataset balanceado: ~380 ventanas x 6 clases")
    print("=" * 60)

    device = torch.device(DEVICE)
    print(f"\nDispositivo: {device}")
    CHECKPOINTS_PATH.mkdir(parents=True, exist_ok=True)

    print("\n[1/5] Cargando datos S1 (solo gestos)...")
    train_loader, val_loader, test_loader = create_gesture_loaders(batch_size=BATCH_SIZE)

    # Distribución de clases en train
    train_labels = [train_loader.dataset._base.labels[i]
                    for i in train_loader.dataset._indices]
    from collections import Counter
    dist = Counter(train_labels)
    print("  Distribucion train por clase original:")
    for cls in sorted(dist):
        print(f"    Gesto {cls} (-> clase {cls-1}): {dist[cls]} muestras")

    print("\n[2/5] Creando CyborgHybrid 6 clases...")
    model = CyborgHybrid(num_classes=6, manual_features_dim=48).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parametros: {total_params:,}")

    print("\n[3/5] Configurando optimizer y loss...")
    # Dataset balanceado: pesos uniformes
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=7
    )

    patience_es      = 20   # mas paciencia por dataset pequeno
    best_val_acc     = 0.0
    patience_counter = 0
    checkpoint_path  = CHECKPOINTS_PATH / "best_model_v22b_gestures.pth"

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
                "num_classes": 6,
                "manual_features_dim": 48,
                "label_mapping": "gesture_N -> class_N-1 (0-5)",
            }, checkpoint_path)
            print(f"  [OK] Checkpoint guardado (Val Acc: {val_acc*100:.2f}%)")
        else:
            patience_counter += 1
            print(f"  [--] Sin mejora ({patience_counter}/{patience_es})")
            if patience_counter >= patience_es:
                print(f"\nEarly stopping (patience={patience_es})")
                break

    # Test (solo en ventanas de gestos)
    print("\n[5/5] Evaluando en test set (solo gestos)...")
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    _, test_acc, test_bal, test_preds, test_labels_list = validate_epoch(
        model, test_loader, criterion, device
    )
    cm = confusion_matrix(test_labels_list, test_preds)

    print(f"\n{'='*60}")
    print("RESULTADOS v22b — 6 gestos (clases 0-5 = gestos 1-6)")
    print(f"{'='*60}")
    print(f"Test Accuracy (gesture-only): {test_acc*100:.2f}%")
    print(f"Test Balanced Accuracy:       {test_bal*100:.2f}%")
    print(f"\nAccuracy por gesto:")
    for i in range(6):
        acc_i = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0.0
        print(f"  Gesto {i+1} (clase {i}): {acc_i*100:.2f}% ({cm[i,i]}/{cm[i].sum()})")
    print(f"\nConfusion Matrix (gestos 1-6 -> indices 0-5):")
    print("       " + "".join(f"G{i+1}    " for i in range(6)))
    for i in range(6):
        print(f"G{i+1}:   " + "".join(f"{cm[i,j]:5d} " for j in range(6)))
    print(f"\nMejor Val Acc: {best_val_acc*100:.2f}%")
    print(f"Checkpoint:    {checkpoint_path}")
    print(f"\nNota: Esta es la accuracy SOLO sobre ventanas de gestos.")
    print(f"La accuracy en cascada completa se evalua con eval_v22_cascade.py")


if __name__ == "__main__":
    main()
