"""
LORO-CV: Leave-One-Repetition-Out Cross-Validation

Entrena uno de los modelos (v10, v15, v17, v18) en un fold específico.

Esquema por fold k ∈ {1,...,6}:
    test  = rep k
    val   = rep (k-2) % 6 + 1   (rep anterior, circular)
    train = las 4 repeticiones restantes

Nota: fold 6 reproduce el split original (train=1-4, val=5, test=6).

Uso:
    python -m src.train_loro_cv --model v10 --fold 1
    python -m src.train_loro_cv --model v17 --fold 3
    python -m src.train_loro_cv --model v18 --fold 5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    BATCH_SIZE,
    CHECKPOINTS_PATH,
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    SEED,
)
from src.data_loader import NinaProDB3Dataset
from src.models_cyborg import CyborgHybrid
from src.models_v18 import CyborgHybridAttention
from src.train_v15_augmentation import augment_signals
from src.train_v17_features import NinaProExtendedFeatures

torch.manual_seed(SEED)
np.random.seed(SEED)

LORO_CHECKPOINT_DIR = CHECKPOINTS_PATH / "loro"

# --------------------------------------------------------------------------- #
# Fold helpers
# --------------------------------------------------------------------------- #

def get_fold_reps(fold: int) -> tuple[list[int], list[int], list[int]]:
    """
    Devuelve (train_reps, val_reps, test_reps) para el fold k.

    Args:
        fold: Pliegue k ∈ {1,...,6}.

    Returns:
        Tupla (train_reps, val_reps, test_reps).
    """
    test_rep = fold
    val_rep = (fold - 2) % 6 + 1
    train_reps = [r for r in range(1, 7) if r != test_rep and r != val_rep]
    return train_reps, [val_rep], [test_rep]


# --------------------------------------------------------------------------- #
# Dataset factory
# --------------------------------------------------------------------------- #

def build_loaders(
    model_name: str,
    train_reps: list[int],
    val_reps: list[int],
    test_reps: list[int],
    batch_size: int = BATCH_SIZE,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Crea DataLoaders para un fold dado según el modelo.

    v17 usa NinaProExtendedFeatures (132D); los demás usan NinaProDB3Dataset (48D).
    v15 activa augmentation online en train.

    Args:
        model_name: 'v10', 'v15', 'v17' o 'v18'.
        train_reps: Repeticiones de entrenamiento.
        val_reps: Repeticiones de validación.
        test_reps: Repeticiones de prueba.
        batch_size: Tamaño de batch.

    Returns:
        train_loader, val_loader, test_loader
    """
    dataset_cls = NinaProExtendedFeatures if model_name == "v17" else NinaProDB3Dataset
    use_augment = model_name == "v15"

    train_ds = dataset_cls(
        subjects=[1],
        reps_to_use=train_reps,
        mode="train",
        augment=use_augment,
        target_subject=1,
    )
    val_ds = dataset_cls(
        subjects=[1],
        reps_to_use=val_reps,
        mode="val",
        target_subject=1,
    )
    test_ds = dataset_cls(
        subjects=[1],
        reps_to_use=test_reps,
        mode="test",
        target_subject=1,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


# --------------------------------------------------------------------------- #
# Model factory
# --------------------------------------------------------------------------- #

def build_model(model_name: str) -> nn.Module:
    """
    Instancia el modelo según nombre.

    Args:
        model_name: 'v10', 'v15', 'v17' o 'v18'.

    Returns:
        Modelo PyTorch sin inicializar en device.
    """
    if model_name in ("v10", "v15"):
        return CyborgHybrid(manual_features_dim=48)
    elif model_name == "v17":
        return CyborgHybrid(manual_features_dim=132)
    elif model_name == "v18":
        return CyborgHybridAttention(manual_features_dim=48)
    else:
        raise ValueError(f"Modelo desconocido: {model_name}")


# --------------------------------------------------------------------------- #
# Class weights
# --------------------------------------------------------------------------- #

def calculate_class_weights(dataset: NinaProDB3Dataset) -> torch.Tensor:
    """Pesos sqrt-inverse por clase."""
    counts = np.bincount(dataset.labels, minlength=7).astype(np.float64)
    counts[counts == 0] = 1.0
    total = counts.sum()
    weights = np.sqrt(total / (len(counts) * counts))
    return torch.FloatTensor(weights)


# --------------------------------------------------------------------------- #
# Train / validate
# --------------------------------------------------------------------------- #

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    use_augment: bool = False,
    max_grad_norm: float = 1.0,
) -> tuple[float, float]:
    """Entrena una época."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="Train", leave=False)

    for signals, features, labels in pbar:
        signals = signals.to(device)
        features = features.to(device)
        labels = labels.to(device)

        if use_augment:
            signals = augment_signals(signals)

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
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.3f}"})

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Evalúa el modelo en un DataLoader.

    Returns:
        avg_loss, accuracy, balanced_accuracy, all_preds, all_labels
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for signals, features, labels in tqdm(loader, desc="Eval", leave=False):
        signals = signals.to(device)
        features = features.to(device)
        labels = labels.to(device)

        outputs = model(signals, features)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    preds = np.array(all_preds)
    labels_arr = np.array(all_labels)
    acc = accuracy_score(labels_arr, preds)
    bal_acc = balanced_accuracy_score(labels_arr, preds)

    return total_loss / len(loader), acc, bal_acc, preds, labels_arr


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(model_name: str, fold: int) -> None:
    """
    Entrena el modelo dado en el fold especificado y reporta métricas.

    Args:
        model_name: 'v10', 'v15', 'v17' o 'v18'.
        fold: Pliegue k ∈ {1,...,6}.
    """
    print("=" * 60)
    print(f"LORO-CV  |  Modelo: {model_name.upper()}  |  Fold: {fold}/6")
    print("=" * 60)

    LORO_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(DEVICE)

    # Splits
    train_reps, val_reps, test_reps = get_fold_reps(fold)
    print(f"\nSplit — train: {train_reps}  val: {val_reps}  test: {test_reps}")

    # Datos
    print("\n[1/4] Cargando datos...")
    train_loader, val_loader, test_loader = build_loaders(
        model_name, train_reps, val_reps, test_reps
    )

    # Modelo
    print("\n[2/4] Creando modelo...")
    model = build_model(model_name).to(device)

    # Loss y optimizer
    class_weights = calculate_class_weights(train_loader.dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    use_augment = model_name == "v15"
    patience_es = 15
    best_val_acc = 0.0
    patience_counter = 0
    ckpt_path = LORO_CHECKPOINT_DIR / f"{model_name}_fold_{fold}.pth"

    # Entrenamiento
    print("\n[3/4] Entrenando...")
    print(f"Epochs: {EPOCHS} | LR: {LEARNING_RATE} | Patience: {patience_es}")
    print("-" * 60)

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, use_augment
        )
        val_loss, val_acc, val_bal_acc, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step(val_acc)
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Ep {epoch+1:3d}/{EPOCHS} | "
            f"Train {train_acc*100:.2f}% | "
            f"Val {val_acc*100:.2f}% (BalAcc {val_bal_acc*100:.2f}%) | "
            f"LR {lr:.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_accuracy": val_acc,
                    "val_balanced_accuracy": val_bal_acc,
                    "fold": fold,
                    "model_name": model_name,
                    "train_reps": train_reps,
                    "val_reps": val_reps,
                    "test_reps": test_reps,
                },
                ckpt_path,
            )
            print(f"  [SAVED] Val Acc: {val_acc*100:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience_es:
                print(f"\nEarly stopping (epoch {epoch+1})")
                break

    # Evaluación final en test
    print("\n[4/4] Evaluando en test set...")
    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    _, test_acc, test_bal_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )

    class_names = [f"Clase {i}" for i in range(7)]
    report = classification_report(
        test_labels, test_preds, target_names=class_names, digits=4
    )

    print(f"\n{'='*60}")
    print(f"RESULTADOS  |  {model_name.upper()}  |  Fold {fold}")
    print(f"{'='*60}")
    print(f"Accuracy:          {test_acc*100:.2f}%")
    print(f"Balanced Accuracy: {test_bal_acc*100:.2f}%")
    print(f"\n{report}")

    # Guardar métricas en JSON para agregación posterior
    metrics_path = LORO_CHECKPOINT_DIR / f"{model_name}_fold_{fold}_metrics.json"
    from sklearn.metrics import precision_score, recall_score, f1_score

    metrics = {
        "model": model_name,
        "fold": fold,
        "test_rep": test_reps[0],
        "accuracy": float(test_acc),
        "balanced_accuracy": float(test_bal_acc),
        "precision_macro": float(precision_score(test_labels, test_preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(test_labels, test_preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(test_labels, test_preds, average="macro", zero_division=0)),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Métricas guardadas en: {metrics_path}")
    print(f"Checkpoint guardado en: {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LORO-CV training")
    parser.add_argument(
        "--model",
        type=str,
        choices=["v10", "v15", "v17", "v18"],
        required=True,
        help="Modelo a entrenar",
    )
    parser.add_argument(
        "--fold",
        type=int,
        choices=list(range(1, 7)),
        required=True,
        help="Pliegue k ∈ {1,...,6}",
    )
    args = parser.parse_args()
    main(args.model, args.fold)
