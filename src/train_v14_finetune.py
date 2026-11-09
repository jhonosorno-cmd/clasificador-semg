"""
Entrenamiento v14: Fine-tuning S2+S3+S4 → S1
Estrategia en 2 fases:
  Fase 1: Pre-entrenar con S2+S3+S4 (features generales, sin Mixup)
  Fase 2: Fine-tune en S1 con LR diferencial baja + Mixup
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from src.config import DEVICE, BATCH_SIZE, CHECKPOINTS_PATH, SEED
from src.models_cyborg import CyborgHybrid
from src.data_loader import create_data_loaders

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Helpers (reusados de v13)
# ---------------------------------------------------------------------------

def mixup_data(
    x_signal: torch.Tensor,
    x_features: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.4,
    device: str = "cpu",
):
    """
    Mixup Augmentation: Mezcla pares de ejemplos y sus etiquetas.

    Args:
        x_signal: Tensor (Batch, 1, 125, 13) - señales EMG.
        x_features: Tensor (Batch, 48) - features manuales.
        y: Tensor (Batch,) - etiquetas.
        alpha: Parámetro de distribución Beta.
        device: Dispositivo (cpu/cuda).

    Returns:
        Tuple con (mixed_signal, mixed_features, y_a, y_b, lam).
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x_signal.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_signal = lam * x_signal + (1 - lam) * x_signal[index]
    mixed_features = lam * x_features + (1 - lam) * x_features[index]
    y_a, y_b = y, y[index]
    return mixed_signal, mixed_features, y_a, y_b, lam


def calculate_class_weights(dataset) -> torch.Tensor:
    """Calcula pesos sqrt-inverse de clase a partir de un dataset."""
    labels = dataset.labels
    counts = np.bincount(labels, minlength=7).astype(np.float64)
    counts[counts == 0] = 1.0
    total = counts.sum()
    weights = np.sqrt(total / (len(counts) * counts))
    print("Class weights (sqrt-inverse):")
    for i in range(len(counts)):
        print(f"  Clase {i}: {int(counts[i])} muestras, peso {weights[i]:.3f}")
    return torch.FloatTensor(weights)


def train_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    max_grad_norm: float = 1.0,
    use_mixup: bool = True,
) -> tuple[float, float]:
    """
    Entrena una época con Mixup opcional.

    Args:
        model: Modelo a entrenar.
        loader: DataLoader de entrenamiento.
        criterion: Función de pérdida.
        optimizer: Optimizador.
        device: Dispositivo (cpu/cuda).
        max_grad_norm: Umbral de gradient clipping.
        use_mixup: Si aplicar Mixup.

    Returns:
        Tuple (avg_loss, accuracy).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for signals, features, labels in pbar:
        signals = signals.to(device)
        features = features.to(device)
        labels = labels.to(device)

        if use_mixup:
            signals, features, labels_a, labels_b, lam = mixup_data(
                signals, features, labels, alpha=0.4, device=device
            )

        optimizer.zero_grad()
        outputs = model(signals, features)

        if use_mixup:
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        else:
            loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        if use_mixup:
            target = labels_a if lam >= 0.5 else labels_b
            correct += (predicted == target).sum().item()
            total += labels_a.size(0)
        else:
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{correct/total:.3f}"})

    return total_loss / len(loader), correct / total


def validate_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float, float, list, list]:
    """
    Valida el modelo.

    Args:
        model: Modelo a evaluar.
        loader: DataLoader de validación.
        criterion: Función de pérdida.
        device: Dispositivo (cpu/cuda).

    Returns:
        Tuple (avg_loss, accuracy, balanced_accuracy, preds, labels).
    """
    model.eval()
    total_loss = 0.0
    all_preds: list = []
    all_labels: list = []

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

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), accuracy, balanced_acc, all_preds, all_labels


# ---------------------------------------------------------------------------
# Fase 1: Pre-entrenamiento con S2+S3+S4
# ---------------------------------------------------------------------------

def phase1_pretrain(device: str) -> CyborgHybrid:
    """
    Pre-entrena CyborgHybrid con datos de S2, S3, S4.

    Args:
        device: Dispositivo de cómputo.

    Returns:
        Modelo con el mejor checkpoint de pre-entrenamiento cargado.
    """
    print("=" * 60)
    print("FASE 1: PRE-ENTRENAMIENTO (S2 + S3 + S4)")
    print("=" * 60)

    pretrain_subjects = [2, 3, 4]
    target_subject = 1
    lr_pretrain = 0.0005
    patience_pretrain = 10
    max_epochs_pretrain = 60
    checkpoint_path = CHECKPOINTS_PATH / "best_model_v14_pretrain.pth"

    print(f"\nTrain subjects: {pretrain_subjects}")
    print(f"Val/Test subject: S{target_subject}")
    print(f"LR: {lr_pretrain} | Patience: {patience_pretrain} | Max epochs: {max_epochs_pretrain}")

    # Datos
    print("\n[F1-1/4] Cargando datos S2+S3+S4...")
    train_loader, val_loader, _ = create_data_loaders(
        train_subjects=pretrain_subjects,
        target_subject=target_subject,
        batch_size=BATCH_SIZE,
    )
    print(f"  Train: {len(train_loader.dataset)} muestras")
    print(f"  Val:   {len(val_loader.dataset)} muestras (S{target_subject})")

    # Modelo
    print("\n[F1-2/4] Creando modelo desde cero...")
    model = CyborgHybrid().to(device)
    model.get_model_info()

    # Loss y optimizer
    print("\n[F1-3/4] Configurando optimizer y loss...")
    class_weights = calculate_class_weights(train_loader.dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_pretrain, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # Entrenamiento sin Mixup (datos abundantes de 3 sujetos)
    print("\n[F1-4/4] Entrenando (sin Mixup)...")
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(max_epochs_pretrain):
        print(f"\nEpoch {epoch+1}/{max_epochs_pretrain}")
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, use_mixup=False
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
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_acc,
                    "val_balanced_accuracy": val_bal_acc,
                    "train_subjects": pretrain_subjects,
                    "target_subject": target_subject,
                },
                checkpoint_path,
            )
            print(f"[OK] Pretrain checkpoint guardado (Val Acc: {val_acc*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience_pretrain:
                print(f"\nEarly stopping (patience={patience_pretrain})")
                break

    print(f"\nFase 1 completada. Mejor Val Acc: {best_val_acc*100:.2f}%")

    # Cargar mejor checkpoint
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Modelo pre-entrenado cargado desde: {checkpoint_path}")
    return model


# ---------------------------------------------------------------------------
# Fase 2: Fine-tuning en S1 con LR diferencial
# ---------------------------------------------------------------------------

def phase2_finetune(model: CyborgHybrid, device: str) -> CyborgHybrid:
    """
    Fine-tune del modelo pre-entrenado usando únicamente datos de S1.

    Args:
        model: Modelo pre-entrenado en S2+S3+S4.
        device: Dispositivo de cómputo.

    Returns:
        Modelo con el mejor checkpoint de fine-tuning cargado.
    """
    print("\n" + "=" * 60)
    print("FASE 2: FINE-TUNING (S1 solo, LR diferencial)")
    print("=" * 60)

    target_subject = 1
    patience_finetune = 15
    max_epochs_finetune = 80
    checkpoint_path = CHECKPOINTS_PATH / "best_model_v14_finetune.pth"

    print(f"\nTrain/Val/Test subject: S{target_subject}")
    print(f"Mixup alpha: 0.4 | Patience: {patience_finetune} | Max epochs: {max_epochs_finetune}")
    print("LR diferencial:")
    print("  conv1, conv2, conv3 -> 5e-5  (preservar features de bajo nivel)")
    print("  lstm                -> 1e-4")
    print("  feature_mlp         -> 1e-4")
    print("  classifier          -> 5e-4  (adaptar cabeza de clasificacion)")

    # Datos S1
    print("\n[F2-1/4] Cargando datos S1...")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_subjects=[target_subject],
        target_subject=target_subject,
        batch_size=BATCH_SIZE,
    )
    print(f"  Train: {len(train_loader.dataset)} muestras")
    print(f"  Val:   {len(val_loader.dataset)} muestras")
    print(f"  Test:  {len(test_loader.dataset)} muestras")

    # Loss y optimizer diferencial
    print("\n[F2-2/4] Configurando optimizer diferencial...")
    class_weights = calculate_class_weights(train_loader.dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.conv1.parameters(), "lr": 5e-5},
            {"params": model.conv2.parameters(), "lr": 5e-5},
            {"params": model.conv3.parameters(), "lr": 5e-5},
            {"params": model.lstm.parameters(), "lr": 1e-4},
            {"params": model.feature_mlp.parameters(), "lr": 1e-4},
            {"params": model.classifier.parameters(), "lr": 5e-4},
        ],
        weight_decay=1e-4,
    )
    # Scheduler basado en la LR del grupo de mayor LR (clasificador)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # Fine-tuning con Mixup
    print("\n[F2-3/4] Fine-tuning con Mixup alpha=0.4...")
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(max_epochs_finetune):
        print(f"\nEpoch {epoch+1}/{max_epochs_finetune}")
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, use_mixup=True
        )
        val_loss, val_acc, val_bal_acc, _, _ = validate_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step(val_acc)
        # Reportar LR del clasificador (la más alta)
        current_lr_clf = optimizer.param_groups[5]["lr"]
        current_lr_cnn = optimizer.param_groups[0]["lr"]

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | BalAcc: {val_bal_acc*100:.2f}%")
        print(f"LR CNN: {current_lr_cnn:.2e} | LR Clf: {current_lr_clf:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_acc,
                    "val_balanced_accuracy": val_bal_acc,
                    "train_subjects": [target_subject],
                    "target_subject": target_subject,
                    "phase": "finetune",
                },
                checkpoint_path,
            )
            print(f"[OK] Finetune checkpoint guardado (Val Acc: {val_acc*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience_finetune:
                print(f"\nEarly stopping (patience={patience_finetune})")
                break

    print(f"\nFase 2 completada. Mejor Val Acc: {best_val_acc*100:.2f}%")

    # Cargar mejor checkpoint
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Modelo fine-tuned cargado desde: {checkpoint_path}")
    return model, test_loader, criterion


# ---------------------------------------------------------------------------
# Evaluación final
# ---------------------------------------------------------------------------

def evaluate_final(
    model: CyborgHybrid,
    test_loader,
    criterion: nn.Module,
    device: str,
) -> None:
    """
    Evalúa el modelo en el test set de S1 e imprime comparación con baselines.

    Args:
        model: Modelo fine-tuned.
        test_loader: DataLoader del test set (S1).
        criterion: Función de pérdida.
        device: Dispositivo de cómputo.
    """
    print("\n[F2-4/4] Evaluando en test set (S1)...")
    test_loss, test_acc, test_bal_acc, test_preds, test_labels = validate_epoch(
        model, test_loader, criterion, device
    )

    print(f"\n{'='*60}")
    print("RESULTADOS FINALES - v14 FINE-TUNING")
    print(f"{'='*60}")
    print(f"Test Accuracy:          {test_acc*100:.2f}%")
    print(f"Test Balanced Accuracy: {test_bal_acc*100:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    print("\nConfusion Matrix:")
    header = "       " + "".join(f"Pred{i} " for i in range(7))
    print(header)
    for i in range(7):
        row = f"Real{i}: " + "".join(f"{cm[i,j]:5d} " for j in range(7))
        print(row)

    print("\nAccuracy por clase:")
    for i in range(7):
        class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0.0
        print(f"  Clase {i}: {class_acc*100:.2f}% ({cm[i,i]}/{cm[i].sum()})")

    # Comparación con baselines
    print(f"\n{'='*60}")
    print("COMPARACION CON BASELINES")
    print(f"{'='*60}")
    print(f"v14 Fine-tuning (S2+S3+S4->S1): {test_acc*100:.2f}% (BalAcc: {test_bal_acc*100:.2f}%)")
    print(f"v13 Multi-subject (S1-S4):       68.91%")
    print(f"v10 Cyborg (S1 only):            71.32% (BalAcc: 41.92%)")
    print(f"Objetivo paper:                  79.72%")

    if test_acc > 0.7972:
        print(f"\n*** OBJETIVO ALCANZADO! (+{(test_acc - 0.7972)*100:.2f}%) ***")
    elif test_acc > 0.7132:
        print(f"\n*** SUPERA v10 CYBORG (+{(test_acc - 0.7132)*100:.2f}%) ***")
        gap = (0.7972 - test_acc) * 100
        print(f"Gap al objetivo: {gap:.2f}%")
    else:
        gap_v10 = (0.7132 - test_acc) * 100
        gap_obj = (0.7972 - test_acc) * 100
        print(f"\nDebajo de v10 Cyborg por {gap_v10:.2f}%")
        print(f"Gap al objetivo: {gap_obj:.2f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Ejecuta el pipeline completo de fine-tuning v14."""
    print("=" * 60)
    print("ENTRENAMIENTO v14: FINE-TUNING S2+S3+S4 -> S1")
    print("=" * 60)

    device = torch.device(DEVICE)
    print(f"\nDispositivo: {device}")
    CHECKPOINTS_PATH.mkdir(parents=True, exist_ok=True)

    # Fase 1: saltar si ya existe checkpoint
    pretrain_ckpt = CHECKPOINTS_PATH / "best_model_v14_pretrain.pth"
    if pretrain_ckpt.exists():
        print(f"\n[SKIP] Checkpoint pretrain encontrado: {pretrain_ckpt}")
        pretrained_model = CyborgHybrid().to(device)
        ckpt = torch.load(pretrain_ckpt, weights_only=False)
        pretrained_model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Val Acc pretrain: {ckpt.get('val_accuracy', '?'):.4f}")
    else:
        pretrained_model = phase1_pretrain(str(device))

    # Fase 2
    finetuned_model, test_loader, criterion = phase2_finetune(pretrained_model, str(device))

    # Evaluación final
    evaluate_final(finetuned_model, test_loader, criterion, str(device))

    print(f"\nCheckpoints guardados en: {CHECKPOINTS_PATH}")
    print(f"  Pretrain: best_model_v14_pretrain.pth")
    print(f"  Finetune: best_model_v14_finetune.pth")


if __name__ == "__main__":
    main()
