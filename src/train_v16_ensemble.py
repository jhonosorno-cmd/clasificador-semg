"""
Entrenamiento v16: Ensemble de modelos CyborgHybrid
Estrategia:
  1. Evaluar ensemble inmediato con checkpoints existentes (v10 + v15)
  2. Entrenar N modelos con seeds distintas usando augmentation (v15)
  3. Ensemble final: promedio de softmax de todos los modelos
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from src.config import DEVICE, BATCH_SIZE, LEARNING_RATE, EPOCHS, CHECKPOINTS_PATH, SEED
from src.models_cyborg import CyborgHybrid
from src.data_loader import create_data_loaders
from src.train_v15_augmentation import augment_signals, calculate_class_weights, train_epoch, validate_epoch


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

def evaluate_ensemble(
    models: list,
    loader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float, list, list]:
    """
    Evalua un ensemble de modelos promediando softmax.

    Args:
        models: Lista de modelos CyborgHybrid en modo eval.
        loader: DataLoader de evaluacion.
        criterion: Funcion de perdida (sobre el promedio).
        device: Dispositivo de computo.

    Returns:
        Tuple (accuracy, balanced_accuracy, preds, labels).
    """
    for m in models:
        m.eval()

    all_preds: list = []
    all_labels: list = []

    with torch.no_grad():
        for signals, features, labels in tqdm(loader, desc="Ensemble", leave=False):
            signals = signals.to(device)
            features = features.to(device)
            labels = labels.to(device)

            # Promediar softmax de cada modelo
            probs_sum = torch.zeros(labels.size(0), 7, device=device)
            for m in models:
                logits = m(signals, features)
                probs_sum += torch.softmax(logits, dim=1)

            avg_probs = probs_sum / len(models)
            _, predicted = torch.max(avg_probs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    return accuracy, balanced_acc, all_preds, all_labels


def train_single_model(
    seed: int,
    train_loader,
    val_loader,
    device: str,
    checkpoint_name: str,
) -> float:
    """
    Entrena un modelo CyborgHybrid con una seed especifica y augmentation.

    Args:
        seed: Semilla de reproducibilidad.
        train_loader: DataLoader de entrenamiento.
        val_loader: DataLoader de validacion.
        device: Dispositivo de computo.
        checkpoint_name: Nombre del archivo de checkpoint.

    Returns:
        Mejor val accuracy alcanzada.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = CyborgHybrid().to(device)
    class_weights = calculate_class_weights(train_loader.dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    best_val_acc = 0.0
    patience_counter = 0
    patience_es = 15
    checkpoint_path = CHECKPOINTS_PATH / checkpoint_name

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_bal_acc, _, _ = validate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"  Epoch {epoch+1}/{EPOCHS} | Train {train_acc*100:.1f}% | Val {val_acc*100:.2f}% | BalAcc {val_bal_acc*100:.2f}% | LR {current_lr:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_accuracy": val_acc,
                "val_balanced_accuracy": val_bal_acc,
                "seed": seed,
            }, checkpoint_path)
            print(f"  [OK] Checkpoint guardado (Val Acc: {val_acc*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience_es:
                print(f"  Early stopping (patience={patience_es})")
                break

    return best_val_acc


def load_model(checkpoint_path: str, device: str) -> CyborgHybrid:
    """Carga un modelo desde checkpoint."""
    model = CyborgHybrid().to(device)
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    val_acc = ckpt.get("val_accuracy", 0)
    print(f"  Cargado: {checkpoint_path} (Val Acc: {val_acc*100:.2f}%)")
    return model


def print_results(
    label: str,
    test_acc: float,
    test_bal_acc: float,
    test_preds: list,
    test_labels: list,
) -> None:
    """Imprime resultados de evaluacion y comparacion con baselines."""
    cm = confusion_matrix(test_labels, test_preds)
    print(f"\n{'='*60}")
    print(f"RESULTADOS: {label}")
    print(f"{'='*60}")
    print(f"Test Accuracy:          {test_acc*100:.2f}%")
    print(f"Test Balanced Accuracy: {test_bal_acc*100:.2f}%")

    print("\nAccuracy por clase:")
    for i in range(7):
        acc_i = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0.0
        print(f"  Clase {i}: {acc_i*100:.2f}% ({cm[i,i]}/{cm[i].sum()})")

    print(f"\n{'='*60}")
    print("COMPARACION CON BASELINES")
    print(f"{'='*60}")
    print(f"{label}: {test_acc*100:.2f}% (BalAcc: {test_bal_acc*100:.2f}%)")
    print(f"v15 Augmentation (S1): 70.74%")
    print(f"v10 Cyborg (S1):       71.32% (BalAcc: 41.92%)")
    print(f"Objetivo paper:        79.72%")

    if test_acc > 0.7972:
        print(f"\n*** OBJETIVO ALCANZADO! (+{(test_acc - 0.7972)*100:.2f}%) ***")
    elif test_acc > 0.7132:
        print(f"\n*** SUPERA v10! (+{(test_acc - 0.7132)*100:.2f}%) ***")
        print(f"Gap al objetivo: {(0.7972 - test_acc)*100:.2f}%")
    else:
        print(f"\nDebajo de v10 por {(0.7132 - test_acc)*100:.2f}%")
        print(f"Gap al objetivo: {(0.7972 - test_acc)*100:.2f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Pipeline completo de ensemble."""
    print("=" * 60)
    print("ENTRENAMIENTO v16: ENSEMBLE")
    print("=" * 60)

    device = torch.device(DEVICE)
    print(f"\nDispositivo: {device}")
    CHECKPOINTS_PATH.mkdir(parents=True, exist_ok=True)

    # Datos S1
    print("\n[1] Cargando datos S1...")
    train_loader, val_loader, test_loader = create_data_loaders(batch_size=BATCH_SIZE)
    print(f"  Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    class_weights = calculate_class_weights(train_loader.dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # -----------------------------------------------------------------------
    # PASO A: Ensemble inmediato con checkpoints existentes (v10 + v15)
    # -----------------------------------------------------------------------
    print("\n[2] ENSEMBLE INMEDIATO: v10 + v15")
    existing_checkpoints = [
        CHECKPOINTS_PATH / "best_model_cyborg.pth",       # v10: 71.32%
        CHECKPOINTS_PATH / "best_model_v15_augmentation.pth",  # v15: 70.74%
    ]
    existing_models = []
    for ckpt_path in existing_checkpoints:
        if ckpt_path.exists():
            existing_models.append(load_model(str(ckpt_path), device))
        else:
            print(f"  [WARN] No encontrado: {ckpt_path}")

    if len(existing_models) >= 2:
        acc_2, bal_2, preds_2, labels_2 = evaluate_ensemble(
            existing_models, test_loader, criterion, device
        )
        print_results("Ensemble v10+v15 (2 modelos)", acc_2, bal_2, preds_2, labels_2)
    else:
        print("  Solo hay 1 checkpoint disponible, omitiendo ensemble inmediato.")

    # -----------------------------------------------------------------------
    # PASO B: Entrenar 3 modelos adicionales con seeds distintas
    # -----------------------------------------------------------------------
    seeds = [7, 13, 99]
    print(f"\n[3] ENTRENANDO {len(seeds)} MODELOS ADICIONALES (seeds: {seeds})...")

    new_checkpoints = []
    for i, seed in enumerate(seeds):
        ckpt_name = f"best_model_v16_seed{seed}.pth"
        ckpt_path = CHECKPOINTS_PATH / ckpt_name
        print(f"\n--- Modelo {i+1}/{len(seeds)} | seed={seed} ---")

        if ckpt_path.exists():
            print(f"  Checkpoint existente, omitiendo entrenamiento.")
        else:
            best_val = train_single_model(seed, train_loader, val_loader, device, ckpt_name)
            print(f"  Completado. Mejor Val Acc: {best_val*100:.2f}%")

        new_checkpoints.append(ckpt_path)

    # -----------------------------------------------------------------------
    # PASO C: Ensemble completo (hasta 5 modelos)
    # -----------------------------------------------------------------------
    print("\n[4] ENSEMBLE COMPLETO")
    all_checkpoint_paths = existing_checkpoints + new_checkpoints
    all_models = []
    for ckpt_path in all_checkpoint_paths:
        if ckpt_path.exists():
            all_models.append(load_model(str(ckpt_path), device))

    print(f"\nEnsemble con {len(all_models)} modelos:")
    acc_all, bal_all, preds_all, labels_all = evaluate_ensemble(
        all_models, test_loader, criterion, device
    )
    print_results(f"Ensemble completo ({len(all_models)} modelos)", acc_all, bal_all, preds_all, labels_all)

    # Guardar predicciones del ensemble
    torch.save(
        {
            "test_accuracy": acc_all,
            "test_balanced_accuracy": bal_all,
            "n_models": len(all_models),
            "seeds": [SEED] + [SEED] + seeds,
        },
        CHECKPOINTS_PATH / "ensemble_v16_results.pth",
    )
    print(f"\nResultados guardados en: {CHECKPOINTS_PATH / 'ensemble_v16_results.pth'}")


if __name__ == "__main__":
    main()
