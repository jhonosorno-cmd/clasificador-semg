"""
Evaluacion de la cascada v22: Binario + 6-gestos.

Pipeline:
  1. Clasificador binario (v22a): rest(0) vs gesture(1)
  2. Si predice gesture: clasificador 6-clases (v22b) -> gesto 1-6
  3. Si predice rest:    prediccion final = 0

Metricas calculadas en las 1,914 ventanas de test (todas las clases):
  - 7-class accuracy (comparable con el paper)
  - 7-class balanced accuracy
  - Confusion matrix 7x7
  - Accuracy por clase
  - Estadisticas de routing (cuantas van a cada etapa)
"""

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

from src.config import DEVICE, BATCH_SIZE, CHECKPOINTS_PATH, SEED
from src.models_cyborg import CyborgHybrid
from src.data_loader import create_data_loaders

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Carga de modelos
# ---------------------------------------------------------------------------

def load_binary_model(path, device) -> CyborgHybrid:
    """Carga clasificador binario v22a."""
    model = CyborgHybrid(num_classes=2, manual_features_dim=48).to(device)
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    val_acc = ckpt.get("val_accuracy", 0.0)
    print(f"  Binario  (v22a): {path.name}  (Val {val_acc*100:.2f}%)")
    return model


def load_gesture_model(path, device) -> CyborgHybrid:
    """Carga clasificador 6-gestos v22b."""
    model = CyborgHybrid(num_classes=6, manual_features_dim=48).to(device)
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    val_acc = ckpt.get("val_accuracy", 0.0)
    print(f"  6-gestos (v22b): {path.name}  (Val {val_acc*100:.2f}%)")
    return model


# ---------------------------------------------------------------------------
# Inferencia en cascada
# ---------------------------------------------------------------------------

def cascade_predict(
    model_binary: CyborgHybrid,
    model_gesture: CyborgHybrid,
    loader,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predice con la cascada sobre todas las ventanas del loader.

    Returns:
        preds:  (N,) predicciones finales 0-6
        labels: (N,) etiquetas reales 0-6
    """
    all_preds:  list[int] = []
    all_labels: list[int] = []

    routed_rest    = 0
    routed_gesture = 0

    with torch.no_grad():
        for signals, features, labels in tqdm(loader, desc="Cascade", leave=False):
            signals  = signals.to(device)
            features = features.to(device)

            # Etapa A: binario
            bin_logits = model_binary(signals, features)
            bin_preds  = torch.argmax(bin_logits, dim=1)  # 0=rest, 1=gesture

            # Etapa B: 6-gestos (solo para los enrutados como gesture)
            gest_preds = torch.zeros(len(labels), dtype=torch.long)
            gesture_mask = bin_preds == 1
            if gesture_mask.any():
                gest_logits = model_gesture(
                    signals[gesture_mask], features[gesture_mask]
                )
                # Salida 0-5 → remap a 1-6
                gest_preds[gesture_mask] = torch.argmax(gest_logits, dim=1) + 1

            # Prediccion final: 0 si binario=rest, 1-6 si binario=gesture
            final_preds = torch.where(bin_preds == 0,
                                      torch.zeros_like(gest_preds),
                                      gest_preds)

            all_preds.extend(final_preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            routed_rest    += (bin_preds == 0).sum().item()
            routed_gesture += (bin_preds == 1).sum().item()

    total = len(all_preds)
    print(f"\n  Routing: {routed_rest} -> rest ({routed_rest/total*100:.1f}%)  |  "
          f"{routed_gesture} -> gesture ({routed_gesture/total*100:.1f}%)")

    return np.array(all_preds), np.array(all_labels)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("EVALUACION CASCADA v22: Binario + 6-Gestos")
    print("=" * 60)

    device = torch.device(DEVICE)
    print(f"\nDispositivo: {device}")

    print("\n[1/3] Cargando modelos...")
    model_binary  = load_binary_model(
        CHECKPOINTS_PATH / "best_model_v22a_binary.pth", device
    )
    model_gesture = load_gesture_model(
        CHECKPOINTS_PATH / "best_model_v22b_gestures.pth", device
    )

    print("\n[2/3] Cargando datos S1 (test set completo)...")
    _, val_loader, test_loader = create_data_loaders(batch_size=BATCH_SIZE)
    print(f"  Val:  {len(val_loader.dataset)} muestras")
    print(f"  Test: {len(test_loader.dataset)} muestras")

    print("\n[3/3] Evaluando cascada...")

    # Validacion
    print("\n--- Evaluacion en VALIDACION ---")
    val_preds, val_labels = cascade_predict(
        model_binary, model_gesture, val_loader, str(device)
    )
    val_acc = np.mean(val_preds == val_labels)
    val_bal = balanced_accuracy_score(val_labels, val_preds)
    print(f"Val Accuracy:          {val_acc*100:.2f}%")
    print(f"Val Balanced Accuracy: {val_bal*100:.2f}%")

    # Test
    print("\n--- Evaluacion en TEST ---")
    test_preds, test_labels = cascade_predict(
        model_binary, model_gesture, test_loader, str(device)
    )
    test_acc = np.mean(test_preds == test_labels)
    test_bal = balanced_accuracy_score(test_labels, test_preds)
    cm = confusion_matrix(test_labels, test_preds)

    print(f"\n{'='*60}")
    print("RESULTADOS FINALES — Cascada v22")
    print(f"{'='*60}")
    print(f"Test Accuracy:          {test_acc*100:.2f}%")
    print(f"Test Balanced Accuracy: {test_bal*100:.2f}%")

    print("\nConfusion Matrix (7x7):")
    print("       " + "".join(f"Pred{i} " for i in range(7)))
    for i in range(7):
        print(f"Real{i}: " + "".join(f"{cm[i,j]:5d} " for j in range(7)))

    print("\nAccuracy por clase:")
    for i in range(7):
        acc_i = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0.0
        print(f"  Clase {i}: {acc_i*100:.2f}% ({cm[i,i]}/{cm[i].sum()})")

    print(f"\n{'='*60}")
    print("COMPARACION")
    print(f"{'='*60}")
    print(f"Cascada v22 (Binario+6-gestos): {test_acc*100:.2f}% (BalAcc: {test_bal*100:.2f}%)")
    print(f"Cascada antigua (HybridCNNLSTM): 67.45%")
    print(f"v10 CyborgHybrid (S1):           71.32% (BalAcc: 41.92%)")
    print(f"Ensemble 4M:                     72.78%")
    print(f"Objetivo paper:                  79.72%")

    if test_acc > 0.7972:
        print(f"\n*** OBJETIVO ALCANZADO! (+{(test_acc-0.7972)*100:.2f}%) ***")
    elif test_acc > 0.7278:
        print(f"\n*** SUPERA ENSEMBLE 4M! (+{(test_acc-0.7278)*100:.2f}%) ***")
        print(f"  Gap al objetivo: {(0.7972-test_acc)*100:.2f}%")
    elif test_acc > 0.7132:
        print(f"\n*** Supera v10 (+{(test_acc-0.7132)*100:.2f}%) pero bajo ensemble ***")
    elif test_acc > 0.6745:
        print(f"\n*** Supera cascada antigua (+{(test_acc-0.6745)*100:.2f}%) ***")
    else:
        print(f"\nBajo cascada antigua ({test_acc*100:.2f}% < 67.45%)")


if __name__ == "__main__":
    main()
