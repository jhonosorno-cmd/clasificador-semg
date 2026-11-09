"""
Evaluacion rapida: Ensemble v10 + v15 + v17
v10/v15 usan features 48D; v17 usa features 132D.
Se evaluan en el mismo test set pero con DataLoaders distintos.
El ensemble promedia softmax antes del argmax.
"""

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from src.config import DEVICE, BATCH_SIZE, CHECKPOINTS_PATH, SEED
from src.models_cyborg import CyborgHybrid
from src.data_loader import create_data_loaders
from src.train_v17_features import create_data_loaders_v17

torch.manual_seed(SEED)


def load_model(path: str, features_dim: int, device: str) -> CyborgHybrid:
    """Carga checkpoint en modelo con la dimension de features correcta."""
    model = CyborgHybrid(manual_features_dim=features_dim).to(device)
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    val_acc = ckpt.get("val_accuracy", 0.0)
    print(f"  Cargado: {path}  (Val Acc: {val_acc*100:.2f}%, features: {features_dim}D)")
    return model


def get_softmax_probs(model: CyborgHybrid, loader, device: str) -> np.ndarray:
    """Devuelve matriz (N, 7) de probabilidades softmax para todo el loader."""
    all_probs = []
    with torch.no_grad():
        for signals, features, _ in tqdm(loader, desc=f"  Inferencia", leave=False):
            signals = signals.to(device)
            features = features.to(device)
            logits = model(signals, features)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)  # (N, 7)


def get_labels(loader) -> np.ndarray:
    """Extrae todas las etiquetas del loader."""
    all_labels = []
    for _, _, labels in loader:
        all_labels.extend(labels.numpy())
    return np.array(all_labels)


def print_results(label: str, preds: np.ndarray, labels: np.ndarray) -> float:
    """Imprime metricas y devuelve accuracy."""
    acc = np.mean(preds == labels)
    bal_acc = balanced_accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds)

    print(f"\n{'='*60}")
    print(f"RESULTADOS: {label}")
    print(f"{'='*60}")
    print(f"Test Accuracy:          {acc*100:.2f}%")
    print(f"Test Balanced Accuracy: {bal_acc*100:.2f}%")
    print("\nAccuracy por clase:")
    for i in range(7):
        acc_i = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0.0
        print(f"  Clase {i}: {acc_i*100:.2f}% ({cm[i,i]}/{cm[i].sum()})")
    return acc


def main() -> None:
    print("=" * 60)
    print("ENSEMBLE v10 + v15 + v17")
    print("=" * 60)

    device = torch.device(DEVICE)
    print(f"\nDispositivo: {device}")

    # --- Cargar modelos ---
    print("\nCargando modelos...")
    m_v10 = load_model(str(CHECKPOINTS_PATH / "best_model_cyborg.pth"), 48, str(device))
    m_v15 = load_model(str(CHECKPOINTS_PATH / "best_model_v15_augmentation.pth"), 48, str(device))
    m_v17 = load_model(str(CHECKPOINTS_PATH / "best_model_v17_features.pth"), 132, str(device))

    # --- DataLoaders ---
    print("\nCargando datos test S1...")
    _, _, test_loader_48  = create_data_loaders(batch_size=BATCH_SIZE)
    _, _, test_loader_132 = create_data_loaders_v17(batch_size=BATCH_SIZE)

    labels = get_labels(test_loader_48)
    print(f"  Test samples: {len(labels)}")

    # --- Softmax de cada modelo ---
    print("\nCalculando softmax por modelo...")
    print("  v10 (48D):")
    p_v10 = get_softmax_probs(m_v10, test_loader_48, str(device))
    print("  v15 (48D):")
    p_v15 = get_softmax_probs(m_v15, test_loader_48, str(device))
    print("  v17 (132D):")
    p_v17 = get_softmax_probs(m_v17, test_loader_132, str(device))

    # --- Ensemble 2M de referencia: v10+v15 ---
    preds_2m = np.argmax((p_v10 + p_v15) / 2, axis=1)
    acc_2m = print_results("Ensemble v10+v15 (2M) — referencia", preds_2m, labels)

    # --- Ensemble 3M: v10+v15+v17 ---
    preds_3m = np.argmax((p_v10 + p_v15 + p_v17) / 3, axis=1)
    acc_3m = print_results("Ensemble v10+v15+v17 (3M)", preds_3m, labels)

    # --- Ensemble 3M ponderado: mayor peso a v10 y v17 (mas diversos) ---
    # v10: 0.4, v15: 0.2, v17: 0.4  (v15 es similar a v10; v17 es distinto)
    p_weighted = 0.4 * p_v10 + 0.2 * p_v15 + 0.4 * p_v17
    preds_w = np.argmax(p_weighted, axis=1)
    acc_w = print_results("Ensemble ponderado (v10:0.4, v15:0.2, v17:0.4)", preds_w, labels)

    # --- Comparacion final ---
    print(f"\n{'='*60}")
    print("RESUMEN COMPARATIVO")
    print(f"{'='*60}")
    print(f"Ensemble v10+v15 (2M):           {acc_2m*100:.2f}%")
    print(f"Ensemble v10+v15+v17 (3M unif):  {acc_3m*100:.2f}%")
    print(f"Ensemble ponderado (3M pond):     {acc_w*100:.2f}%")
    print(f"Objetivo paper:                   79.72%")

    best = max(acc_2m, acc_3m, acc_w)
    if best > 0.7972:
        print(f"\n*** OBJETIVO ALCANZADO! ({best*100:.2f}%) ***")
    elif best > 0.7241:
        print(f"\n*** NUEVO MAXIMO! ({best*100:.2f}%) ***  (+{(best-0.7241)*100:.2f}% vs 2M)")
    else:
        print(f"\nMejor ensemble: {best*100:.2f}%  (ref 2M: 72.41%)")


if __name__ == "__main__":
    main()
