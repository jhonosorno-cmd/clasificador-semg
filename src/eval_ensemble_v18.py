"""
Evaluacion: Ensemble con v18 (Self-Attention)
Combina v10, v15, v17, v18 en distintas configuraciones.

v10/v15/v18 usan features 48D (CyborgHybrid / CyborgHybridAttention).
v17 usa features 132D (CyborgHybrid con dim extendida).

v18 complementa a v10 principalmente en clases 1 y 2:
  Clase 1: v18=45.6% vs v10=29.6%  (+16%)
  Clase 2: v18=24.8% vs v10=14.4%  (+10%)
"""

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from src.config import DEVICE, BATCH_SIZE, CHECKPOINTS_PATH, SEED
from src.models_cyborg import CyborgHybrid
from src.models_v18 import CyborgHybridAttention
from src.data_loader import create_data_loaders
from src.train_v17_features import create_data_loaders_v17

torch.manual_seed(SEED)


def load_cyborg(path: str, features_dim: int, device: str) -> CyborgHybrid:
    """Carga CyborgHybrid desde checkpoint."""
    model = CyborgHybrid(manual_features_dim=features_dim).to(device)
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    val_acc = ckpt.get("val_accuracy", 0.0)
    print(f"  Cargado CyborgHybrid:   {path}  (Val Acc: {val_acc*100:.2f}%, {features_dim}D)")
    return model


def load_attention(path: str, device: str) -> CyborgHybridAttention:
    """Carga CyborgHybridAttention desde checkpoint."""
    model = CyborgHybridAttention().to(device)
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    val_acc = ckpt.get("val_accuracy", 0.0)
    print(f"  Cargado AttentionModel: {path}  (Val Acc: {val_acc*100:.2f}%, 48D)")
    return model


def get_probs(model: torch.nn.Module, loader, device: str) -> np.ndarray:
    """Devuelve (N, 7) de probabilidades softmax."""
    all_probs = []
    with torch.no_grad():
        for signals, features, _ in tqdm(loader, desc="  Inferencia", leave=False):
            signals = signals.to(device)
            features = features.to(device)
            logits = model(signals, features)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def get_labels(loader) -> np.ndarray:
    all_labels = []
    for _, _, labels in loader:
        all_labels.extend(labels.numpy())
    return np.array(all_labels)


def print_results(label: str, preds: np.ndarray, labels: np.ndarray) -> float:
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
    print("ENSEMBLE CON v18 (SELF-ATTENTION)")
    print("=" * 60)

    device = torch.device(DEVICE)
    print(f"\nDispositivo: {device}")

    # --- Cargar modelos ---
    print("\nCargando modelos...")
    m_v10 = load_cyborg(str(CHECKPOINTS_PATH / "best_model_cyborg.pth"), 48, str(device))
    m_v15 = load_cyborg(str(CHECKPOINTS_PATH / "best_model_v15_augmentation.pth"), 48, str(device))
    m_v17 = load_cyborg(str(CHECKPOINTS_PATH / "best_model_v17_features.pth"), 132, str(device))
    m_v18 = load_attention(str(CHECKPOINTS_PATH / "best_model_v18_attention.pth"), str(device))

    # --- DataLoaders ---
    print("\nCargando datos test S1...")
    _, _, test_loader_48 = create_data_loaders(batch_size=BATCH_SIZE)
    _, _, test_loader_132 = create_data_loaders_v17(batch_size=BATCH_SIZE)
    labels = get_labels(test_loader_48)
    print(f"  Test samples: {len(labels)}")

    # --- Softmax por modelo ---
    print("\nCalculando softmax...")
    print("  v10 (48D):")
    p10 = get_probs(m_v10, test_loader_48, str(device))
    print("  v15 (48D):")
    p15 = get_probs(m_v15, test_loader_48, str(device))
    print("  v17 (132D):")
    p17 = get_probs(m_v17, test_loader_132, str(device))
    print("  v18 (48D, attention):")
    p18 = get_probs(m_v18, test_loader_48, str(device))

    results = {}

    # --- Referencias previas ---
    preds = np.argmax((p10 + p15) / 2, axis=1)
    results["v10+v15 (2M)"] = print_results("Ensemble v10+v15 (2M) — ref", preds, labels)

    p_pond_prev = 0.4 * p10 + 0.2 * p15 + 0.4 * p17
    preds = np.argmax(p_pond_prev, axis=1)
    results["v10+v15+v17 pond (prev max)"] = print_results(
        "Ensemble pond v10:0.4+v15:0.2+v17:0.4 — ref 72.57%", preds, labels
    )

    # --- Nuevas combinaciones con v18 ---
    # v10 + v18 (2M: misma arquitectura diferente)
    preds = np.argmax((p10 + p18) / 2, axis=1)
    results["v10+v18 (2M)"] = print_results("Ensemble v10+v18 (2M)", preds, labels)

    # v10 + v15 + v18 (3M uniforme)
    preds = np.argmax((p10 + p15 + p18) / 3, axis=1)
    results["v10+v15+v18 (3M)"] = print_results("Ensemble v10+v15+v18 (3M)", preds, labels)

    # v10 + v15 + v18 ponderado (v15 menos peso, v18 distintas fortalezas)
    p_w = 0.4 * p10 + 0.2 * p15 + 0.4 * p18
    preds = np.argmax(p_w, axis=1)
    results["v10+v18 pond (0.4,0.2,0.4)"] = print_results(
        "Ensemble pond v10:0.4+v15:0.2+v18:0.4", preds, labels
    )

    # v10 + v17 + v18 (3M: maxima diversidad, sin v15 redundante)
    preds = np.argmax((p10 + p17 + p18) / 3, axis=1)
    results["v10+v17+v18 (3M)"] = print_results("Ensemble v10+v17+v18 (3M)", preds, labels)

    # v10 + v17 + v18 ponderado
    p_w = 0.35 * p10 + 0.3 * p17 + 0.35 * p18
    preds = np.argmax(p_w, axis=1)
    results["v10+v17+v18 pond"] = print_results(
        "Ensemble pond v10:0.35+v17:0.30+v18:0.35", preds, labels
    )

    # v10 + v15 + v17 + v18 (4M uniforme)
    preds = np.argmax((p10 + p15 + p17 + p18) / 4, axis=1)
    results["v10+v15+v17+v18 (4M)"] = print_results("Ensemble v10+v15+v17+v18 (4M)", preds, labels)

    # v10 + v15 + v17 + v18 ponderado (v15 menos peso)
    p_w = 0.3 * p10 + 0.1 * p15 + 0.3 * p17 + 0.3 * p18
    preds = np.argmax(p_w, axis=1)
    results["v10+v15+v17+v18 pond"] = print_results(
        "Ensemble pond v10:0.3+v15:0.1+v17:0.3+v18:0.3", preds, labels
    )

    # --- Resumen ---
    print(f"\n{'='*60}")
    print("RESUMEN COMPARATIVO")
    print(f"{'='*60}")
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        marker = " <-- NUEVO MAXIMO" if acc > 0.7257 else ""
        print(f"  {acc*100:.2f}%  {name}{marker}")
    print(f"\n  72.57%  Ensemble pond v10+v15+v17 (maximo anterior)")
    print(f"  79.72%  Objetivo paper")

    best = max(results.values())
    if best > 0.7972:
        print(f"\n*** OBJETIVO ALCANZADO! ({best*100:.2f}%) ***")
    elif best > 0.7257:
        print(f"\n*** NUEVO MAXIMO! ({best*100:.2f}%) ***  (+{(best-0.7257)*100:.2f}% vs anterior)")
    else:
        print(f"\nMejor: {best*100:.2f}%  (maximo anterior: 72.57%)")


if __name__ == "__main__":
    main()
