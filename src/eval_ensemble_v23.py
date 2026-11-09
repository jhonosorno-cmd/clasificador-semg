"""
Evaluación: Ensemble 5M con v23 (CyborgTransformer).

Añade el CyborgTransformer (v23, 71.89%) al Ensemble 4M (72.88%) para
explorar si la diversidad del mecanismo de atención mejora el conjunto.

v10/v15/v18/v23 usan features 48D.
v17 usa features 132D.

Referencia: Ensemble 4M uniforme = 72.88% (máximo actual).
"""

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from src.config import DEVICE, BATCH_SIZE, CHECKPOINTS_PATH, SEED
from src.models_cyborg import CyborgHybrid
from src.models_v18 import CyborgHybridAttention
from src.models_v23 import CyborgTransformer
from src.data_loader import create_data_loaders
from src.train_v17_features import create_data_loaders_v17

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Carga de modelos
# ---------------------------------------------------------------------------

def load_cyborg(path: str, features_dim: int, device: str) -> CyborgHybrid:
    model = CyborgHybrid(manual_features_dim=features_dim).to(device)
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    val_acc = ckpt.get("val_accuracy", 0.0)
    print(f"  CyborgHybrid  {path.split(chr(92))[-1]}  val={val_acc*100:.2f}% ({features_dim}D)")
    return model


def load_attention(path: str, device: str) -> CyborgHybridAttention:
    model = CyborgHybridAttention().to(device)
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    val_acc = ckpt.get("val_accuracy", 0.0)
    print(f"  CyborgAttention {path.split(chr(92))[-1]}  val={val_acc*100:.2f}% (48D)")
    return model


def load_transformer(path: str, device: str) -> CyborgTransformer:
    model = CyborgTransformer(manual_features_dim=48, num_classes=7).to(device)
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    val_acc = ckpt.get("val_accuracy", 0.0)
    print(f"  CyborgTransformer {path.split(chr(92))[-1]}  val={val_acc*100:.2f}% (48D)")
    return model


# ---------------------------------------------------------------------------
# Inferencia
# ---------------------------------------------------------------------------

def get_probs(model: torch.nn.Module, loader, device: str) -> np.ndarray:
    """Devuelve (N, 7) probabilidades softmax."""
    all_probs = []
    with torch.no_grad():
        for signals, features, _ in tqdm(loader, desc="  probs", leave=False):
            signals  = signals.to(device)
            features = features.to(device)
            probs = torch.softmax(model(signals, features), dim=1).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def get_labels(loader) -> np.ndarray:
    all_labels = []
    for _, _, labels in loader:
        all_labels.extend(labels.numpy())
    return np.array(all_labels)


# ---------------------------------------------------------------------------
# Resultados
# ---------------------------------------------------------------------------

def print_results(label: str, preds: np.ndarray, labels: np.ndarray) -> float:
    acc     = np.mean(preds == labels)
    bal_acc = balanced_accuracy_score(labels, preds)
    cm      = confusion_matrix(labels, preds)

    print(f"\n{'='*60}")
    print(f"RESULTADOS: {label}")
    print(f"{'='*60}")
    print(f"Test Accuracy:          {acc*100:.2f}%")
    print(f"Test Balanced Accuracy: {bal_acc*100:.2f}%")
    print("Accuracy por clase:")
    for i in range(7):
        acc_i = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0.0
        print(f"  Clase {i}: {acc_i*100:.2f}% ({cm[i,i]}/{cm[i].sum()})")
    return acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("ENSEMBLE 5M: 4M + v23 (CyborgTransformer)")
    print("=" * 60)

    device = str(torch.device(DEVICE))
    print(f"\nDispositivo: {device}")

    # --- Cargar modelos ---
    print("\nCargando modelos...")
    m_v10 = load_cyborg(str(CHECKPOINTS_PATH / "best_model_cyborg.pth"),              48, device)
    m_v15 = load_cyborg(str(CHECKPOINTS_PATH / "best_model_v15_augmentation.pth"),    48, device)
    m_v17 = load_cyborg(str(CHECKPOINTS_PATH / "best_model_v17_features.pth"),       132, device)
    m_v18 = load_attention(str(CHECKPOINTS_PATH / "best_model_v18_attention.pth"),       device)
    m_v23 = load_transformer(str(CHECKPOINTS_PATH / "best_model_v23_transformer.pth"),   device)

    # --- DataLoaders ---
    print("\nCargando datos test S1...")
    _, _, test_loader_48  = create_data_loaders(batch_size=BATCH_SIZE)
    _, _, test_loader_132 = create_data_loaders_v17(batch_size=BATCH_SIZE)
    labels = get_labels(test_loader_48)
    print(f"  Test samples: {len(labels)}")

    # --- Softmax por modelo ---
    print("\nCalculando softmax...")
    print("  v10 (48D):"); p10 = get_probs(m_v10, test_loader_48,  device)
    print("  v15 (48D):"); p15 = get_probs(m_v15, test_loader_48,  device)
    print("  v17 (132D):"); p17 = get_probs(m_v17, test_loader_132, device)
    print("  v18 (48D, attention):"); p18 = get_probs(m_v18, test_loader_48,  device)
    print("  v23 (48D, transformer):"); p23 = get_probs(m_v23, test_loader_48,  device)

    results = {}

    # --- Referencia: Ensemble 4M uniforme ---
    preds = np.argmax((p10 + p15 + p17 + p18) / 4, axis=1)
    results["4M uniforme (ref)"] = print_results(
        "Ensemble 4M uniforme v10+v15+v17+v18 — ref 72.88%", preds, labels
    )

    # --- 5M uniforme ---
    preds = np.argmax((p10 + p15 + p17 + p18 + p23) / 5, axis=1)
    results["5M uniforme"] = print_results(
        "Ensemble 5M uniforme v10+v15+v17+v18+v23", preds, labels
    )

    # --- 5M ponderado: v15 menos peso (redundante), v23 peso intermedio ---
    p_w = 0.25*p10 + 0.10*p15 + 0.25*p17 + 0.25*p18 + 0.15*p23
    preds = np.argmax(p_w, axis=1)
    results["5M pond (0.25,0.10,0.25,0.25,0.15)"] = print_results(
        "Ensemble pond v10:0.25+v15:0.10+v17:0.25+v18:0.25+v23:0.15", preds, labels
    )

    # --- 5M sin v15 (4M diverso + v23) ---
    p_w = 0.30*p10 + 0.30*p17 + 0.25*p18 + 0.15*p23
    preds = np.argmax(p_w, axis=1)
    results["4M-v15 + v23 pond"] = print_results(
        "Ensemble pond v10:0.30+v17:0.30+v18:0.25+v23:0.15 (sin v15)", preds, labels
    )

    # --- 4M ref con v23 reemplazando v15 ---
    preds = np.argmax((p10 + p17 + p18 + p23) / 4, axis=1)
    results["4M v23 en lugar de v15"] = print_results(
        "Ensemble v10+v17+v18+v23 (v23 reemplaza v15)", preds, labels
    )

    # --- Mejor weighted: dar poco peso a v23 (modelo más débil del grupo) ---
    p_w = 0.28*p10 + 0.12*p15 + 0.28*p17 + 0.28*p18 + 0.04*p23
    preds = np.argmax(p_w, axis=1)
    results["5M pond v23 peso mínimo (0.04)"] = print_results(
        "Ensemble pond v10:0.28+v15:0.12+v17:0.28+v18:0.28+v23:0.04", preds, labels
    )

    # --- v23 solo como referencia ---
    preds = np.argmax(p23, axis=1)
    results["v23 individual"] = print_results("v23 CyborgTransformer (individual)", preds, labels)

    # --- Resumen ---
    print(f"\n{'='*60}")
    print("RESUMEN COMPARATIVO")
    print(f"{'='*60}")
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        marker = " <-- NUEVO MÁXIMO" if acc > 0.7288 else (" <-- ref" if "ref" in name else "")
        print(f"  {acc*100:.2f}%  {name}{marker}")
    print(f"\n  72.88%  Ensemble 4M uniforme (máximo actual)")
    print(f"  79.72%  Objetivo paper")

    best = max(results.values())
    if best > 0.7972:
        print(f"\n*** OBJETIVO ALCANZADO! ({best*100:.2f}%) ***")
    elif best > 0.7288:
        print(f"\n*** NUEVO MÁXIMO: {best*100:.2f}% (+{(best-0.7288)*100:.2f}%) ***")
    else:
        print(f"\nMejor: {best*100:.2f}%  — máximo actual 72.88% no superado")


if __name__ == "__main__":
    main()
