"""
Temperature Scaling sobre el Ensemble 4M (v10+v15+v17+v18).

Idea: aprender un escalar T que divide los logits del ensemble antes del softmax.
  softmax(logits / T)

Cuando T > 1 el modelo se vuelve menos confiado (distribución mas suave).
Cuando T < 1 el modelo se vuelve mas confiado (distribución mas peaked).

T se optimiza minimizando la NLL en VALIDACION, luego se evalua en TEST.
Esto evita overfitting al test set.

Referencia: Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.
"""

import torch
import numpy as np
from scipy.optimize import minimize_scalar
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from src.config import DEVICE, BATCH_SIZE, CHECKPOINTS_PATH, SEED
from src.models_cyborg import CyborgHybrid
from src.models_v18 import CyborgHybridAttention
from src.data_loader import create_data_loaders
from src.train_v17_features import create_data_loaders_v17

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Carga de modelos
# ---------------------------------------------------------------------------

def load_cyborg(path: str, features_dim: int, device: str) -> CyborgHybrid:
    """Carga CyborgHybrid desde checkpoint."""
    model = CyborgHybrid(manual_features_dim=features_dim).to(device)
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    val_acc = ckpt.get("val_accuracy", 0.0)
    print(f"  Cargado: {path.split('/')[-1]}  (val={val_acc*100:.2f}%, {features_dim}D)")
    return model


def load_attention(path: str, device: str) -> CyborgHybridAttention:
    """Carga CyborgHybridAttention desde checkpoint."""
    model = CyborgHybridAttention().to(device)
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    val_acc = ckpt.get("val_accuracy", 0.0)
    print(f"  Cargado: {path.split('/')[-1]}  (val={val_acc*100:.2f}%, 48D attention)")
    return model


# ---------------------------------------------------------------------------
# Recolectar logits promediados del ensemble
# ---------------------------------------------------------------------------

def collect_ensemble_logits(
    models_48: list,
    model_v17: CyborgHybrid,
    loader_48,
    loader_132,
    device: str,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula el promedio de logits de los 4 modelos para un split.

    Args:
        models_48: lista de modelos que usan features 48D [v10, v15, v18]
        model_v17: modelo con features 132D
        loader_48: DataLoader con features 48D
        loader_132: DataLoader con features 132D
        device: dispositivo
        split_name: nombre para la barra de progreso

    Returns:
        avg_logits: (N, 7) float32 — promedio de logits crudos
        labels: (N,) int
    """
    all_logits: list[np.ndarray] = []
    all_labels: list[int] = []

    iter_132 = iter(loader_132)

    for signals_48, features_48, labels in tqdm(loader_48, desc=split_name, leave=False):
        signals_48 = signals_48.to(device)
        features_48 = features_48.to(device)
        signals_132, features_132, _ = next(iter_132)
        signals_132 = signals_132.to(device)
        features_132 = features_132.to(device)

        logits_sum = None
        with torch.no_grad():
            for m in models_48:
                l = m(signals_48, features_48).cpu().numpy()
                logits_sum = l if logits_sum is None else logits_sum + l
            l = model_v17(signals_132, features_132).cpu().numpy()
            logits_sum = logits_sum + l

        all_logits.append(logits_sum / 4)  # promedio de los 4 modelos
        all_labels.extend(labels.numpy().tolist())

    return np.concatenate(all_logits, axis=0), np.array(all_labels)


# ---------------------------------------------------------------------------
# Temperature Scaling
# ---------------------------------------------------------------------------

def softmax_np(logits: np.ndarray) -> np.ndarray:
    """Softmax numericamente estable sobre eje 1."""
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def nll_with_temperature(T: float, logits: np.ndarray, labels: np.ndarray) -> float:
    """Negative Log-Likelihood del ensemble escalado a temperatura T."""
    probs = softmax_np(logits / T)
    probs = np.clip(probs, 1e-9, 1.0)  # evitar log(0)
    nll = -np.mean(np.log(probs[np.arange(len(labels)), labels]))
    return float(nll)


def find_optimal_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """Busca T optimo minimizando NLL en validacion."""
    result = minimize_scalar(
        nll_with_temperature,
        bounds=(0.1, 5.0),
        method="bounded",
        args=(val_logits, val_labels),
    )
    return float(result.x)


# ---------------------------------------------------------------------------
# Evaluacion
# ---------------------------------------------------------------------------

def evaluate(logits: np.ndarray, labels: np.ndarray, T: float, label: str) -> float:
    """Evalua accuracy y balanced accuracy con temperatura T."""
    probs = softmax_np(logits / T)
    preds = np.argmax(probs, axis=1)
    acc = np.mean(preds == labels)
    bal = balanced_accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds)

    print(f"\n{'='*60}")
    print(f"{label}  (T={T:.4f})")
    print(f"{'='*60}")
    print(f"Test Accuracy:          {acc*100:.2f}%")
    print(f"Test Balanced Accuracy: {bal*100:.2f}%")
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
    print("TEMPERATURE SCALING — Ensemble 4M (v10+v15+v17+v18)")
    print("=" * 60)

    device = torch.device(DEVICE)
    print(f"\nDispositivo: {device}")

    # --- Cargar modelos ---
    print("\nCargando modelos...")
    m_v10 = load_cyborg(str(CHECKPOINTS_PATH / "best_model_cyborg.pth"), 48, str(device))
    m_v15 = load_cyborg(str(CHECKPOINTS_PATH / "best_model_v15_augmentation.pth"), 48, str(device))
    m_v17 = load_cyborg(str(CHECKPOINTS_PATH / "best_model_v17_features.pth"), 132, str(device))
    m_v18 = load_attention(str(CHECKPOINTS_PATH / "best_model_v18_attention.pth"), str(device))
    models_48 = [m_v10, m_v15, m_v18]

    # --- DataLoaders ---
    print("\nCargando datos S1...")
    _, val_loader_48, test_loader_48 = create_data_loaders(batch_size=BATCH_SIZE)
    _, val_loader_132, test_loader_132 = create_data_loaders_v17(batch_size=BATCH_SIZE)
    print(f"  Val: {len(val_loader_48.dataset)} | Test: {len(test_loader_48.dataset)}")

    # --- Recolectar logits ---
    print("\nRecolectando logits del ensemble...")
    val_logits, val_labels = collect_ensemble_logits(
        models_48, m_v17, val_loader_48, val_loader_132, str(device), "Val"
    )
    test_logits, test_labels = collect_ensemble_logits(
        models_48, m_v17, test_loader_48, test_loader_132, str(device), "Test"
    )

    # --- Baseline (T=1.0) ---
    print()
    baseline_acc = evaluate(test_logits, test_labels, T=1.0, label="Baseline (sin TS, T=1.0)")

    # --- Optimizar T en validacion ---
    print(f"\nOptimizando temperatura en validacion...")
    val_nll_baseline = nll_with_temperature(1.0, val_logits, val_labels)
    T_opt = find_optimal_temperature(val_logits, val_labels)
    val_nll_opt = nll_with_temperature(T_opt, val_logits, val_labels)
    print(f"  T optimo:  {T_opt:.4f}")
    print(f"  NLL val antes: {val_nll_baseline:.6f}")
    print(f"  NLL val despues: {val_nll_opt:.6f}  (mejora: {val_nll_baseline - val_nll_opt:.6f})")

    # Accuracy en val con T optimo
    val_probs_T = softmax_np(val_logits / T_opt)
    val_preds_T = np.argmax(val_probs_T, axis=1)
    val_acc_baseline = np.mean(np.argmax(softmax_np(val_logits), axis=1) == val_labels)
    val_acc_T = np.mean(val_preds_T == val_labels)
    print(f"  Val acc baseline: {val_acc_baseline*100:.2f}%")
    print(f"  Val acc con T:    {val_acc_T*100:.2f}%")

    # --- Evaluar T optimo en test ---
    ts_acc = evaluate(test_logits, test_labels, T=T_opt, label=f"Temperature Scaling optimo (val-tuned)")

    # --- Barrido de T en test (para analisis) ---
    print(f"\n{'='*60}")
    print("BARRIDO DE T EN TEST (referencia)")
    print(f"{'='*60}")
    print(f"{'T':>6}  {'Acc':>8}  {'BalAcc':>8}  {'NLL':>8}")
    print("-" * 40)
    best_ts_acc = 0.0
    for T in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0]:
        probs = softmax_np(test_logits / T)
        preds = np.argmax(probs, axis=1)
        acc = np.mean(preds == test_labels)
        bal = balanced_accuracy_score(test_labels, preds)
        nll = nll_with_temperature(T, test_logits, test_labels)
        marker = " <-- optimo" if abs(T - T_opt) < 0.15 else ""
        print(f"  {T:5.2f}   {acc*100:7.2f}%  {bal*100:7.2f}%  {nll:8.5f}{marker}")
        best_ts_acc = max(best_ts_acc, acc)

    # --- Resumen ---
    print(f"\n{'='*60}")
    print("RESUMEN FINAL")
    print(f"{'='*60}")
    print(f"  Baseline (T=1.0):        {baseline_acc*100:.2f}%")
    print(f"  Temperature Scaling:     {ts_acc*100:.2f}%  (T={T_opt:.4f}, val-tuned)")
    delta = ts_acc - baseline_acc
    sign = "+" if delta >= 0 else ""
    print(f"  Delta:                   {sign}{delta*100:.2f}%")
    print(f"  Maximo previo:           72.88%")
    print(f"  Objetivo paper:          79.72%")

    if ts_acc > 0.7288:
        print(f"\n*** NUEVO MAXIMO: {ts_acc*100:.2f}% (+{(ts_acc-0.7288)*100:.2f}%) ***")
    elif ts_acc > 0.7278:
        print(f"\n  Mejora marginal: {ts_acc*100:.2f}%")
    else:
        print(f"\n  Sin mejora sobre baseline con Temperature Scaling.")


if __name__ == "__main__":
    main()
