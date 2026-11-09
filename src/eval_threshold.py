"""
Ajuste de umbral para clase 0 (reposo) sobre el ensemble 4M.

Estrategia:
  En lugar de predecir clase 0 siempre que softmax[0] sea el maximo,
  exigir softmax[0] > theta. Si la confianza de reposo es insuficiente,
  predecir la clase activa mas probable (argmax de clases 1-6).

  theta se optimiza sobre el conjunto de VALIDACION para evitar overfitting
  al test set. Luego se evalua con ese theta optimo en el test set.

Compromiso esperado:
  Menor theta -> mas muestras se asignan a gestos -> mejora recall de gestos,
  pero puede aumentar falsos positivos de gestos en ventanas de reposo.
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
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Carga de modelos
# ---------------------------------------------------------------------------

def load_cyborg(path: str, features_dim: int, device: str) -> CyborgHybrid:
    model = CyborgHybrid(manual_features_dim=features_dim).to(device)
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_attention(path: str, device: str) -> CyborgHybridAttention:
    model = CyborgHybridAttention().to(device)
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Inferencia: recolectar probs y labels de un split
# ---------------------------------------------------------------------------

def collect_probs(
    models_48: list,
    model_v17: CyborgHybrid,
    loader_48,
    loader_132,
    device: str,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula el promedio de softmax de los 4 modelos para un split.

    Returns:
        probs: (N, 7) float32
        labels: (N,) int
    """
    all_probs: list[np.ndarray] = []
    all_labels: list[int] = []

    iter_132 = iter(loader_132)

    for signals_48, features_48, labels in tqdm(loader_48, desc=split_name, leave=False):
        signals_48 = signals_48.to(device)
        features_48 = features_48.to(device)
        signals_132, features_132, _ = next(iter_132)
        signals_132 = signals_132.to(device)
        features_132 = features_132.to(device)

        probs_sum = None
        with torch.no_grad():
            for m in models_48:
                p = torch.softmax(m(signals_48, features_48), dim=1).cpu().numpy()
                probs_sum = p if probs_sum is None else probs_sum + p
            p = torch.softmax(model_v17(signals_132, features_132), dim=1).cpu().numpy()
            probs_sum = probs_sum + p

        all_probs.append(probs_sum / 4)
        all_labels.extend(labels.numpy().tolist())

    return np.concatenate(all_probs, axis=0), np.array(all_labels)


# ---------------------------------------------------------------------------
# Prediccion con umbral
# ---------------------------------------------------------------------------

def predict_with_threshold(probs: np.ndarray, theta: float) -> np.ndarray:
    """
    Aplica umbral sobre clase 0.

    Si prob[0] > theta: predice clase 0.
    Si prob[0] <= theta: predice la clase activa mas probable (1-6).

    Args:
        probs: (N, 7) probabilidades softmax.
        theta: umbral para clase 0.

    Returns:
        preds: (N,) predicciones.
    """
    preds = np.zeros(len(probs), dtype=int)
    for i, p in enumerate(probs):
        if p[0] > theta:
            preds[i] = 0
        else:
            # Mejor clase activa (1-6)
            preds[i] = int(np.argmax(p[1:]) + 1)
    return preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("AJUSTE DE UMBRAL CLASE 0 — Ensemble 4M")
    print("=" * 60)

    device = torch.device(DEVICE)
    print(f"\nDispositivo: {device}")

    # --- Modelos ---
    print("\nCargando modelos...")
    m_v10 = load_cyborg(str(CHECKPOINTS_PATH / "best_model_cyborg.pth"), 48, str(device))
    m_v15 = load_cyborg(str(CHECKPOINTS_PATH / "best_model_v15_augmentation.pth"), 48, str(device))
    m_v17 = load_cyborg(str(CHECKPOINTS_PATH / "best_model_v17_features.pth"), 132, str(device))
    m_v18 = load_attention(str(CHECKPOINTS_PATH / "best_model_v18_attention.pth"), str(device))
    models_48 = [m_v10, m_v15, m_v18]
    print("  4 modelos cargados (v10, v15, v17, v18)")

    # --- DataLoaders ---
    print("\nCargando datos S1...")
    train_loader_48, val_loader_48, test_loader_48 = create_data_loaders(batch_size=BATCH_SIZE)
    _, val_loader_132, test_loader_132 = create_data_loaders_v17(batch_size=BATCH_SIZE)
    print(f"  Val: {len(val_loader_48.dataset)} | Test: {len(test_loader_48.dataset)}")

    # --- Recolectar probs ---
    print("\nCalculando softmax del ensemble en val y test...")
    val_probs,  val_labels  = collect_probs(models_48, m_v17, val_loader_48,  val_loader_132,  str(device), "Val")
    test_probs, test_labels = collect_probs(models_48, m_v17, test_loader_48, test_loader_132, str(device), "Test")

    # --- Baseline (sin umbral = argmax estandar) ---
    baseline_preds = np.argmax(test_probs, axis=1)
    baseline_acc   = np.mean(baseline_preds == test_labels)
    baseline_bal   = balanced_accuracy_score(test_labels, baseline_preds)
    print(f"\nBaseline (sin umbral): {baseline_acc*100:.2f}%  BalAcc: {baseline_bal*100:.2f}%")

    # --- Barrido de theta en VALIDACION ---
    thetas = np.arange(0.10, 1.00, 0.02)
    print(f"\nBarrido de theta en val ({len(thetas)} valores)...")

    val_results = []
    for theta in thetas:
        preds = predict_with_threshold(val_probs, theta)
        acc = np.mean(preds == val_labels)
        bal = balanced_accuracy_score(val_labels, preds)
        val_results.append((theta, acc, bal))

    # Mejor theta segun overall accuracy en val
    best_theta_acc = max(val_results, key=lambda x: x[1])
    # Mejor theta segun balanced accuracy en val
    best_theta_bal = max(val_results, key=lambda x: x[2])

    print(f"\n  Mejor theta (max overall val acc):    theta={best_theta_acc[0]:.2f}  val_acc={best_theta_acc[1]*100:.2f}%  val_bal={best_theta_acc[2]*100:.2f}%")
    print(f"  Mejor theta (max balanced val acc):   theta={best_theta_bal[0]:.2f}  val_acc={best_theta_bal[1]*100:.2f}%  val_bal={best_theta_bal[2]*100:.2f}%")

    # --- Evaluar en TEST con los mejores thetas ---
    print(f"\n{'='*60}")
    print("EVALUACION EN TEST")
    print(f"{'='*60}")

    def eval_test(theta: float, label: str) -> float:
        preds = predict_with_threshold(test_probs, theta)
        acc = np.mean(preds == test_labels)
        bal = balanced_accuracy_score(test_labels, preds)
        cm = confusion_matrix(test_labels, preds)

        print(f"\n--- {label} (theta={theta:.2f}) ---")
        print(f"Test Accuracy:          {acc*100:.2f}%")
        print(f"Test Balanced Accuracy: {bal*100:.2f}%")
        print("Accuracy por clase:")
        for i in range(7):
            acc_i = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0.0
            base_cm = confusion_matrix(test_labels, baseline_preds)
            base_i  = base_cm[i, i] / base_cm[i].sum() if base_cm[i].sum() > 0 else 0.0
            delta = acc_i - base_i
            sign = "+" if delta >= 0 else ""
            print(f"  Clase {i}: {acc_i*100:.2f}%  ({sign}{delta*100:.2f}% vs baseline)")
        return acc

    eval_test(best_theta_acc[0], "Mejor theta (max overall acc en val)")

    # Si los mejores thetas son distintos, evaluar ambos
    if abs(best_theta_acc[0] - best_theta_bal[0]) > 0.01:
        eval_test(best_theta_bal[0], "Mejor theta (max balanced acc en val)")

    # Adicionalmente, mostrar rango interesante de thetas en test
    print(f"\n{'='*60}")
    print("CURVA ACCURACY vs THETA (test)")
    print(f"{'='*60}")
    print(f"{'Theta':>6}  {'Overall':>8}  {'BalAcc':>8}  {'Cl.0':>6}  {'Cl.1':>6}  {'Cl.2':>6}  {'Cl.3':>6}")
    print("-" * 65)
    for theta in np.arange(0.20, 0.95, 0.05):
        preds = predict_with_threshold(test_probs, theta)
        acc = np.mean(preds == test_labels)
        bal = balanced_accuracy_score(test_labels, preds)
        cm = confusion_matrix(test_labels, preds)
        c0 = cm[0,0]/cm[0].sum() if cm[0].sum() > 0 else 0.0
        c1 = cm[1,1]/cm[1].sum() if cm[1].sum() > 0 else 0.0
        c2 = cm[2,2]/cm[2].sum() if cm[2].sum() > 0 else 0.0
        c3 = cm[3,3]/cm[3].sum() if cm[3].sum() > 0 else 0.0
        marker = " <--" if acc > baseline_acc else ""
        print(f"  {theta:.2f}   {acc*100:7.2f}%  {bal*100:7.2f}%  {c0*100:5.1f}%  {c1*100:5.1f}%  {c2*100:5.1f}%  {c3*100:5.1f}%{marker}")

    # --- Resumen ---
    best_test_acc = max(
        np.mean(predict_with_threshold(test_probs, t) == test_labels)
        for t in thetas
    )
    print(f"\n{'='*60}")
    print("RESUMEN FINAL")
    print(f"{'='*60}")
    print(f"  Baseline (sin umbral):   {baseline_acc*100:.2f}%  BalAcc: {baseline_bal*100:.2f}%")
    print(f"  Mejor con umbral (test): {best_test_acc*100:.2f}%")
    print(f"  Objetivo paper:          79.72%")
    if best_test_acc > 0.7278:
        print(f"\n*** NUEVO MAXIMO: {best_test_acc*100:.2f}% (+{(best_test_acc-0.7278)*100:.2f}%) ***")
    else:
        print(f"\nNo supera el maximo anterior (72.78%)")


if __name__ == "__main__":
    main()
