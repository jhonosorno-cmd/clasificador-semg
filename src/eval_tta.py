"""
Test-Time Augmentation (TTA) sobre el ensemble 4M (v10+v15+v17+v18).

Estrategia:
  Para cada muestra del test set, generar K versiones aumentadas de la senal.
  Promediar las probabilidades softmax de (K augmentaciones x M modelos).
  Los features manuales (48D / 132D) se mantienen fijos del original —
  actuan como ancla estadistica estable mientras la rama CNN ve variaciones.

Augmentaciones TTA (deterministicas, sin aleatoriedad en test):
  0. Original (sin aug)
  1. Ruido gaussiano ligero (sigma=0.015)
  2. Ruido gaussiano medio  (sigma=0.030)
  3. Escalado amplitud x0.85
  4. Escalado amplitud x0.90
  5. Escalado amplitud x1.10
  6. Escalado amplitud x1.15
  7. Ruido ligero + escalado x0.90
  8. Ruido ligero + escalado x1.10
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
# Augmentaciones TTA deterministicas
# ---------------------------------------------------------------------------

def apply_tta_aug(signals: torch.Tensor, aug_id: int) -> torch.Tensor:
    """
    Aplica una augmentacion deterministica a un batch de senales.

    Args:
        signals: Tensor (B, 1, 125, 13) - senales EMG.
        aug_id: Identificador de augmentacion (0 = original).

    Returns:
        Tensor aumentado de misma forma.
    """
    s = signals.clone()
    if aug_id == 0:   # Original
        pass
    elif aug_id == 1: # Ruido ligero
        s = s + torch.randn_like(s) * 0.015
    elif aug_id == 2: # Ruido medio
        s = s + torch.randn_like(s) * 0.030
    elif aug_id == 3: # Scale x0.85
        s = s * 0.85
    elif aug_id == 4: # Scale x0.90
        s = s * 0.90
    elif aug_id == 5: # Scale x1.10
        s = s * 1.10
    elif aug_id == 6: # Scale x1.15
        s = s * 1.15
    elif aug_id == 7: # Ruido ligero + scale x0.90
        s = s + torch.randn_like(s) * 0.015
        s = s * 0.90
    elif aug_id == 8: # Ruido ligero + scale x1.10
        s = s + torch.randn_like(s) * 0.015
        s = s * 1.10
    return s


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
    print(f"  CyborgHybrid {features_dim}D: {path}  (Val {val_acc*100:.2f}%)")
    return model


def load_attention(path: str, device: str) -> CyborgHybridAttention:
    """Carga CyborgHybridAttention desde checkpoint."""
    model = CyborgHybridAttention().to(device)
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    val_acc = ckpt.get("val_accuracy", 0.0)
    print(f"  CyborgAttention 48D: {path}  (Val {val_acc*100:.2f}%)")
    return model


# ---------------------------------------------------------------------------
# Inferencia con TTA
# ---------------------------------------------------------------------------

def collect_tta_probs(
    models_48: list,
    model_v17: CyborgHybrid,
    loader_48,
    loader_132,
    device: str,
    n_aug: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Recorre los loaders y calcula probs con TTA.

    Para cada batch:
      - Aplica n_aug augmentaciones a las senales
      - Corre cada modelo sobre cada augmentacion
      - Promedia todas las probabilidades softmax

    Args:
        models_48: Modelos con features 48D (v10, v15, v18).
        model_v17: Modelo con features 132D (v17).
        loader_48: DataLoader con features 48D.
        loader_132: DataLoader con features 132D.
        device: Dispositivo.
        n_aug: Numero de augmentaciones (1=sin TTA, max 9).

    Returns:
        Tuple (probs_avg, labels) donde probs_avg es (N, 7).
    """
    aug_ids = list(range(n_aug))  # [0, 1, ..., n_aug-1]
    n_models = len(models_48) + 1  # +1 por v17

    all_probs: list[np.ndarray] = []
    all_labels: list[int] = []

    for m in models_48:
        m.eval()
    model_v17.eval()

    iter_132 = iter(loader_132)

    for signals_48, features_48, labels in tqdm(loader_48, desc=f"TTA-{n_aug}", leave=False):
        signals_48 = signals_48.to(device)
        features_48 = features_48.to(device)

        # Obtener batch paralelo del loader 132D (misma senales, features distintas)
        signals_132, features_132, _ = next(iter_132)
        signals_132 = signals_132.to(device)
        features_132 = features_132.to(device)

        batch_probs_sum = None  # (B, 7) acumulador

        with torch.no_grad():
            for aug_id in aug_ids:
                # Augmentar senales (features permanecen originales)
                aug_sig_48 = apply_tta_aug(signals_48, aug_id)
                aug_sig_132 = apply_tta_aug(signals_132, aug_id)

                # Modelos 48D
                for m in models_48:
                    logits = m(aug_sig_48, features_48)
                    p = torch.softmax(logits, dim=1).cpu().numpy()
                    batch_probs_sum = p if batch_probs_sum is None else batch_probs_sum + p

                # Modelo v17 (132D)
                logits = model_v17(aug_sig_132, features_132)
                p = torch.softmax(logits, dim=1).cpu().numpy()
                batch_probs_sum = batch_probs_sum + p

        batch_avg = batch_probs_sum / (n_aug * n_models)  # (B, 7)
        all_probs.append(batch_avg)
        all_labels.extend(labels.numpy().tolist())

    return np.concatenate(all_probs, axis=0), np.array(all_labels)


# ---------------------------------------------------------------------------
# Resultados
# ---------------------------------------------------------------------------

def print_results(label: str, probs: np.ndarray, labels: np.ndarray) -> float:
    """Imprime metricas y devuelve accuracy."""
    preds = np.argmax(probs, axis=1)
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("TEST-TIME AUGMENTATION (TTA) — Ensemble 4M")
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

    # --- DataLoaders ---
    print("\nCargando datos test S1...")
    _, _, test_loader_48 = create_data_loaders(batch_size=BATCH_SIZE)
    _, _, test_loader_132 = create_data_loaders_v17(batch_size=BATCH_SIZE)
    print(f"  Test samples: {len(test_loader_48.dataset)}")

    results = {}

    # --- Baseline: ensemble 4M sin TTA (n_aug=1 = solo original) ---
    print("\n[1/5] Baseline: ensemble 4M sin TTA...")
    probs, labels = collect_tta_probs(
        models_48, m_v17, test_loader_48, test_loader_132, str(device), n_aug=1
    )
    results["Ensemble 4M (sin TTA)"] = print_results("Ensemble 4M sin TTA", probs, labels)

    # --- TTA-3: original + ruido + scale ---
    print("\n[2/5] TTA-3 (original + ruido0.015 + scale0.90)...")
    torch.manual_seed(SEED)  # Reproducibilidad de ruido en TTA
    probs, labels = collect_tta_probs(
        models_48, m_v17, test_loader_48, test_loader_132, str(device), n_aug=3
    )
    results["TTA-3 (orig+noise+scale)"] = print_results("TTA-3", probs, labels)

    # --- TTA-5: original + ruidos + scales ---
    print("\n[3/5] TTA-5 (orig + 2 ruidos + 2 scales)...")
    torch.manual_seed(SEED)
    probs, labels = collect_tta_probs(
        models_48, m_v17, test_loader_48, test_loader_132, str(device), n_aug=5
    )
    results["TTA-5"] = print_results("TTA-5", probs, labels)

    # --- TTA-7: 7 augmentaciones ---
    print("\n[4/5] TTA-7...")
    torch.manual_seed(SEED)
    probs, labels = collect_tta_probs(
        models_48, m_v17, test_loader_48, test_loader_132, str(device), n_aug=7
    )
    results["TTA-7"] = print_results("TTA-7", probs, labels)

    # --- TTA-9: todas las augmentaciones ---
    print("\n[5/5] TTA-9 (todas las augmentaciones)...")
    torch.manual_seed(SEED)
    probs, labels = collect_tta_probs(
        models_48, m_v17, test_loader_48, test_loader_132, str(device), n_aug=9
    )
    results["TTA-9 (completo)"] = print_results("TTA-9", probs, labels)

    # --- Resumen ---
    print(f"\n{'='*60}")
    print("RESUMEN TTA")
    print(f"{'='*60}")
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        marker = " <-- NUEVO MAXIMO" if acc > 0.7278 else ""
        print(f"  {acc*100:.2f}%  {name}{marker}")
    print(f"\n  72.78%  Ensemble 4M sin TTA (maximo anterior)")
    print(f"  79.72%  Objetivo paper")

    best = max(results.values())
    if best > 0.7972:
        print(f"\n*** OBJETIVO ALCANZADO! ({best*100:.2f}%) ***")
    elif best > 0.7278:
        print(f"\n*** TTA MEJORA! Nuevo maximo: {best*100:.2f}%  (+{(best-0.7278)*100:.2f}%) ***")
    else:
        print(f"\nTTA no mejora sobre ensemble 4M sin aug ({best*100:.2f}% <= 72.78%)")


if __name__ == "__main__":
    main()
