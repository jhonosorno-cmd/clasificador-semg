"""Seleccion de configuracion de ensemble sobre VALIDACION (procedimiento correcto).

Replica el torneo de configuraciones de eval_ensemble_v18.py, pero con el
protocolo metodologicamente correcto:

  1. Evaluar todas las configuraciones candidatas sobre la repeticion 5 (val).
  2. Elegir la ganadora por exactitud de validacion.
  3. Evaluar UNICAMENTE la ganadora sobre la repeticion 6 (test), una sola vez.
  4. Adicionalmente, optimizar el umbral theta de reposo sobre validacion para
     la ganadora y reportar su resultado en test (pipeline completo honesto).

Proposito: verificar a posteriori si la seleccion original (hecha sobre test)
coincide con la que habria producido la seleccion sobre validacion. Documentado
como respuesta a la retroalimentacion de jurados (punto 3).

Uso:
    python -m src.eval_ensemble_selection_val
"""

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import BATCH_SIZE, CHECKPOINTS_PATH, DEVICE, SEED
from src.data_loader import create_data_loaders
from src.models_cyborg import CyborgHybrid
from src.models_v18 import CyborgHybridAttention
from src.train_v17_features import create_data_loaders_v17

torch.manual_seed(SEED)
np.random.seed(SEED)

# Configuraciones candidatas: nombre -> pesos sobre [v10, v15, v17, v18].
# Mismas 9 configuraciones comparadas originalmente en eval_ensemble_v17/v18.
CANDIDATES: dict[str, tuple[float, float, float, float]] = {
    "v10+v15 (2M)":                      (0.50, 0.50, 0.00, 0.00),
    "v10+v15+v17 pond (0.4,0.2,0.4)":    (0.40, 0.20, 0.40, 0.00),
    "v10+v18 (2M)":                      (0.50, 0.00, 0.00, 0.50),
    "v10+v15+v18 (3M)":                  (1/3,  1/3,  0.00, 1/3),
    "v10+v15+v18 pond (0.4,0.2,0.4)":    (0.40, 0.20, 0.00, 0.40),
    "v10+v17+v18 (3M)":                  (1/3,  0.00, 1/3,  1/3),
    "v10+v17+v18 pond (0.35,0.3,0.35)":  (0.35, 0.00, 0.30, 0.35),
    "v10+v15+v17+v18 (4M uniforme)":     (0.25, 0.25, 0.25, 0.25),
    "v10+v15+v17+v18 pond (0.3,0.1,0.3,0.3)": (0.30, 0.10, 0.30, 0.30),
}


def load_cyborg(path: str, features_dim: int, device: torch.device) -> CyborgHybrid:
    """Carga un CyborgHybrid desde checkpoint en modo evaluacion.

    Args:
        path: Ruta al checkpoint .pth.
        features_dim: Dimension del vector de caracteristicas manuales.
        device: Dispositivo PyTorch.

    Returns:
        Modelo cargado en modo eval.
    """
    model = CyborgHybrid(manual_features_dim=features_dim).to(device)
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_attention(path: str, device: torch.device) -> CyborgHybridAttention:
    """Carga un CyborgHybridAttention desde checkpoint en modo evaluacion.

    Args:
        path: Ruta al checkpoint .pth.
        device: Dispositivo PyTorch.

    Returns:
        Modelo cargado en modo eval.
    """
    model = CyborgHybridAttention().to(device)
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def get_probs(model: torch.nn.Module, loader: DataLoader, device: torch.device, desc: str) -> np.ndarray:
    """Devuelve la matriz (N, 7) de probabilidades softmax del modelo.

    Args:
        model: Modelo en modo eval.
        loader: DataLoader del split a inferir (shuffle=False).
        device: Dispositivo PyTorch.
        desc: Etiqueta para la barra de progreso.

    Returns:
        Matriz (N, 7) float con probabilidades softmax.
    """
    all_probs: list[np.ndarray] = []
    for signals, features, _ in tqdm(loader, desc=desc, leave=False):
        logits = model(signals.to(device), features.to(device))
        all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def get_labels(loader: DataLoader) -> np.ndarray:
    """Extrae las etiquetas de un DataLoader en orden.

    Args:
        loader: DataLoader del split (shuffle=False).

    Returns:
        Vector (N,) de etiquetas enteras.
    """
    return np.concatenate([labels.numpy() for _, _, labels in loader])


def combine(probs: list[np.ndarray], weights: tuple[float, float, float, float]) -> np.ndarray:
    """Combina probabilidades de los 4 modelos con pesos dados.

    Args:
        probs: Lista [p_v10, p_v15, p_v17, p_v18], cada una (N, 7).
        weights: Pesos (w10, w15, w17, w18); los ceros excluyen el modelo.

    Returns:
        Matriz (N, 7) de probabilidades combinadas (normalizadas por suma de pesos).
    """
    total = sum(weights)
    mix = sum(w * p for w, p in zip(weights, probs) if w > 0)
    return mix / total


def predict_with_threshold(probs: np.ndarray, theta: float) -> np.ndarray:
    """Aplica el umbral de reposo con la regla original de eval_threshold.py.

    Predice clase 0 si y solo si prob[0] > theta; en caso contrario predice
    la clase activa (1-6) mas probable. Identica a eval_threshold.py para que
    los resultados sean comparables con el 72,88% reportado.

    Args:
        probs: Matriz (N, 7) de probabilidades.
        theta: Umbral para la clase 0.

    Returns:
        Vector (N,) de predicciones.
    """
    active = np.argmax(probs[:, 1:], axis=1) + 1
    return np.where(probs[:, 0] > theta, 0, active)


def main() -> None:
    """Ejecuta la seleccion sobre validacion y evalua la ganadora en test."""
    print("=" * 72)
    print("SELECCION DE ENSEMBLE SOBRE VALIDACION (procedimiento correcto)")
    print("=" * 72)

    device = torch.device(DEVICE)
    print(f"\nDispositivo: {device}")

    print("\nCargando modelos...")
    m_v10 = load_cyborg(str(CHECKPOINTS_PATH / "best_model_cyborg.pth"), 48, device)
    m_v15 = load_cyborg(str(CHECKPOINTS_PATH / "best_model_v15_augmentation.pth"), 48, device)
    m_v17 = load_cyborg(str(CHECKPOINTS_PATH / "best_model_v17_features.pth"), 132, device)
    m_v18 = load_attention(str(CHECKPOINTS_PATH / "best_model_v18_attention.pth"), device)

    print("\nCargando datos S1 (val=rep 5, test=rep 6)...")
    _, val_loader_48, test_loader_48 = create_data_loaders(batch_size=BATCH_SIZE)
    _, val_loader_132, test_loader_132 = create_data_loaders_v17(batch_size=BATCH_SIZE)
    val_labels = get_labels(val_loader_48)
    test_labels = get_labels(test_loader_48)
    print(f"  Val: {len(val_labels)} ventanas | Test: {len(test_labels)} ventanas")

    print("\nCalculando softmax por modelo (val y test)...")
    val_probs = [
        get_probs(m_v10, val_loader_48, device, "val v10"),
        get_probs(m_v15, val_loader_48, device, "val v15"),
        get_probs(m_v17, val_loader_132, device, "val v17"),
        get_probs(m_v18, val_loader_48, device, "val v18"),
    ]
    test_probs = [
        get_probs(m_v10, test_loader_48, device, "test v10"),
        get_probs(m_v15, test_loader_48, device, "test v15"),
        get_probs(m_v17, test_loader_132, device, "test v17"),
        get_probs(m_v18, test_loader_48, device, "test v18"),
    ]

    # ------------------------------------------------------------------ #
    # Paso 1-2: torneo sobre VALIDACION
    # ------------------------------------------------------------------ #
    print(f"\n{'='*72}")
    print("PASO 1-2: comparacion de candidatas sobre VALIDACION (rep 5)")
    print(f"{'='*72}")
    print(f"{'Configuracion':<42} {'Val Acc':>9} {'Val BalAcc':>11}")
    print("-" * 64)

    val_scores: dict[str, float] = {}
    for name, weights in CANDIDATES.items():
        mix = combine(val_probs, weights)
        preds = np.argmax(mix, axis=1)
        acc = float(np.mean(preds == val_labels))
        bal = balanced_accuracy_score(val_labels, preds)
        val_scores[name] = acc
        print(f"{name:<42} {acc*100:8.2f}% {bal*100:10.2f}%")

    winner = max(val_scores, key=lambda k: val_scores[k])
    ties = [n for n, a in val_scores.items() if abs(a - val_scores[winner]) < 1e-9]
    print(f"\nGanadora en validacion: {winner}  (val acc {val_scores[winner]*100:.2f}%)")
    if len(ties) > 1:
        print(f"ATENCION - empate exacto entre: {ties}")

    # ------------------------------------------------------------------ #
    # Paso 3: evaluar SOLO la ganadora en TEST (sin umbral)
    # ------------------------------------------------------------------ #
    w = CANDIDATES[winner]
    test_mix = combine(test_probs, w)
    test_preds = np.argmax(test_mix, axis=1)
    test_acc = float(np.mean(test_preds == test_labels))
    test_bal = balanced_accuracy_score(test_labels, test_preds)

    print(f"\n{'='*72}")
    print("PASO 3: evaluacion unica de la ganadora en TEST (rep 6, sin umbral)")
    print(f"{'='*72}")
    print(f"  {winner}")
    print(f"  Test Accuracy:          {test_acc*100:.2f}%")
    print(f"  Test Balanced Accuracy: {test_bal*100:.2f}%")

    # ------------------------------------------------------------------ #
    # Paso 4: optimizar theta sobre VALIDACION para la ganadora -> test
    # ------------------------------------------------------------------ #
    val_mix = combine(val_probs, w)
    best_theta, best_val_acc = 0.0, float(np.mean(np.argmax(val_mix, axis=1) == val_labels))
    for theta in np.arange(0.10, 1.00, 0.02):
        acc = float(np.mean(predict_with_threshold(val_mix, theta) == val_labels))
        if acc > best_val_acc:
            best_theta, best_val_acc = float(theta), acc

    preds_theta = predict_with_threshold(test_mix, best_theta)
    acc_theta = float(np.mean(preds_theta == test_labels))
    bal_theta = balanced_accuracy_score(test_labels, preds_theta)

    print(f"\n{'='*72}")
    print("PASO 4: umbral optimizado en VALIDACION para la ganadora -> TEST")
    print(f"{'='*72}")
    print(f"  Theta optimo (val): {best_theta:.2f}  (val acc {best_val_acc*100:.2f}%)")
    print(f"  Test Accuracy con umbral:          {acc_theta*100:.2f}%")
    print(f"  Test Balanced Accuracy con umbral: {bal_theta*100:.2f}%")

    # ------------------------------------------------------------------ #
    # Resumen comparativo contra el resultado reportado en la tesis
    # ------------------------------------------------------------------ #
    print(f"\n{'='*72}")
    print("RESUMEN")
    print(f"{'='*72}")
    print(f"  Seleccion original (sobre test):  4M uniforme -> 72.78% / 72.88% con theta=0.36")
    print(f"  Seleccion correcta (sobre val):   {winner}")
    print(f"    -> test sin umbral: {test_acc*100:.2f}%")
    print(f"    -> test con umbral theta={best_theta:.2f}: {acc_theta*100:.2f}%")
    same = winner == "v10+v15+v17+v18 (4M uniforme)"
    print(f"\n  La seleccion sobre validacion {'COINCIDE' if same else 'NO COINCIDE'} "
          f"con la configuracion reportada en la tesis.")


if __name__ == "__main__":
    main()
