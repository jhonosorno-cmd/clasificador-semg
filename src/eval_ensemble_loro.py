"""
Evaluación del Ensemble 4M (v10+v15+v17+v18) por fold LORO-CV.

Carga los checkpoints de los 4 modelos para el fold k y evalúa el ensemble
(promedio uniforme de probabilidades softmax) en el test set correspondiente.

También agrega resultados de todos los folds para reportar mean ± std.

Uso — evaluar un fold específico:
    python -m src.eval_ensemble_loro --fold 1

Uso — agregar resultados de todos los folds ya evaluados:
    python -m src.eval_ensemble_loro --aggregate
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import BATCH_SIZE, CHECKPOINTS_PATH, DEVICE, SEED
from src.models_cyborg import CyborgHybrid
from src.models_v18 import CyborgHybridAttention
from src.train_loro_cv import build_loaders, get_fold_reps
from src.train_v17_features import NinaProExtendedFeatures
from src.data_loader import NinaProDB3Dataset

torch.manual_seed(SEED)

LORO_DIR = CHECKPOINTS_PATH / "loro"

# --------------------------------------------------------------------------- #
# Model loaders
# --------------------------------------------------------------------------- #

def _load_cyborg(ckpt_path: Path, features_dim: int, device: str) -> CyborgHybrid:
    model = CyborgHybrid(manual_features_dim=features_dim).to(device)
    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _load_attention(ckpt_path: Path, device: str) -> CyborgHybridAttention:
    model = CyborgHybridAttention(manual_features_dim=48).to(device)
    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_4m_models(fold: int, device: str) -> list[torch.nn.Module]:
    """
    Carga los 4 modelos del ensemble para el fold dado.

    Args:
        fold: Pliegue k ∈ {1,...,6}.
        device: Dispositivo PyTorch.

    Returns:
        Lista [model_v10, model_v15, model_v17, model_v18].
    """
    models = []
    for name, dim, loader_fn in [
        ("v10", 48,  _load_cyborg),
        ("v15", 48,  _load_cyborg),
        ("v17", 132, _load_cyborg),
    ]:
        ckpt = LORO_DIR / f"{name}_fold_{fold}.pth"
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Checkpoint no encontrado: {ckpt}\n"
                f"Ejecuta: python -m src.train_loro_cv --model {name} --fold {fold}"
            )
        models.append(loader_fn(ckpt, dim, device))
        print(f"  Cargado {name}: {ckpt.name}")

    ckpt_v18 = LORO_DIR / f"v18_fold_{fold}.pth"
    if not ckpt_v18.exists():
        raise FileNotFoundError(
            f"Checkpoint no encontrado: {ckpt_v18}\n"
            f"Ejecuta: python -m src.train_loro_cv --model v18 --fold {fold}"
        )
    models.append(_load_attention(ckpt_v18, device))
    print(f"  Cargado v18: {ckpt_v18.name}")

    return models


# --------------------------------------------------------------------------- #
# Inference helpers
# --------------------------------------------------------------------------- #

@torch.no_grad()
def get_probs(model: torch.nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    """Devuelve matriz (N, 7) de probabilidades softmax."""
    all_probs = []
    for signals, features, _ in tqdm(loader, desc="  Inference", leave=False):
        signals = signals.to(device)
        features = features.to(device)
        logits = model(signals, features)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def get_labels(loader: DataLoader) -> np.ndarray:
    return np.concatenate([labels.numpy() for _, _, labels in loader])


# --------------------------------------------------------------------------- #
# Evaluation per fold
# --------------------------------------------------------------------------- #

def eval_fold(fold: int) -> dict:
    """
    Evalúa el ensemble 4M en el fold especificado.

    Args:
        fold: Pliegue k ∈ {1,...,6}.

    Returns:
        Diccionario con métricas del fold.
    """
    print("=" * 60)
    print(f"ENSEMBLE 4M  |  Fold {fold}/6")
    print("=" * 60)

    LORO_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(DEVICE)

    train_reps, val_reps, test_reps = get_fold_reps(fold)
    print(f"\nSplit — train: {train_reps}  val: {val_reps}  test: {test_reps}")

    # Cargar modelos
    print("\nCargando modelos...")
    models = load_4m_models(fold, device)

    # Test loaders — cada modelo necesita su propio loader (features dim distinta)
    print("\nCargando test data...")
    loaders_48d = [
        DataLoader(
            NinaProDB3Dataset(subjects=[1], reps_to_use=test_reps, mode="test", target_subject=1),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        )
        for _ in range(3)  # v10, v15, v18
    ]
    loader_132d = DataLoader(
        NinaProExtendedFeatures(subjects=[1], reps_to_use=test_reps, mode="test", target_subject=1),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    # Obtener probabilidades
    print("\nCalculando probabilidades...")
    probs_v10 = get_probs(models[0], loaders_48d[0], device)
    probs_v15 = get_probs(models[1], loaders_48d[1], device)
    probs_v17 = get_probs(models[2], loader_132d,    device)
    probs_v18 = get_probs(models[3], loaders_48d[2], device)

    labels = get_labels(loaders_48d[0])

    # Ensemble uniforme
    ensemble_probs = (probs_v10 + probs_v15 + probs_v17 + probs_v18) / 4.0
    preds = np.argmax(ensemble_probs, axis=1)

    acc      = accuracy_score(labels, preds)
    bal_acc  = balanced_accuracy_score(labels, preds)
    prec     = precision_score(labels, preds, average="macro", zero_division=0)
    rec      = recall_score(labels, preds, average="macro", zero_division=0)
    f1       = f1_score(labels, preds, average="macro", zero_division=0)

    class_names = [f"Clase {i}" for i in range(7)]
    report = classification_report(labels, preds, target_names=class_names, digits=4)

    print(f"\n{'='*60}")
    print(f"RESULTADOS  |  Ensemble 4M  |  Fold {fold}")
    print(f"{'='*60}")
    print(f"Accuracy:          {acc*100:.2f}%")
    print(f"Balanced Accuracy: {bal_acc*100:.2f}%")
    print(f"Precision (macro): {prec*100:.2f}%")
    print(f"Recall (macro):    {rec*100:.2f}%")
    print(f"F1-score (macro):  {f1*100:.2f}%")
    print(f"\n{report}")

    metrics = {
        "model": "ensemble_4m",
        "fold": fold,
        "test_rep": test_reps[0],
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
    }
    out_path = LORO_DIR / f"ensemble_4m_fold_{fold}_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMétricas guardadas en: {out_path}")

    return metrics


# --------------------------------------------------------------------------- #
# Aggregation across all folds
# --------------------------------------------------------------------------- #

def aggregate_results() -> None:
    """
    Lee todos los archivos *_metrics.json en LORO_DIR y reporta mean ± std.
    Incluye los 4 modelos comparados: v10, v17, v18, ensemble_4m.
    """
    model_names = ["v10", "v17", "v18", "ensemble_4m"]
    metrics_keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

    print("=" * 70)
    print("AGREGACIÓN LORO-CV  —  mean ± std sobre 6 folds")
    print("=" * 70)

    results: dict[str, dict[str, list]] = {m: {k: [] for k in metrics_keys} for m in model_names}

    for model in model_names:
        for fold in range(1, 7):
            path = LORO_DIR / f"{model}_fold_{fold}_metrics.json"
            if not path.exists():
                print(f"  [FALTA] {path.name}")
                continue
            with open(path) as f:
                data = json.load(f)
            for key in metrics_keys:
                results[model][key].append(data[key])

    # Tabla resumen
    col_w = 14
    header = f"{'Modelo':<14}" + "".join(f"{k:>{col_w}}" for k in metrics_keys) + f"{'Folds':>{col_w}}"
    print(f"\n{header}")
    print("-" * (14 + col_w * len(metrics_keys) + col_w))

    for model in model_names:
        row = f"{model:<14}"
        n_folds = len(results[model]["accuracy"])
        for key in metrics_keys:
            vals = results[model][key]
            if vals:
                mean = np.mean(vals) * 100
                std  = np.std(vals, ddof=1) * 100
                row += f"{mean:>10.2f}±{std:<3.2f}"
            else:
                row += f"{'N/A':>{col_w}}"
        row += f"{n_folds:>{col_w}}"
        print(row)

    # JSON completo
    summary = {}
    for model in model_names:
        summary[model] = {}
        for key in metrics_keys:
            vals = results[model][key]
            if vals:
                summary[model][key] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)),
                    "values": vals,
                }
    summary_path = LORO_DIR / "loro_cv_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResumen completo guardado en: {summary_path}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval Ensemble 4M — LORO-CV")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fold", type=int, choices=list(range(1, 7)),
                       help="Evalúa ensemble en el fold k")
    group.add_argument("--aggregate", action="store_true",
                       help="Agrega resultados de todos los folds")
    args = parser.parse_args()

    if args.aggregate:
        aggregate_results()
    else:
        eval_fold(args.fold)
