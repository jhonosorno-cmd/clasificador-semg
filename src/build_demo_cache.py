"""Genera el cache precomputado que consume la demo de sustentacion.

Motivacion:
    demo_app.py construia los DataLoaders completos (train + val + test) dos
    veces — una para features 48D y otra para 132D — lo que implicaba leer y
    preprocesar S1_E1_A1.mat (100 MB) en cada arranque: ~27 s de CPU antes de
    mostrar nada. Este script hace ese trabajo una sola vez y lo persiste en un
    .npz de ~15 MB, de modo que la demo arranca sin depender de data/raw/.

Contenido del cache (data/demo_cache.npz):
    signals      (N, 1, 125, 13) float32  Ventanas EMG + canal RMS (test, S1 rep 6)
    feat48       (N, 48)         float32  Features manuales v10/v15/v18
    feat132      (N, 132)        float32  Features extendidas v17
    labels       (N,)            int64    Etiquetas reales 0-6
    probs_*      (N, 7)          float32  Softmax por modelo (v10, v15, v17, v18)
    probs_mean   (N, 7)          float32  Ensemble 4M uniforme
    val_probs    (M, 7)          float32  Ensemble sobre validacion (rep 5)
    val_labels   (M,)            int64    Etiquetas de validacion
    theta_opt    escalar         float32  Umbral de clase 0 optimo segun validacion
    theta_grid / theta_val_acc / theta_val_bal  Curva de barrido en validacion
    latency_*    escalar         float32  Latencia de inferencia medida (ms)
    class_corr   (7, 7)          float32  Correlacion entre perfiles medios 48D

Ejecutar desde la raiz del proyecto:
    python -m src.build_demo_cache
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

from src.config import CHECKPOINTS_PATH, SEED
from src.data_loader import create_data_loaders
from src.models_cyborg import CyborgHybrid
from src.models_v18 import CyborgHybridAttention
from src.train_v17_features import create_data_loaders_v17

torch.manual_seed(SEED)
np.random.seed(SEED)

CACHE_PATH = Path("data") / "demo_cache.npz"
LIGHT_CKPT_DIR = CHECKPOINTS_PATH / "demo"

# Modelos del Ensemble 4M: (alias, clase, checkpoint, dim de features manuales)
ENSEMBLE_SPEC = [
    ("v10", CyborgHybrid, "best_model_cyborg.pth", 48),
    ("v15", CyborgHybrid, "best_model_v15_augmentation.pth", 48),
    ("v17", CyborgHybrid, "best_model_v17_features.pth", 132),
    ("v18", CyborgHybridAttention, "best_model_v18_attention.pth", 48),
]

LATENCY_WARMUP = 20
LATENCY_RUNS = 200


# ---------------------------------------------------------------------------
# Carga de modelos
# ---------------------------------------------------------------------------

def load_models() -> dict[str, torch.nn.Module]:
    """Carga los 4 modelos del ensemble en modo evaluacion sobre CPU.

    Returns:
        Diccionario alias -> modelo en modo eval.
    """
    models: dict[str, torch.nn.Module] = {}
    for alias, model_cls, ckpt_name, feat_dim in ENSEMBLE_SPEC:
        path = CHECKPOINTS_PATH / ckpt_name
        model = model_cls(manual_features_dim=feat_dim)
        ckpt = torch.load(path, weights_only=False, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models[alias] = model
        val_acc = ckpt.get("val_accuracy", 0.0)
        print(f"  {alias}: {ckpt_name}  (val {val_acc*100:.2f}%, {feat_dim}D)")
    return models


def export_light_checkpoints() -> None:
    """Reexporta los checkpoints del ensemble sin estado del optimizador.

    Los .pth originales guardan optimizer_state_dict, que supone ~2/3 del
    tamano y es inutil para inferencia. El paquete portable usa estas copias.
    """
    LIGHT_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    total_src = total_dst = 0
    for alias, _, ckpt_name, feat_dim in ENSEMBLE_SPEC:
        src = CHECKPOINTS_PATH / ckpt_name
        dst = LIGHT_CKPT_DIR / ckpt_name
        ckpt = torch.load(src, weights_only=False, map_location="cpu")
        torch.save(
            {
                "model_state_dict": ckpt["model_state_dict"],
                "val_accuracy": float(ckpt.get("val_accuracy", 0.0)),
                "val_balanced_accuracy": float(ckpt.get("val_balanced_accuracy", 0.0)),
                "manual_features_dim": feat_dim,
            },
            dst,
        )
        total_src += src.stat().st_size
        total_dst += dst.stat().st_size
        print(f"  {alias}: {src.stat().st_size/1e6:.1f} MB -> {dst.stat().st_size/1e6:.1f} MB")
    print(f"  Total: {total_src/1e6:.1f} MB -> {total_dst/1e6:.1f} MB")


# ---------------------------------------------------------------------------
# Extraccion de datos y probabilidades
# ---------------------------------------------------------------------------

def collect_split(loader_48, loader_132) -> tuple[np.ndarray, ...]:
    """Materializa un split completo en arrays numpy.

    Args:
        loader_48:  DataLoader con features manuales 48D.
        loader_132: DataLoader con features extendidas 132D (mismo orden).

    Returns:
        Tupla (signals, feat48, feat132, labels).
    """
    signals, feat48, feat132, labels = [], [], [], []
    for (sig, f48, lbl), (_, f132, lbl132) in zip(loader_48, loader_132):
        assert np.array_equal(lbl.numpy(), lbl132.numpy()), (
            "Los loaders 48D y 132D devolvieron etiquetas distintas: el orden no coincide"
        )
        signals.append(sig.numpy())
        feat48.append(f48.numpy())
        feat132.append(f132.numpy())
        labels.append(lbl.numpy())
    return (
        np.concatenate(signals),
        np.concatenate(feat48),
        np.concatenate(feat132),
        np.concatenate(labels),
    )


def infer_all(
    models: dict[str, torch.nn.Module],
    signals: np.ndarray,
    feat48: np.ndarray,
    feat132: np.ndarray,
) -> dict[str, np.ndarray]:
    """Calcula el softmax de cada modelo sobre un split completo.

    Args:
        models:  Diccionario alias -> modelo.
        signals: (N, 1, 125, 13) ventanas EMG.
        feat48:  (N, 48) features manuales.
        feat132: (N, 132) features extendidas.

    Returns:
        Diccionario alias -> probabilidades (N, 7).
    """
    sig_t = torch.from_numpy(signals)
    f48_t = torch.from_numpy(feat48)
    f132_t = torch.from_numpy(feat132)

    probs: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for alias, _, _, feat_dim in ENSEMBLE_SPEC:
            feats = f132_t if feat_dim == 132 else f48_t
            logits = models[alias](sig_t, feats)
            probs[alias] = torch.softmax(logits, dim=1).numpy().astype(np.float32)
    return probs


# ---------------------------------------------------------------------------
# Umbral de clase 0
# ---------------------------------------------------------------------------

def predict_with_threshold(probs: np.ndarray, theta: float) -> np.ndarray:
    """Predice aplicando un umbral minimo de confianza a la clase 0 (reposo).

    Si prob[0] > theta se predice reposo; en caso contrario se elige la clase
    activa (1-6) mas probable. Vectorizado sobre todas las muestras.

    Args:
        probs: (N, 7) probabilidades softmax del ensemble.
        theta: Umbral de confianza para la clase de reposo.

    Returns:
        (N,) predicciones enteras 0-6.
    """
    active = np.argmax(probs[:, 1:], axis=1) + 1
    return np.where(probs[:, 0] > theta, 0, active).astype(np.int64)


def sweep_theta(
    val_probs: np.ndarray, val_labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Barre el umbral sobre validacion para elegir theta sin tocar el test set.

    Args:
        val_probs:  (M, 7) probabilidades del ensemble en validacion.
        val_labels: (M,) etiquetas reales de validacion.

    Returns:
        Tupla (grid, accuracy, balanced_accuracy, theta_optimo).
    """
    grid = np.arange(0.10, 1.00, 0.02, dtype=np.float32)
    accs, bals = [], []
    for theta in grid:
        preds = predict_with_threshold(val_probs, float(theta))
        accs.append(np.mean(preds == val_labels))
        bals.append(balanced_accuracy_score(val_labels, preds))
    accs_arr = np.array(accs, dtype=np.float32)
    bals_arr = np.array(bals, dtype=np.float32)
    theta_opt = float(grid[int(np.argmax(accs_arr))])
    return grid, accs_arr, bals_arr, theta_opt


# ---------------------------------------------------------------------------
# Latencia
# ---------------------------------------------------------------------------

def measure_latency(
    models: dict[str, torch.nn.Module],
    signals: np.ndarray,
    feat48: np.ndarray,
    feat132: np.ndarray,
) -> dict[str, float]:
    """Mide la latencia del ensemble para una ventana con warm-up previo.

    Se descartan las primeras `LATENCY_WARMUP` iteraciones porque incluyen la
    inicializacion perezosa de los kernels de PyTorch, que no representa el
    regimen estacionario de una protesis en uso.

    Args:
        models:  Diccionario alias -> modelo.
        signals: (N, 1, 125, 13) ventanas del split de test.
        feat48:  (N, 48) features manuales.
        feat132: (N, 132) features extendidas.

    Returns:
        Diccionario con p50, p95, media y desviacion en milisegundos.
    """
    rng = np.random.default_rng(SEED)
    idx_pool = rng.integers(0, len(signals), size=LATENCY_WARMUP + LATENCY_RUNS)

    tensors = [
        (
            torch.from_numpy(signals[i : i + 1]),
            torch.from_numpy(feat48[i : i + 1]),
            torch.from_numpy(feat132[i : i + 1]),
        )
        for i in idx_pool
    ]

    samples: list[float] = []
    with torch.no_grad():
        for run, (sig_t, f48_t, f132_t) in enumerate(tensors):
            t0 = time.perf_counter()
            acc = None
            for alias, _, _, feat_dim in ENSEMBLE_SPEC:
                feats = f132_t if feat_dim == 132 else f48_t
                p = torch.softmax(models[alias](sig_t, feats), dim=1)
                acc = p if acc is None else acc + p
            _ = acc / len(ENSEMBLE_SPEC)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if run >= LATENCY_WARMUP:
                samples.append(elapsed_ms)

    arr = np.array(samples, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "runs": float(len(arr)),
    }


# ---------------------------------------------------------------------------
# Similitud entre clases (sustituye afirmaciones sin respaldo en la demo)
# ---------------------------------------------------------------------------

def class_profile_correlation(feat48: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Correlacion de Pearson entre los perfiles medios de features de cada clase.

    Cada feature se estandariza sobre todo el conjunto antes de promediar por
    clase, de modo que ninguna domine por escala. El resultado cuantifica
    cuanto se parecen dos gestos en el espacio de features que ve el modelo, y
    permite sustentar con datos propios el analisis de confusiones.

    Args:
        feat48: (N, 48) features manuales del split de test.
        labels: (N,) etiquetas reales.

    Returns:
        (7, 7) matriz de correlacion entre perfiles medios por clase.
    """
    z = (feat48 - feat48.mean(axis=0)) / (feat48.std(axis=0) + 1e-8)
    profiles = np.stack([z[labels == c].mean(axis=0) for c in range(7)])
    return np.corrcoef(profiles).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 64)
    print("CACHE DE LA DEMO — Ensemble 4M sobre S1")
    print("=" * 64)

    print("\n[1/6] Cargando modelos...")
    models = load_models()

    print("\n[2/6] Construyendo splits (esto lee data/raw/S1_E1_A1.mat)...")
    t0 = time.perf_counter()
    _, val_48, test_48 = create_data_loaders(batch_size=512)
    _, val_132, test_132 = create_data_loaders_v17(batch_size=512)
    signals, feat48, feat132, labels = collect_split(test_48, test_132)
    v_signals, v_feat48, v_feat132, v_labels = collect_split(val_48, val_132)
    print(f"  Test: {len(labels)} ventanas | Val: {len(v_labels)} ventanas "
          f"({time.perf_counter()-t0:.1f} s)")

    print("\n[3/6] Inferencia del ensemble...")
    probs = infer_all(models, signals, feat48, feat132)
    probs_mean = np.mean([probs[a] for a in probs], axis=0).astype(np.float32)
    v_probs = infer_all(models, v_signals, v_feat48, v_feat132)
    v_probs_mean = np.mean([v_probs[a] for a in v_probs], axis=0).astype(np.float32)

    preds = np.argmax(probs_mean, axis=1)
    acc = np.mean(preds == labels) * 100
    bal = balanced_accuracy_score(labels, preds) * 100
    print(f"  Ensemble 4M uniforme: {acc:.2f}% acc | {bal:.2f}% balanced acc")
    for alias in probs:
        acc_i = np.mean(np.argmax(probs[alias], axis=1) == labels) * 100
        print(f"    {alias}: {acc_i:.2f}%")

    print("\n[4/6] Barrido del umbral de reposo sobre validacion...")
    grid, val_accs, val_bals, theta_opt = sweep_theta(v_probs_mean, v_labels)
    preds_theta = predict_with_threshold(probs_mean, theta_opt)
    acc_theta = np.mean(preds_theta == labels) * 100
    bal_theta = balanced_accuracy_score(labels, preds_theta) * 100
    print(f"  theta optimo (validacion) = {theta_opt:.2f}")
    print(f"  Test con umbral: {acc_theta:.2f}% acc | {bal_theta:.2f}% balanced acc "
          f"({acc_theta-acc:+.2f}% vs argmax)")

    print(f"\n[5/6] Midiendo latencia ({LATENCY_WARMUP} warm-up + {LATENCY_RUNS} muestras)...")
    lat = measure_latency(models, signals, feat48, feat132)
    print(f"  p50 = {lat['p50']:.1f} ms | p95 = {lat['p95']:.1f} ms | "
          f"media = {lat['mean']:.1f} +/- {lat['std']:.1f} ms")

    print("\n[6/6] Guardando cache y checkpoints ligeros...")
    corr = class_profile_correlation(feat48, labels)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE_PATH,
        signals=signals,
        feat48=feat48,
        feat132=feat132,
        labels=labels,
        probs_v10=probs["v10"],
        probs_v15=probs["v15"],
        probs_v17=probs["v17"],
        probs_v18=probs["v18"],
        probs_mean=probs_mean,
        val_probs=v_probs_mean,
        val_labels=v_labels,
        theta_opt=np.float32(theta_opt),
        theta_grid=grid,
        theta_val_acc=val_accs,
        theta_val_bal=val_bals,
        latency_p50=np.float32(lat["p50"]),
        latency_p95=np.float32(lat["p95"]),
        latency_mean=np.float32(lat["mean"]),
        latency_std=np.float32(lat["std"]),
        latency_runs=np.float32(lat["runs"]),
        class_corr=corr,
        subject=np.int32(1),
        test_rep=np.int32(6),
    )
    print(f"  {CACHE_PATH} ({CACHE_PATH.stat().st_size/1e6:.1f} MB)")

    export_light_checkpoints()
    print("\nListo. La demo ya no necesita data/raw/ para arrancar.")


if __name__ == "__main__":
    main()
