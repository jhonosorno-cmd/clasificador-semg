"""Compara el preprocesamiento actual (fuera de línea) contra uno causal, apto para vivo.

El pipeline del trabajo tiene dos etapas que no pueden existir en operación continua:

  1. `filtfilt` recorre la señal hacia adelante y hacia atrás. Da fase cero, pero para
     producir la muestra n necesita muestras posteriores a n. En vivo no existen.
  2. El z-score usa la media y la desviación de **toda la grabación**. En vivo no hay
     "toda la grabación": hay que fijar los estadísticos en una calibración previa.

Este script mide cuánto cuesta reemplazar las dos por sus equivalentes causales, sobre el
mismo conjunto de prueba (S1, repetición 6) y con los mismos cuatro modelos ya entrenados.
El resultado es el desajuste entre entrenamiento e inferencia: los modelos aprendieron con
el pipeline de la izquierda y aquí se los evalúa con el de la derecha.

NO modifica el repositorio: sustituye `_preprocess_emg` en memoria sobre la clase ya
importada, igual que hizo `run_ventana.py`, y escribe sus resultados en esta carpeta.

Uso:
    python experimento_tiempo_real/eval_causal.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, filtfilt, lfilter
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import src.data_loader as dl  # noqa: E402
from src.config import CHECKPOINTS_PATH, TARGET_FREQUENCY  # noqa: E402
from src.models_cyborg import CyborgHybrid  # noqa: E402
from src.models_v18 import CyborgHybridAttention  # noqa: E402
from src.train_v17_features import NinaProExtendedFeatures  # noqa: E402

SUJETO = 1
REPS_CALIBRACION = [1, 2, 3, 4]   # de dónde salen los estadísticos en el caso causal
REP_PRUEBA = [6]
THETA = 0.36

ENSEMBLE = [
    ("v10", CyborgHybrid, "best_model_cyborg.pth", 48),
    ("v15", CyborgHybrid, "best_model_v15_augmentation.pth", 48),
    ("v17", CyborgHybrid, "best_model_v17_features.pth", 132),
    ("v18", CyborgHybridAttention, "best_model_v18_attention.pth", 48),
]

_ORIGINAL = dl.NinaProDB3Dataset._preprocess_emg


def _coeficientes():
    nyq = TARGET_FREQUENCY / 2.0
    return butter(4, [20.0 / nyq, (nyq - 1.0) / nyq], btype="band")


def preprocesar_offline(self, emg):
    """El pipeline tal como está hoy: filtfilt y z-score sobre toda la grabación."""
    b, a = _coeficientes()
    f = np.zeros_like(emg)
    for ch in range(emg.shape[1]):
        f[:, ch] = filtfilt(b, a, emg[:, ch])
    mu, sd = f.mean(axis=0, keepdims=True), f.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1e-8
    return ((f - mu) / sd).astype(np.float32)


def hacer_preprocesar_causal(stats: dict):
    """Devuelve un preprocesador causal que guarda o reutiliza estadísticos fijos.

    Args:
        stats: Diccionario donde se cachean ``mu`` y ``sd`` de la calibración.
    """
    def preprocesar_causal(self, emg):
        b, a = _coeficientes()
        f = np.zeros_like(emg)
        for ch in range(emg.shape[1]):
            f[:, ch] = lfilter(b, a, emg[:, ch])      # causal: solo mira hacia atrás
        if "mu" not in stats:                          # primera pasada = calibración
            mu, sd = f.mean(axis=0, keepdims=True), f.std(axis=0, keepdims=True)
            sd[sd < 1e-8] = 1e-8
            stats["mu"], stats["sd"] = mu, sd
        return ((f - stats["mu"]) / stats["sd"]).astype(np.float32)
    return preprocesar_causal


def cargar_modelos() -> list:
    modelos = []
    for alias, clase, archivo, dim in ENSEMBLE:
        m = clase(manual_features_dim=dim)
        ck = torch.load(Path(CHECKPOINTS_PATH) / archivo, map_location="cpu", weights_only=False)
        m.load_state_dict(ck["model_state_dict"])
        m.eval()
        modelos.append((alias, m, dim))
    return modelos


def evaluar(ds48, ds132, modelos) -> dict:
    """Corre el ensemble sobre los dos datasets y devuelve métricas con y sin umbral.

    Args:
        ds48:  Dataset base, provee las señales y las 48 características.
        ds132: Mismo split con `NinaProExtendedFeatures`, provee las 132 de v17.
        modelos: Lista (alias, modelo, dimensión de features).

    Raises:
        RuntimeError: si los dos datasets no generaron las mismas ventanas.
    """
    if len(ds48) != len(ds132) or not np.array_equal(ds48.labels, ds132.labels):
        raise RuntimeError("los dos datasets no coinciden en ventanas o etiquetas")

    señales = torch.from_numpy(np.asarray(ds48.samples, dtype=np.float32))
    etiquetas = np.asarray(ds48.labels)
    f48 = np.stack([ds48._extract_manual_features(s[0, :, :12]) for s in ds48.samples])
    f132 = np.stack([ds132._extract_manual_features(s[0, :, :12]) for s in ds132.samples])

    acumulado = None
    with torch.no_grad():
        for alias, modelo, dim in modelos:
            feats = torch.from_numpy(f132 if dim == 132 else f48).float()
            p = torch.softmax(modelo(señales, feats), dim=1)
            acumulado = p if acumulado is None else acumulado + p
    probs = (acumulado / len(modelos)).numpy()

    arg = probs.argmax(axis=1)
    umb = np.where(probs[:, 0] > THETA, 0, probs[:, 1:].argmax(axis=1) + 1)
    salida = {"ventanas": int(len(etiquetas))}
    for nombre, pred in (("argmax", arg), ("umbral", umb)):
        salida[nombre] = {
            "accuracy": float((pred == etiquetas).mean() * 100),
            "balanced_accuracy": float(balanced_accuracy_score(etiquetas, pred) * 100),
            "f1_macro": float(f1_score(etiquetas, pred, average="macro", zero_division=0)),
            "recall_por_clase": [float((pred[etiquetas == c] == c).mean() * 100)
                                 if (etiquetas == c).any() else None for c in range(7)],
        }
    return salida


def main() -> int:
    print("=" * 74)
    print("PREPROCESAMIENTO CAUSAL vs FUERA DE LÍNEA · S1 rep. 6 · Ensemble 4M")
    print("=" * 74)
    t0 = time.time()
    modelos = cargar_modelos()
    print(f"\n{len(modelos)} modelos cargados")

    resultados = {}

    for nombre, fabricar in (("offline", lambda: preprocesar_offline),
                             ("causal", None)):
        print(f"\n[{nombre}] preprocesando...")
        if nombre == "causal":
            stats: dict = {}
            dl.NinaProDB3Dataset._preprocess_emg = hacer_preprocesar_causal(stats)
            # Primera pasada sobre las reps de calibración: fija mu y sd.
            dl.NinaProDB3Dataset(subjects=[SUJETO], reps_to_use=REPS_CALIBRACION,
                                 mode="train", augment=False, target_subject=SUJETO)
            print(f"  estadísticos de calibración fijados desde las reps {REPS_CALIBRACION}")
        else:
            dl.NinaProDB3Dataset._preprocess_emg = fabricar()

        ds48 = dl.NinaProDB3Dataset(subjects=[SUJETO], reps_to_use=REP_PRUEBA,
                                    mode="test", augment=False, target_subject=SUJETO)
        ds132 = NinaProExtendedFeatures(subjects=[SUJETO], reps_to_use=REP_PRUEBA,
                                        mode="test", augment=False, target_subject=SUJETO)
        resultados[nombre] = evaluar(ds48, ds132, modelos)
        print(f"  ventanas: {resultados[nombre]['ventanas']}")

    dl.NinaProDB3Dataset._preprocess_emg = _ORIGINAL

    print("\n" + "=" * 74)
    print(f"{'':22}{'exactitud':>12}{'balanceada':>13}{'F1 macro':>11}")
    for regla in ("argmax", "umbral"):
        print(f"\n  regla: {regla}")
        for nombre in ("offline", "causal"):
            r = resultados[nombre][regla]
            print(f"    {nombre:18}{r['accuracy']:11.2f}%{r['balanced_accuracy']:12.2f}%"
                  f"{r['f1_macro']:11.3f}")
        d_acc = resultados["causal"][regla]["accuracy"] - resultados["offline"][regla]["accuracy"]
        d_bal = (resultados["causal"][regla]["balanced_accuracy"]
                 - resultados["offline"][regla]["balanced_accuracy"])
        print(f"    {'diferencia':18}{d_acc:+11.2f} {'':>1}{d_bal:+12.2f}")

    registro = {
        "descripcion": "Costo de sustituir filtfilt por lfilter y el z-score global por "
                       "estadisticos fijos de calibracion, sobre los modelos ya entrenados.",
        "sujeto": SUJETO, "reps_calibracion": REPS_CALIBRACION, "rep_prueba": REP_PRUEBA,
        "theta": THETA, "timestamp": datetime.now().isoformat(timespec="seconds"),
        "elapsed_min": round((time.time() - t0) / 60, 2),
        "resultados": resultados,
    }
    salida = HERE / "resultados_causal.json"
    salida.write_text(json.dumps(registro, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado en {salida.relative_to(ROOT)}  ·  {registro['elapsed_min']} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
