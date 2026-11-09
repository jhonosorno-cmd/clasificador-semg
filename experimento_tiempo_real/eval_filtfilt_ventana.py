"""Descompone el costo de la causalidad en sus dos ingredientes: filtro y normalización.

El experimento anterior (`eval_causal.py`) cambió dos cosas a la vez —`filtfilt` por
`lfilter`, y z-score global por estadísticos de calibración— y midió el efecto conjunto
(-1,83 pp). Este script separa esas dos causas y agrega una tercera configuración que
surgió de una observación válida: si de todos modos hay que esperar los 250 ms a que la
ventana se complete, `filtfilt` **dentro de esa ventana** no viola la causalidad.

Es cierto que no la viola. Pero no es la misma operación que hace el documento, que filtra
la grabación completa ANTES de cortar las ventanas. Sobre 125 muestras el filtro nunca se
asienta (su respuesta impulsiva necesita 199 muestras para agotar el 99,9 % de su energía)
y `filtfilt` rellena 27 muestras por lado, o sea el 43 % de la ventana. Este script mide
cuánto cuesta eso.

Cuatro ramas, todas sobre los mismos checkpoints ya entrenados (no hay reentrenamiento):

    A  filtfilt(grabación) + z-score global        -> control, debe reproducir 72,78 / 72,88
    B  filtfilt(grabación) + z-score calibración   -> aísla el costo de la normalización
    C  filtfilt(VENTANA)   + z-score calibración   -> la propuesta: causal, pero por ventana
    D  lfilter(grabación)  + z-score calibración   -> causal pleno, debe reproducir 70,95

A y D son puntos de control con valor conocido: si no salen, el arnés está mal y el resto
no se lee. B y C son lo nuevo.

NO modifica el repositorio: sustituye `_preprocess_emg` y `_create_windows` en memoria
sobre la clase ya importada, y escribe sus resultados en esta carpeta.

Uso:
    python experimento_tiempo_real/eval_filtfilt_ventana.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io
import torch
from scipy.signal import butter, filtfilt, lfilter
from sklearn.metrics import balanced_accuracy_score, f1_score

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import src.data_loader as dl  # noqa: E402
from src.config import (CHECKPOINTS_PATH, DOWNSAMPLING_FACTOR,  # noqa: E402
                        RAW_DATA_PATH, TARGET_FREQUENCY)
from src.models_cyborg import CyborgHybrid  # noqa: E402
from src.models_v18 import CyborgHybridAttention  # noqa: E402
from src.train_v17_features import NinaProExtendedFeatures  # noqa: E402

SUJETO = 1
REPS_CALIBRACION = [1, 2, 3, 4]
REP_PRUEBA = [6]
THETA = 0.36

ENSEMBLE = [
    ("v10", CyborgHybrid, "best_model_cyborg.pth", 48),
    ("v15", CyborgHybrid, "best_model_v15_augmentation.pth", 48),
    ("v17", CyborgHybrid, "best_model_v17_features.pth", 132),
    ("v18", CyborgHybridAttention, "best_model_v18_attention.pth", 48),
]

RAMAS = {
    "A_control":        ("filtfilt(grabacion)", "z-score global"),
    "D0_causal_global": ("lfilter(grabacion)",  "z-score global"),
    "B_zscore_calib":   ("filtfilt(grabacion)", "z-score calibracion"),
    "C_filtfilt_vent":  ("filtfilt(VENTANA)",   "z-score calibracion"),
    "D_causal":         ("lfilter(grabacion)",  "z-score calibracion"),
}

# Ramas cuya normalización usa SOLO las repeticiones de calibración.
CON_CALIBRACION = {"B_zscore_calib", "C_filtfilt_vent", "D_causal"}

_PREPROC_ORIG = dl.NinaProDB3Dataset._preprocess_emg
_VENTANAS_ORIG = dl.NinaProDB3Dataset._create_windows

# Estado compartido entre los dos métodos parcheados.
ESTADO: dict = {"rama": None, "stats": {}, "calibrando": False}


def _coeficientes():
    nyq = TARGET_FREQUENCY / 2.0
    return butter(4, [20.0 / nyq, (nyq - 1.0) / nyq], btype="band")


_MASCARA: dict = {}


def mascara_calibracion() -> np.ndarray:
    """Índices de muestra que pertenecen a las repeticiones de calibración.

    `_preprocess_emg` recibe la grabación completa, antes de que el dataset separe las
    repeticiones, así que un `mean(axis=0)` ahí adentro promedia las 6 —incluida la de
    prueba—. Para normalizar de verdad con datos de calibración hay que reconstruir la
    máquina de repeticiones por fuera, desde el `.mat`, con el mismo diezmado.
    """
    if "m" not in _MASCARA:
        mat = scipy.io.loadmat(str(RAW_DATA_PATH / f"S{SUJETO}_E1_A1.mat"))
        reps = mat["rerepetition"][::DOWNSAMPLING_FACTOR].flatten()
        _MASCARA["m"] = np.isin(reps, REPS_CALIBRACION)
        print(f"  mascara de calibracion: {_MASCARA['m'].sum()} de {len(reps)} muestras "
              f"({_MASCARA['m'].mean() * 100:.1f} %) de las reps {REPS_CALIBRACION}")
    return _MASCARA["m"]


def _preprocess_parcheado(self, emg_signal):
    """Filtra la grabación completa, salvo en la rama C, que difiere el filtro a la ventana."""
    rama = ESTADO["rama"]
    if rama == "C_filtfilt_vent":
        return np.asarray(emg_signal, dtype=np.float32)      # crudo: se filtra por ventana

    b, a = _coeficientes()
    pasada = lfilter if rama.startswith("D") else filtfilt
    f = np.zeros_like(emg_signal, dtype=np.float64)
    for ch in range(emg_signal.shape[1]):
        f[:, ch] = pasada(b, a, emg_signal[:, ch])

    if rama in CON_CALIBRACION:
        m = mascara_calibracion()
        if len(m) != len(f):
            raise RuntimeError(f"mascara de {len(m)} muestras contra senal de {len(f)}")
        base = f[m]                                           # solo reps de calibracion
    else:
        base = f                                              # z-score global (transductivo)

    mu, sd = base.mean(axis=0, keepdims=True), base.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1e-8
    return ((f - mu) / sd).astype(np.float32)


def _create_windows_parcheado(self, emg, stim, rep, target_reps):
    """En la rama C filtra cada ventana por separado; en el resto delega al original."""
    if ESTADO["rama"] != "C_filtfilt_vent":
        return _VENTANAS_ORIG(self, emg, stim, rep, target_reps)

    b, a = _coeficientes()
    acumulado: list[np.ndarray] = []
    validas = 0

    for start in range(0, emg.shape[0] - dl.WINDOW_SIZE + 1, dl.STRIDE):
        end = start + dl.WINDOW_SIZE
        etiquetas = stim[start:end].flatten()
        reps_vent = rep[start:end].flatten()
        etiqueta = etiquetas[len(etiquetas) // 2]
        repeticion = reps_vent[len(reps_vent) // 2]

        if repeticion not in target_reps or etiqueta < 0:
            continue
        if not np.all(etiquetas == etiqueta):
            continue

        ventana = np.asarray(emg[start:end, :], dtype=np.float64)
        f = np.zeros_like(ventana)
        for ch in range(ventana.shape[1]):
            f[:, ch] = filtfilt(b, a, ventana[:, ch])         # legal: la ventana ya existe

        if ESTADO["calibrando"]:
            acumulado.append(f)
            continue

        mu, sd = ESTADO["stats"]["mu"], ESTADO["stats"]["sd"]
        norm = ((f - mu) / sd).astype(np.float32)
        rms = np.sqrt(np.mean(norm ** 2, axis=1, keepdims=True))
        self.samples.append(np.concatenate([norm, rms], axis=1)[np.newaxis, :, :])
        self.labels.append(etiqueta)
        validas += 1

    if ESTADO["calibrando"] and acumulado:
        todo = np.concatenate(acumulado, axis=0)               # (N*125, 12)
        mu, sd = todo.mean(axis=0, keepdims=True), todo.std(axis=0, keepdims=True)
        sd[sd < 1e-8] = 1e-8
        ESTADO["stats"]["mu"], ESTADO["stats"]["sd"] = mu, sd
        print(f"  -> estadisticos de {len(acumulado)} ventanas de calibracion fijados")
    else:
        print(f"  -> Generadas {validas} ventanas validas (filtfilt por ventana)")


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
    """Corre el ensemble y devuelve métricas con argmax y con umbral.

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
        for _, modelo, dim in modelos:
            feats = torch.from_numpy(f132 if dim == 132 else f48).float()
            p = torch.softmax(modelo(señales, feats), dim=1)
            acumulado = p if acumulado is None else acumulado + p
    probs = (acumulado / len(modelos)).numpy()

    salida = {"ventanas": int(len(etiquetas))}
    reglas = (("argmax", probs.argmax(axis=1)),
              ("umbral", np.where(probs[:, 0] > THETA, 0, probs[:, 1:].argmax(axis=1) + 1)))
    for nombre, pred in reglas:
        salida[nombre] = {
            "accuracy": float((pred == etiquetas).mean() * 100),
            "balanced_accuracy": float(balanced_accuracy_score(etiquetas, pred) * 100),
            "f1_macro": float(f1_score(etiquetas, pred, average="macro", zero_division=0)),
        }
    return salida


def main() -> int:
    print("=" * 78)
    print("DESCOMPOSICION DEL COSTO CAUSAL · S1 rep. 6 · Ensemble 4M · sin reentrenar")
    print("=" * 78)
    t0 = time.time()

    dl.NinaProDB3Dataset._preprocess_emg = _preprocess_parcheado
    dl.NinaProDB3Dataset._create_windows = _create_windows_parcheado

    modelos = cargar_modelos()
    print(f"\n{len(modelos)} modelos cargados\n")
    resultados = {}

    try:
        for rama in RAMAS:
            print(f"\n{'-' * 78}\n[{rama}] {RAMAS[rama][0]} + {RAMAS[rama][1]}")
            ESTADO["rama"], ESTADO["stats"] = rama, {}

            if rama == "C_filtfilt_vent":     # unica rama que calibra a nivel de ventana
                ESTADO["calibrando"] = True
                dl.NinaProDB3Dataset(subjects=[SUJETO], reps_to_use=REPS_CALIBRACION,
                                     mode="train", augment=False, target_subject=SUJETO)
                ESTADO["calibrando"] = False
                if "mu" not in ESTADO["stats"]:
                    raise RuntimeError(f"{rama}: la calibracion no fijo estadisticos")

            ds48 = dl.NinaProDB3Dataset(subjects=[SUJETO], reps_to_use=REP_PRUEBA,
                                        mode="test", augment=False, target_subject=SUJETO)
            ds132 = NinaProExtendedFeatures(subjects=[SUJETO], reps_to_use=REP_PRUEBA,
                                            mode="test", augment=False, target_subject=SUJETO)
            resultados[rama] = evaluar(ds48, ds132, modelos)
    finally:
        dl.NinaProDB3Dataset._preprocess_emg = _PREPROC_ORIG
        dl.NinaProDB3Dataset._create_windows = _VENTANAS_ORIG

    # ---- controles: A y D tienen valores conocidos del experimento anterior ----
    print("\n" + "=" * 78)
    print("CONTROLES")
    esperado = {("A_control", "argmax"): 72.78, ("A_control", "umbral"): 72.88,
                ("D0_causal_global", "argmax"): 70.95}
    ok = True
    for (rama, regla), ref in esperado.items():
        obt = resultados[rama][regla]["accuracy"]
        bien = abs(obt - ref) < 0.05
        ok &= bien
        print(f"  {'OK   ' if bien else 'FALLA'}  {rama}/{regla}: {obt:.2f}% (esperado {ref:.2f}%)")
    print("  -> el arnes reproduce los valores conocidos" if ok
          else "  -> CUIDADO: los controles no reproducen, no leer el resto")

    # ---- tabla ----
    print("\n" + "=" * 78)
    ancho = max(len(r) for r in RAMAS) + 2
    for regla in ("argmax", "umbral"):
        print(f"\n  regla: {regla}")
        print(f"    {'rama':{ancho}}{'filtro':22}{'normalizacion':22}"
              f"{'exact.':>9}{'balanc.':>10}{'F1':>8}")
        base = resultados["A_control"][regla]["accuracy"]
        for rama, (filtro, norm) in RAMAS.items():
            r = resultados[rama][regla]
            delta = r["accuracy"] - base
            marca = "" if rama == "A_control" else f"  ({delta:+.2f} pp)"
            print(f"    {rama:{ancho}}{filtro:22}{norm:22}"
                  f"{r['accuracy']:8.2f}%{r['balanced_accuracy']:9.2f}%"
                  f"{r['f1_macro']:8.3f}{marca}")

    # ---- descomposición ----
    g = lambda k: resultados[k]["argmax"]["accuracy"]  # noqa: E731
    a, d0 = g("A_control"), g("D0_causal_global")
    b, c, d = g("B_zscore_calib"), g("C_filtfilt_vent"), g("D_causal")
    print("\n" + "=" * 78)
    print("DESCOMPOSICION (argmax, exactitud global)")
    print(f"  costo de la normalizacion por calibracion sola   A -> B   {b - a:+.2f} pp")
    print(f"  costo del filtro causal solo                     A -> D0  {d0 - a:+.2f} pp")
    print(f"  costo de mover filtfilt dentro de la ventana     B -> C   {c - b:+.2f} pp")
    print(f"  ambos cambios juntos (causal pleno)              A -> D   {d - a:+.2f} pp")
    print(f"\n  LA PROPUESTA contra el causal pleno              D -> C   {c - d:+.2f} pp")
    print(f"  LA PROPUESTA contra el control offline           A -> C   {c - a:+.2f} pp")

    registro = {
        "descripcion": "Descomposicion del costo causal: filtro vs normalizacion, mas la "
                       "variante filtfilt aplicado dentro de cada ventana ya capturada.",
        "sujeto": SUJETO, "reps_calibracion": REPS_CALIBRACION, "rep_prueba": REP_PRUEBA,
        "theta": THETA, "ramas": {k: {"filtro": v[0], "normalizacion": v[1]}
                                  for k, v in RAMAS.items()},
        "controles_ok": bool(ok),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "elapsed_min": round((time.time() - t0) / 60, 2),
        "resultados": resultados,
    }
    salida = HERE / "resultados_filtfilt_ventana.json"
    salida.write_text(json.dumps(registro, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado en {salida.relative_to(ROOT)}  ·  {registro['elapsed_min']} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
