"""Mide la latencia de cada etapa del sistema sobre una ventana de 250 ms.

Reemplaza una medición anterior cuyo script se perdió y de la cual no quedó constancia de
si el filtro se cronometró por ventana o sobre la grabación completa. La distinción importa:
el anexo afirma que el sistema procesa una ventana en N ms, y eso solo es comprobable si el
filtro se midió sobre una ventana.

Cronometra las dos configuraciones que el anexo considera implementables en vivo:

    fase cero por ventana   filtfilt sobre las 125 muestras ya capturadas (recalcula todo)
    causal de una pasada    lfilter con estado arrastrado entre llamadas (solo lo nuevo)

Protocolo: 20 corridas de calentamiento descartadas, 200 cronometradas, se reporta mediana
y percentil 95. En tiempo real la cola importa más que la mediana, así que ambos se reportan.

NO modifica el repositorio.

Uso:
    python experimento_tiempo_real/medir_latencia.py
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
from scipy.signal import butter, filtfilt, lfilter, lfilter_zi

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import src.data_loader as dl  # noqa: E402
from src.config import (CHECKPOINTS_PATH, DOWNSAMPLING_FACTOR,  # noqa: E402
                        RAW_DATA_PATH, TARGET_FREQUENCY)
from src.models_cyborg import CyborgHybrid  # noqa: E402
from src.models_v18 import CyborgHybridAttention  # noqa: E402

CALENTAMIENTO, CORRIDAS, REPETICIONES = 20, 200, 5
VENTANA, STRIDE = 125, 25

ENSEMBLE = [
    ("v10", CyborgHybrid, "best_model_cyborg.pth", 48),
    ("v15", CyborgHybrid, "best_model_v15_augmentation.pth", 48),
    ("v17", CyborgHybrid, "best_model_v17_features.pth", 132),
    ("v18", CyborgHybridAttention, "best_model_v18_attention.pth", 48),
]


def cronometrar(fn, *args) -> dict:
    """Cronometra `fn` en varias tandas independientes y resume la muestra agrupada.

    Una sola tanda no basta. La inferencia sobre CPU en una máquina de propósito general
    compite con el planificador del sistema operativo, y repetir el mismo benchmark ocho
    veces produjo medianas entre 16,3 y 23,9 ms: un 43 % de dispersión. Reportar la mediana
    de una corrida sería reportar una tirada, no una medición.

    Se agrupan todas las corridas de todas las tandas para la mediana y el percentil 95, y
    se conserva el rango de las medianas por tanda como medida explícita de esa dispersión.
    El mínimo se reporta aparte: el ruido de planificación solo puede sumar tiempo, de modo
    que el mínimo es la estimación menos contaminada del costo intrínseco de la operación.
    """
    medianas, todas = [], []
    for _ in range(REPETICIONES):
        for _ in range(CALENTAMIENTO):
            fn(*args)
        tanda = []
        for _ in range(CORRIDAS):
            t = time.perf_counter()
            fn(*args)
            tanda.append((time.perf_counter() - t) * 1000.0)
        medianas.append(float(np.median(tanda)))
        todas.extend(tanda)
    return {
        "mediana": float(np.median(todas)),
        "p95": float(np.percentile(todas, 95)),
        "minimo": float(np.min(todas)),
        "mediana_min": float(np.min(medianas)),
        "mediana_max": float(np.max(medianas)),
    }


def main() -> int:
    nyq = TARGET_FREQUENCY / 2.0
    b, a = butter(4, [20.0 / nyq, (nyq - 1.0) / nyq], btype="band")

    mat = scipy.io.loadmat(str(RAW_DATA_PATH / "S1_E1_A1.mat"))
    crudo = mat["emg"]                                   # (N, 12) a 2000 Hz
    bloque_crudo = crudo[:VENTANA * DOWNSAMPLING_FACTOR]  # lo que llega en 250 ms
    nuevo_crudo = crudo[:STRIDE * DOWNSAMPLING_FACTOR]    # lo que llega en un stride

    ventana = crudo[::DOWNSAMPLING_FACTOR][:VENTANA].astype(np.float64)   # (125, 12)
    mu = ventana.mean(axis=0, keepdims=True)
    sd = ventana.std(axis=0, keepdims=True); sd[sd < 1e-8] = 1e-8
    estado = [lfilter_zi(b, a) * 0.0 for _ in range(12)]

    def diezmar(x):
        return x[::DOWNSAMPLING_FACTOR]

    def filtro_fase_cero(x):
        f = np.zeros_like(x)
        for ch in range(x.shape[1]):
            f[:, ch] = filtfilt(b, a, x[:, ch])
        return f

    def filtro_causal_incremental(x):
        f = np.zeros_like(x)
        for ch in range(x.shape[1]):
            f[:, ch], estado[ch] = lfilter(b, a, x[:, ch], zi=estado[ch])
        return f

    def normalizar(x):
        return ((x - mu) / sd).astype(np.float32)

    def canal_rms(x):
        return np.concatenate([x, np.sqrt(np.mean(x ** 2, axis=1, keepdims=True))], axis=1)

    ds = dl.NinaProDB3Dataset.__new__(dl.NinaProDB3Dataset)
    norm = normalizar(filtro_fase_cero(ventana))
    con_rms = canal_rms(norm)

    def features_48(x):
        return ds._extract_manual_features(x)

    modelos = []
    for _, Clase, archivo, dim in ENSEMBLE:
        m = Clase(manual_features_dim=dim)
        ck = torch.load(Path(CHECKPOINTS_PATH) / archivo, map_location="cpu",
                        weights_only=False)
        m.load_state_dict(ck["model_state_dict"]); m.eval()
        modelos.append((m, dim))

    señal_t = torch.from_numpy(con_rms[np.newaxis, np.newaxis, :, :].astype(np.float32))
    f48_t = torch.from_numpy(features_48(norm)[np.newaxis, :]).float()
    f132_t = torch.zeros(1, 132)

    def inferir():
        acum = None
        with torch.no_grad():
            for m, dim in modelos:
                p = torch.softmax(m(señal_t, f132_t if dim == 132 else f48_t), dim=1)
                acum = p if acum is None else acum + p
        return (acum / len(modelos)).numpy()

    etapas = [
        ("Diezmado x4 (ventana)",            cronometrar(diezmar, bloque_crudo)),
        ("Filtro fase cero POR VENTANA",     cronometrar(filtro_fase_cero, ventana)),
        ("Filtro causal incremental",        cronometrar(filtro_causal_incremental,
                                                         diezmar(nuevo_crudo).astype(np.float64))),
        ("Normalizacion Z-score",            cronometrar(normalizar, ventana)),
        ("Canal RMS agregado",               cronometrar(canal_rms, norm)),
        ("Extraccion de 48 caracteristicas", cronometrar(features_48, norm)),
        ("Inferencia del Ensemble 4M",       cronometrar(inferir)),
    ]

    print("=" * 84)
    print("LATENCIA POR ETAPA · ventana de 250 ms · CPU · lote unitario")
    print(f"{REPETICIONES} tandas independientes x {CORRIDAS} corridas "
          f"(+{CALENTAMIENTO} de calentamiento cada una)")
    print("=" * 84)
    print(f"\n{'Etapa':38}{'Mediana':>10}{'p95':>9}{'Minimo':>9}{'rango medianas':>18}")
    for nombre, r in etapas:
        print(f"  {nombre:36}{r['mediana']:10.3f}{r['p95']:9.3f}{r['minimo']:9.3f}"
              f"{r['mediana_min']:9.2f}–{r['mediana_max']:<8.2f}")

    d = dict(etapas)
    PRE = ("Diezmado x4 (ventana)", "Filtro fase cero POR VENTANA", "Normalizacion Z-score",
           "Canal RMS agregado", "Extraccion de 48 caracteristicas")
    pre = {k: sum(d[e][k] for e in PRE) for k in ("mediana", "p95", "minimo")}
    inf = d["Inferencia del Ensemble 4M"]
    tot = {k: pre[k] + inf[k] for k in pre}

    print(f"\n  {'Subtotal preprocesamiento':36}{pre['mediana']:10.3f}{pre['p95']:9.3f}"
          f"{pre['minimo']:9.3f}")
    print(f"  {'TOTAL':36}{tot['mediana']:10.3f}{tot['p95']:9.3f}{tot['minimo']:9.3f}")
    print(f"\n  presupuesto por stride: 50 ms")
    for k in ("minimo", "mediana", "p95"):
        print(f"    {k:8}{tot[k]:7.2f} ms  ->  {tot[k] / 50 * 100:5.1f} % del presupuesto")
    print(f"\n  la inferencia es el {inf['mediana'] / tot['mediana'] * 100:.1f} % del total")
    print(f"  dispersion de la inferencia entre tandas: "
          f"{inf['mediana_min']:.2f}–{inf['mediana_max']:.2f} ms "
          f"({(inf['mediana_max'] - inf['mediana_min']) / inf['mediana'] * 100:.0f} %)")

    registro = {
        "descripcion": "Latencia por etapa sobre UNA ventana de 250 ms, filtro cronometrado "
                       "por ventana (configuracion implementable en vivo). Varias tandas "
                       "independientes: la mediana de una sola tanda no es reproducible.",
        "calentamiento": CALENTAMIENTO, "corridas_por_tanda": CORRIDAS,
        "tandas": REPETICIONES,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "etapas_ms": {n: r for n, r in etapas},
        "subtotal_preprocesamiento_ms": pre,
        "total_ms": tot,
    }
    salida = HERE / "resultados_latencia.json"
    salida.write_text(json.dumps(registro, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado en {salida.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
