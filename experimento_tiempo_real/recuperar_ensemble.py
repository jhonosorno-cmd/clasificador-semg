"""Recupera la evaluación del ensemble causal a partir de los checkpoints ya entrenados.

El reentrenamiento completó los cuatro modelos pero murió al imprimir el resumen, antes de
escribir el JSON. Los pesos están en `checkpoints/`, así que solo falta reevaluar y guardar.
Los resultados individuales se toman del log de esa corrida.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import src.data_loader as dl
from reentrenar_causal import SEED, SUJETO, evaluar_ensemble, preprocesar_causal

# Del log de la corrida del 15-sep: alias -> (test acc, test balanceada, épocas, minutos)
INDIVIDUALES = {
    "v10": (70.32, 42.60, 25, 14.82),
    "v15": (71.84, 46.42, 38, 22.72),
    "v17": (69.54, 38.37, 48, 41.10),
    "v18": (70.64, 41.99, 38, 25.02),
}

dl.NinaProDB3Dataset._preprocess_emg = preprocesar_causal
ens = evaluar_ensemble()

registro = {
    "descripcion": "Ensemble 4M reentrenado con pipeline causal (lfilter + z-score de "
                   "calibracion). Entrenamiento e inferencia comparten preprocesamiento.",
    "sujeto": SUJETO, "semilla": SEED,
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "individuales": {k: {"test_accuracy": a, "test_balanced": b, "epocas": e, "minutos": m}
                     for k, (a, b, e, m) in INDIVIDUALES.items()},
    "ensemble": ens,
}
salida = HERE / "resultados_reentreno.json"
salida.write_text(json.dumps(registro, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"\n{'':26}{'exactitud':>11}{'balanceada':>13}")
for regla in ("argmax", "theta_documento", "theta_recalibrado"):
    r = ens[regla]
    print(f"  {regla:24}{r['accuracy']:10.2f}%{r['balanced_accuracy']:12.2f}%")
print(f"\n  theta del documento : {ens['theta_documento_valor']}")
print(f"  theta recalibrado   : {ens['theta_recalibrado_valor']}")
print(f"\nGuardado en {salida.name}")
