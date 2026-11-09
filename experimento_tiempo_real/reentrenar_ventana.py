"""Reentrena los cuatro componentes del Ensemble 4M con la configuración C.

La configuración C es `filtfilt` aplicado DENTRO de cada ventana de 250 ms ya capturada,
más z-score con estadísticos de una calibración previa. Es implementable en vivo: la
ventana está completa cuando se la filtra, así que la pasada hacia atrás solo lee muestras
que ya existen.

Evaluada sobre los checkpoints originales dio 72,31 % (-0,47 pp contra el control fuera de
línea), pero ahí todavía hay desajuste: esos modelos aprendieron con `filtfilt` de
grabación completa. Este script entrena de cero con el mismo pipeline con el que después se
infiere, para que la diferencia restante sea atribuible solo a la causalidad.

Reutiliza la maquinaria de `reentrenar_causal.py` y el preprocesamiento ya verificado de
`eval_filtfilt_ventana.py`, en lugar de duplicarlos. Escribe en su propia carpeta de
checkpoints, de modo que NO pisa los del experimento causal ni nada del repositorio.

Uso:
    python experimento_tiempo_real/reentrenar_ventana.py
    python experimento_tiempo_real/reentrenar_ventana.py --solo v10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import src.data_loader as dl  # noqa: E402
import reentrenar_causal as rc  # noqa: E402
from eval_filtfilt_ventana import (ESTADO, _create_windows_parcheado,  # noqa: E402
                                   _preprocess_parcheado)

RAMA = "C_filtfilt_vent"
CKPT_VENTANA = HERE / "checkpoints_ventana"

_PREPROC_ORIG = dl.NinaProDB3Dataset._preprocess_emg
_VENTANAS_ORIG = dl.NinaProDB3Dataset._create_windows


def calibrar() -> None:
    """Fija mu/sd a partir de las ventanas filtradas de las repeticiones de entrenamiento.

    Hay que hacerlo una sola vez y antes de construir cualquier otro conjunto: el
    preprocesador parcheado normaliza con estos estadísticos, y validación y prueba deben
    heredarlos, como haría un sistema calibrado antes de usarse.

    Raises:
        RuntimeError: si la pasada no dejó estadísticos.
    """
    print("=" * 74)
    print(f"CALIBRACION · ventanas de las reps {rc.TRAIN_REPS} · filtfilt por ventana")
    print("=" * 74)
    ESTADO["rama"], ESTADO["stats"] = RAMA, {}
    ESTADO["calibrando"] = True
    dl.NinaProDB3Dataset(subjects=[rc.SUJETO], reps_to_use=rc.TRAIN_REPS, mode="train",
                         augment=False, target_subject=rc.SUJETO)
    ESTADO["calibrando"] = False
    if "mu" not in ESTADO["stats"]:
        raise RuntimeError("la pasada de calibracion no fijo estadisticos")
    mu, sd = ESTADO["stats"]["mu"], ESTADO["stats"]["sd"]
    print(f"  mu en [{mu.min():.4f}, {mu.max():.4f}] · sd en [{sd.min():.4f}, {sd.max():.4f}]")


def main() -> int:
    ap = argparse.ArgumentParser(description="Reentrena el ensemble con filtfilt por ventana")
    ap.add_argument("--solo", choices=list(rc.COMPONENTES), help="entrena un solo componente")
    a = ap.parse_args()

    # Checkpoints propios: no se pisan los del experimento causal.
    rc.CKPT = CKPT_VENTANA
    dl.NinaProDB3Dataset._preprocess_emg = _preprocess_parcheado
    dl.NinaProDB3Dataset._create_windows = _create_windows_parcheado

    t0 = time.time()
    try:
        calibrar()
        aliases = [a.solo] if a.solo else list(rc.COMPONENTES)
        individuales = {alias: rc.entrenar(alias) for alias in aliases}

        registro = {
            "descripcion": "Reentrenamiento de los 4 componentes con filtfilt aplicado dentro "
                           "de cada ventana ya capturada (configuracion C) y z-score de "
                           "calibracion. Separa el costo de la causalidad del desajuste entre "
                           "entrenamiento e inferencia.",
            "configuracion": "filtfilt(VENTANA) + z-score calibracion",
            "referencia_sin_reentrenar": {"accuracy": 72.31, "balanced_accuracy": 43.37},
            "control_offline": {"accuracy": 72.78, "balanced_accuracy": 44.92},
            "sujeto": rc.SUJETO, "semilla": rc.SEED,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "individuales": individuales,
        }
        if not a.solo:
            print("\n" + "=" * 74 + "\nENSEMBLE 4M · filtfilt por ventana\n" + "=" * 74)
            registro["ensemble"] = rc.evaluar_ensemble()
            e = registro["ensemble"]
            print(f"  argmax              {e['argmax']['accuracy']:.2f}%  "
                  f"bal {e['argmax']['balanced_accuracy']:.2f}%")
            print(f"  θ=0,36 (documento)  {e['theta_documento']['accuracy']:.2f}%  "
                  f"bal {e['theta_documento']['balanced_accuracy']:.2f}%")
            print(f"  θ={e['theta_recalibrado_valor']:.2f} (recalibrado)  "
                  f"{e['theta_recalibrado']['accuracy']:.2f}%  "
                  f"bal {e['theta_recalibrado']['balanced_accuracy']:.2f}%")
            print(f"\n  contra control fuera de linea (72,78%): "
                  f"{e['argmax']['accuracy'] - 72.78:+.2f} pp")
            print(f"  contra el mismo pipeline sin reentrenar (72,31%): "
                  f"{e['argmax']['accuracy'] - 72.31:+.2f} pp")
    finally:
        dl.NinaProDB3Dataset._preprocess_emg = _PREPROC_ORIG
        dl.NinaProDB3Dataset._create_windows = _VENTANAS_ORIG

    registro["minutos_totales"] = round((time.time() - t0) / 60, 2)
    salida = HERE / ("resultados_reentreno_ventana.json" if not a.solo
                     else f"resultados_reentreno_ventana_{a.solo}.json")
    salida.write_text(json.dumps(registro, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado en {salida.relative_to(ROOT)} · {registro['minutos_totales']} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
