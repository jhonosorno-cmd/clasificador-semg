"""¿Cambia la separabilidad si se alarga la ventana de análisis?

Los 250 ms están fijos en `src/config.py` y ninguna de las 28 versiones del trabajo los
varió. Este script mide la razón de Fisher sobre las 48 características para varias
longitudes de ventana, **sin entrenar ningún modelo**, para decidir barato si vale la pena
gastar horas de entrenamiento en esa dirección.

La constante se sustituye en memoria sobre el módulo ya importado. NO se modifica
`src/config.py` ni `src/data_loader.py`, que usan la global en tiempo de ejecución.
Se verifica en cada corrida que la sustitución tuvo efecto, comprobando la forma real de
las ventanas generadas.

El traslape se mantiene en 80 %, igual que la configuración original (stride = ventana/5),
para que la densidad de muestreo sea comparable entre configuraciones.

Salidas: resultados/ventana_separabilidad.json. No escribe fuera de esta carpeta.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import src.data_loader as dl  # noqa: E402
from medir_calidad_senal import fisher_ratio  # noqa: E402

TODAS_LAS_REPS = [1, 2, 3, 4, 5, 6]
FRECUENCIA = 500  # Hz, tras el remuestreo

# (muestras, milisegundos). 125 es la configuración actual del trabajo.
VENTANAS = [(62, 124), (125, 250), (250, 500), (500, 1000)]

# Sujetos que cubren el rango de separabilidad observado: el del trabajo, el mejor,
# uno intermedio y el peor.
SUJETOS = [1, 3, 9, 7]


def medir_ventana(subject: int, win: int) -> dict:
    """Construye el dataset con una ventana dada y mide separabilidad.

    Args:
        subject: Id del sujeto de DB3.
        win: Longitud de ventana en muestras a 500 Hz.

    Returns:
        Diccionario con conteos y razones de Fisher.

    Raises:
        RuntimeError: si la sustitución de la constante no tuvo efecto.
    """
    stride = max(1, win // 5)  # 80 % de traslape, igual que el original
    dl.WINDOW_SIZE, dl.STRIDE = win, stride

    t0 = time.time()
    ds = dl.NinaProDB3Dataset(subjects=[subject], reps_to_use=TODAS_LAS_REPS,
                              mode="train", augment=False, target_subject=subject)
    señales = np.asarray(ds.samples)
    labels = np.asarray(ds.labels)

    real = señales.shape[2]
    if real != win:
        raise RuntimeError(
            f"la sustitución no tuvo efecto: se pidió ventana {win} y se generó {real}")

    feats = np.stack([ds._extract_manual_features(señales[i, 0, :, :12])
                      for i in range(len(labels))])
    rms = np.sqrt((señales[:, 0, :, :12] ** 2).mean(axis=(1, 2)))
    rms_reposo = float(rms[labels == 0].mean())
    rms_gesto = float(rms[labels != 0].mean())

    return {
        "subject": subject,
        "ventana_muestras": win,
        "ventana_ms": round(win / FRECUENCIA * 1000),
        "stride_muestras": stride,
        "marcos_tras_cnn": win // 8,  # tres MaxPool(2) en la rama convolucional
        "n_ventanas": int(len(labels)),
        "prop_reposo": float((labels == 0).mean() * 100),
        "contraste_gesto_reposo": rms_gesto / rms_reposo if rms_reposo else float("nan"),
        "fisher_7_clases": fisher_ratio(feats, labels, list(range(7))),
        "fisher_6_gestos": fisher_ratio(feats, labels, list(range(1, 7))),
        "segundos": round(time.time() - t0, 1),
    }


def main() -> int:
    ventana_original = dl.WINDOW_SIZE
    stride_original = dl.STRIDE
    print(f"Configuración original del proyecto: ventana {ventana_original}, stride {stride_original}\n")

    medidas = []
    for s in SUJETOS:
        if not (ROOT / "data" / "raw" / f"S{s}_E1_A1.mat").exists():
            print(f"[OMITIDO] S{s}: archivo ausente")
            continue
        for win, _ in VENTANAS:
            print(f"\n--- S{s} · ventana {win} muestras ---")
            try:
                m = medir_ventana(s, win)
                medidas.append(m)
                print(f"S{s} {m['ventana_ms']:>5} ms | {m['n_ventanas']:>6} vent | "
                      f"Fisher7 {m['fisher_7_clases']:.4f} | Fisher6 {m['fisher_6_gestos']:.4f} | "
                      f"{m['segundos']} s")
            except Exception as exc:
                print(f"[FALLO] S{s} ventana {win}: {type(exc).__name__}: {exc}")

    # Restaurar el estado del módulo por higiene, aunque el proceso termine aquí.
    dl.WINDOW_SIZE, dl.STRIDE = ventana_original, stride_original

    out = HERE / "resultados" / "ventana_separabilidad.json"
    out.write_text(json.dumps(medidas, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 88)
    print("SEPARABILIDAD DE FISHER SEGÚN LA LONGITUD DE VENTANA (sin entrenar)")
    print("=" * 88)
    print(f"{'Sujeto':<8}{'ventana':>10}{'marcos':>9}{'ventanas':>10}{'Fisher7':>10}"
          f"{'vs 250ms':>11}{'Fisher6':>10}{'contraste':>11}")
    for s in SUJETOS:
        base = next((m for m in medidas if m["subject"] == s and m["ventana_muestras"] == 125), None)
        for m in [x for x in medidas if x["subject"] == s]:
            delta = ("base" if m["ventana_muestras"] == 125
                     else (f"{(m['fisher_7_clases']/base['fisher_7_clases']-1)*100:+.1f} %"
                           if base else "—"))
            print(f"S{m['subject']:<7}{m['ventana_ms']:>8} ms{m['marcos_tras_cnn']:>9}"
                  f"{m['n_ventanas']:>10}{m['fisher_7_clases']:>10.4f}{delta:>11}"
                  f"{m['fisher_6_gestos']:>10.4f}{m['contraste_gesto_reposo']:>11.3f}")
        print()

    print(f"Guardado en {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
