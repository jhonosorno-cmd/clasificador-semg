"""Mide por qué el desempeño varía entre sujetos, sin entrenar ningún modelo.

Calcula tres propiedades de la señal de cada sujeto y las correlaciona con la exactitud
balanceada ya obtenida y con la proporción de reposo:

  1. Contraste reposo-gesto: RMS medio en ventanas de gesto dividido por el de reposo.
  2. Separabilidad de Fisher en el espacio de 48 características, con y sin la clase reposo.
  3. Duración media de los segmentos etiquetados como gesto, en ventanas.

Hipótesis a distinguir:
  - Etiquetado: gestos nítidos dan segmentos cortos (más reposo) y más separables.
  - Contraste: contracciones fuertes dan mejor recall de todas las clases activas.

No entrena, no escribe fuera de esta carpeta, no toca src/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import NinaProDB3Dataset  # noqa: E402

TODAS_LAS_REPS = [1, 2, 3, 4, 5, 6]
ANTEBRAZO = {1: 50, 2: 70, 3: 30, 4: 40, 5: 90, 6: 40, 7: 0, 8: 50, 9: 90, 10: 50, 11: 90}


def fisher_ratio(feats: np.ndarray, labels: np.ndarray, clases: list[int]) -> float:
    """Razón de dispersión entre clases sobre dispersión dentro de clase.

    Se estandariza cada característica antes, para que ninguna domine por escala.
    Valores altos indican clases más separables sin necesidad de entrenar.

    Args:
        feats: matriz (N, D) de características.
        labels: vector (N,) de etiquetas.
        clases: clases a considerar.

    Returns:
        traza(S_entre) / traza(S_dentro).
    """
    mask = np.isin(labels, clases)
    X, y = feats[mask], labels[mask]
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    media_global = X.mean(axis=0)
    s_entre = s_dentro = 0.0
    for c in clases:
        Xc = X[y == c]
        if len(Xc) < 2:
            continue
        mu = Xc.mean(axis=0)
        s_entre += len(Xc) * float(((mu - media_global) ** 2).sum())
        s_dentro += float(((Xc - mu) ** 2).sum())
    return s_entre / s_dentro if s_dentro else float("nan")


def duracion_segmentos(labels: np.ndarray) -> float:
    """Duración media, en ventanas, de los tramos consecutivos etiquetados como gesto."""
    duraciones, actual = [], 0
    for v in labels:
        if v != 0:
            actual += 1
        elif actual:
            duraciones.append(actual)
            actual = 0
    if actual:
        duraciones.append(actual)
    return float(np.mean(duraciones)) if duraciones else float("nan")


def medir(subject: int) -> dict:
    """Calcula las tres propiedades de señal de un sujeto sobre las seis repeticiones."""
    ds = NinaProDB3Dataset(subjects=[subject], reps_to_use=TODAS_LAS_REPS,
                           mode="train", augment=False, target_subject=subject)
    señales = np.asarray(ds.samples)          # (N, 1, 125, 13)
    labels = np.asarray(ds.labels)            # (N,)
    # Las 48 características se calculan al vuelo en __getitem__; se reproduce aquí la
    # misma llamada, sobre los 12 canales EMG, para no duplicar la definición.
    feats = np.stack([ds._extract_manual_features(señales[i, 0, :, :12])
                      for i in range(len(labels))])

    # RMS medio por ventana sobre los 12 canales originales, sin el canal RMS sintético.
    rms = np.sqrt((señales[:, 0, :, :12] ** 2).mean(axis=(1, 2)))
    rms_reposo = float(rms[labels == 0].mean())
    rms_gesto = float(rms[labels != 0].mean())

    return {
        "subject": subject,
        "ventanas": int(len(labels)),
        "contraste_gesto_reposo": rms_gesto / rms_reposo if rms_reposo else float("nan"),
        "fisher_7_clases": fisher_ratio(feats, labels, list(range(7))),
        "fisher_6_gestos": fisher_ratio(feats, labels, list(range(1, 7))),
        "duracion_media_gesto_ventanas": duracion_segmentos(labels),
        "prop_reposo_total": float((labels == 0).mean() * 100),
    }


def main() -> int:
    medidas = []
    for s in range(1, 12):
        if not (ROOT / "data" / "raw" / f"S{s}_E1_A1.mat").exists():
            print(f"[OMITIDO] S{s}")
            continue
        print(f"\n--- midiendo S{s} ---")
        try:
            m = medir(s)
            medidas.append(m)
            print(f"S{s}: contraste {m['contraste_gesto_reposo']:.3f} | "
                  f"Fisher7 {m['fisher_7_clases']:.4f} | Fisher6 {m['fisher_6_gestos']:.4f} | "
                  f"dur {m['duracion_media_gesto_ventanas']:.1f} vent")
        except Exception as exc:  # se documenta, no se oculta
            print(f"[FALLO] S{s}: {type(exc).__name__}: {exc}")

    out = HERE / "resultados" / "calidad_senal.json"
    out.write_text(json.dumps(medidas, indent=2, ensure_ascii=False), encoding="utf-8")

    # Cruce con los resultados de entrenamiento ya obtenidos.
    res = {json.loads(p.read_text(encoding="utf-8"))["subject"]:
           json.loads(p.read_text(encoding="utf-8"))
           for p in (HERE / "resultados").glob("v10_S*_seed42.json")}
    sub = [m["subject"] for m in medidas if m["subject"] in res]
    bal = np.array([res[s]["balanced_accuracy"] * 100 for s in sub])
    acc = np.array([res[s]["accuracy"] * 100 for s in sub])
    rep = np.array([res[s]["test_class_distribution"][0] /
                    sum(res[s]["test_class_distribution"]) * 100 for s in sub])

    print("\n" + "=" * 78)
    print(f"{'S':<4}{'BalAcc':>9}{'contraste':>11}{'Fisher7':>10}{'Fisher6':>10}{'dur.vent':>10}{'antebr.':>9}")
    for s in sub:
        m = next(x for x in medidas if x["subject"] == s)
        print(f"S{s:<3}{res[s]['balanced_accuracy']*100:>8.2f}%{m['contraste_gesto_reposo']:>11.3f}"
              f"{m['fisher_7_clases']:>10.4f}{m['fisher_6_gestos']:>10.4f}"
              f"{m['duracion_media_gesto_ventanas']:>10.1f}{ANTEBRAZO[s]:>8}%")

    print("\nCORRELACIONES CON LA EXACTITUD BALANCEADA (n=%d)" % len(sub))
    print(f"{'predictor':<34}{'r':>9}{'p':>9}")
    for clave, nombre in [("contraste_gesto_reposo", "contraste gesto/reposo"),
                          ("fisher_7_clases", "separabilidad Fisher, 7 clases"),
                          ("fisher_6_gestos", "separabilidad Fisher, 6 gestos"),
                          ("duracion_media_gesto_ventanas", "duracion media de gesto")]:
        v = np.array([next(x for x in medidas if x["subject"] == s)[clave] for s in sub])
        r, p = stats.pearsonr(v, bal)
        print(f"{nombre:<34}{r:>9.3f}{p:>9.4f}")
    r, p = stats.pearsonr(rep, bal)
    print(f"{'proporcion de reposo (referencia)':<34}{r:>9.3f}{p:>9.4f}")
    fa = np.array([ANTEBRAZO[s] for s in sub], dtype=float)
    r, p = stats.pearsonr(fa, bal)
    print(f"{'antebrazo remanente (referencia)':<34}{r:>9.3f}{p:>9.4f}")

    print("\nRELACION ENTRE PREDICTORES Y LA PROPORCION DE REPOSO")
    for clave, nombre in [("contraste_gesto_reposo", "contraste gesto/reposo"),
                          ("duracion_media_gesto_ventanas", "duracion media de gesto")]:
        v = np.array([next(x for x in medidas if x["subject"] == s)[clave] for s in sub])
        r, p = stats.pearsonr(v, rep)
        print(f"{nombre:<34}{r:>9.3f}{p:>9.4f}")
    print(f"\nGuardado en {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
