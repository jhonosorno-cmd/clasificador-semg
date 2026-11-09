"""F15. Matriz de confusión del Ensemble 4M mostrando los ceros explícitamente.

Diferencias con `assets/fig_confusion.png`, que se conserva intacta:
  - Se imprime el valor en las 49 celdas, incluidos los ceros, para que el vacío se lea
    como resultado y no como información faltante.
  - La escala de color se recorta en el máximo fuera de la celda reposo-reposo, de modo
    que el 97,2 % de esa esquina deje de consumir todo el rango y los aciertos de las
    clases activas se vean como lo que son.
  - Formato más ancho y tipografía mayor, para leerse proyectada.

Datos: data/demo_cache.npz con la regla de umbral de src/eval_threshold.py, es decir la
misma configuración del pie del slide: Ensemble 4M con θ=0,36 sobre S1 rep. 6.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import data_sources as ds
import style as st

SALIDA_DECK = Path(__file__).resolve().parents[2] / "presentacion" / "assets" / "fig_confusion_ceros.png"


def main() -> None:
    cache = ds.load_demo_cache()
    labels = cache["labels"]
    theta = float(cache["theta_opt"])
    pred = ds.predict_with_threshold(cache["probs_mean"].astype(float), theta)

    cm = np.zeros((7, 7), dtype=int)
    for real, p in zip(labels, pred):
        cm[real, p] += 1
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    acc = (pred == labels).mean() * 100
    bal = np.mean([(pred[labels == c] == c).mean() for c in range(7)]) * 100

    # La escala se recorta en el mayor valor fuera de la esquina reposo-reposo.
    fuera = pct.copy()
    fuera[0, 0] = 0
    tope = float(np.ceil(fuera.max() / 10) * 10)

    cmap = LinearSegmentedColormap.from_list(
        "uvblues", ["#FFFFFF", "#BFD3E6", "#5B8DB8", st.MODEL_COLORS["v10"]])

    fig = st.new_figure(11.2, 7.6)
    ax = fig.add_subplot(111)
    ax.grid(False)
    im = ax.imshow(pct, cmap=cmap, vmin=0, vmax=tope)

    for i in range(7):
        for j in range(7):
            v = pct[i, j]
            saturada = v >= tope
            color = st.SURFACE if (v > tope * 0.55 or saturada) else (
                st.INK_MUTED if v == 0 else st.INK)
            ax.text(j, i, f"{st.es_number(v, 1)}", ha="center", va="center",
                    fontsize=15 if v > 0 else 13,
                    fontweight="semibold" if i == j else "normal", color=color)

    etiquetas = ["Reposo", "G1\nPulgar", "G2\nÍnd+medio", "G3\nAnu+meñ",
                 "G4\nPulgar op", "G5\nAbducción", "G6\nPuño"]
    ax.set_xticks(range(7)); ax.set_xticklabels(etiquetas, fontsize=12)
    ax.set_yticks(range(7)); ax.set_yticklabels(etiquetas, fontsize=12)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 7, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 7, 1), minor=True)
    ax.grid(which="minor", color=st.SURFACE, linewidth=2.5)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("Clase predicha", fontsize=13, labelpad=10)
    ax.set_ylabel("Clase real", fontsize=13, labelpad=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.03)
    cbar.set_label(f"% de la clase real (escala recortada en {int(tope)} %)", fontsize=11)
    cbar.outline.set_visible(False)

    ceros = int((cm == 0).sum())
    ax.set_title(
        f"Ensemble 4M · umbral θ = {st.es_number(theta)} · exactitud {st.es_number(acc)} % · "
        f"balanceada {st.es_number(bal)} %\n"
        f"{ceros} de 49 celdas valen exactamente cero: los errores no se reparten, "
        f"colapsan hacia el reposo",
        fontsize=13, loc="left", pad=16)

    fig.tight_layout()
    st.source_note(fig, "Fuente: data/demo_cache.npz con la regla de src/eval_threshold.py · "
                        f"test S1 rep. 6, {cm.sum()} ventanas · "
                        f"la celda reposo-reposo ({st.es_number(pct[0,0])} %) satura la escala a propósito.")
    rutas = st.save(fig, "fig15_matriz_confusion_con_ceros")
    SALIDA_DECK.write_bytes(rutas[0].read_bytes())
    print(f"celdas en cero: {ceros}/49 · tope de escala {tope:.0f} % · "
          f"acc {acc:.2f} · balacc {bal:.2f}")
    print(f"copiada a {SALIDA_DECK}")


if __name__ == "__main__":
    main()
