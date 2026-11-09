"""F16. Exactitud por clase de los cuatro modelos, no solo v10 contra el ensemble.

Reemplaza a `assets/tikz_metricas_clase.png` en la diapositiva de ganancia por clase, que
comparaba únicamente v10 con el Ensemble 4M. Con los cuatro modelos se ve el patrón que el
deck afirma en varias partes: los tres grupos de desempeño son consistentes entre modelos,
y la ganancia del ensemble está concentrada en G2.

Los valores reproducen exactamente el mapa de calor por clase del documento: modelos
individuales por argmax y la fila del Ensemble 4M con el umbral θ=0,36.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import data_sources as ds
import style as st

SALIDA_DECK = Path(__file__).resolve().parents[2] / "presentacion" / "assets" / "fig_clase_4modelos.png"

MODELOS = [
    ("v10", "probs_v10", "v10 · CyborgHybrid 48D"),
    ("v17", "probs_v17", "v17 · 132D"),
    ("v18", "probs_v18", "v18 · Atención"),
    ("ens", "probs_mean", "Ensemble 4M"),
]
CLASES = ["Reposo", "G1\nPulgar\narriba", "G2\nÍnd+\nmedio", "G3\nAnu+\nmeñ",
          "G4\nPulgar\nopuesto", "G5\nAbduc-\nción", "G6\nPuño"]


def main() -> None:
    cache = ds.load_demo_cache()
    y = cache["labels"]

    # El documento evalúa los modelos individuales por argmax y la fila del Ensemble 4M
    # con el umbral θ=0,36. Se replica esa convención para que cada valor de esta figura
    # coincida con el mapa de calor por clase del trabajo escrito.
    theta = float(cache["theta_opt"])
    recalls = {}
    for clave, campo, _ in MODELOS:
        probs = cache[campo].astype(float)
        pred = ds.predict_with_threshold(probs, theta) if clave == "ens" else probs.argmax(axis=1)
        recalls[clave] = np.array([(pred[y == c] == c).mean() * 100 for c in range(7)])

    st.apply_style()
    fig = st.new_figure(13.0, 6.4)
    ax = fig.add_subplot(111)

    x = np.arange(7)
    ancho = 0.2
    for i, (clave, _, etiqueta) in enumerate(MODELOS):
        desplazamiento = (i - 1.5) * ancho
        destacado = clave == "ens"
        ax.bar(x + desplazamiento, recalls[clave], ancho,
               label=etiqueta, color=st.MODEL_COLORS[clave],
               edgecolor=st.SURFACE, linewidth=0.8,
               zorder=3 if destacado else 2,
               alpha=1.0 if destacado else 0.85)

    # Solo se rotula el ensemble: con 28 barras, rotularlas todas es ilegible.
    for xi, v in zip(x + 1.5 * ancho, recalls["ens"]):
        ax.text(xi, v + 1.6, st.es_number(v, 1), ha="center", va="bottom",
                fontsize=10, color=st.MODEL_COLORS["ens"], fontweight="semibold")

    # Las tres bandas de desempeño que el deck sostiene en la diapositiva 19.
    ax.axvspan(-0.5, 0.5, color=st.INK_MUTED, alpha=0.05, zorder=0)
    ax.axvspan(0.5, 4.5, color=st.MODEL_COLORS["v17"], alpha=0.05, zorder=0)
    ax.axvspan(4.5, 6.5, color=st.MODEL_COLORS["ens"], alpha=0.05, zorder=0)
    for xi, texto in ((0, "fácil"), (2.5, "gestos finos"), (5.5, "gestos gruesos")):
        ax.text(xi, 103, texto, ha="center", va="bottom", fontsize=10,
                color=st.INK_MUTED, style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(CLASES, fontsize=11)
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Exactitud por clase (%)", fontsize=12)
    st.pct_axis(ax, axis="y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4,
              frameon=False, fontsize=11)
    ax.set_title("El patrón de los tres grupos se repite en los cuatro modelos · S1 rep. 6",
                 fontsize=13, loc="left", pad=26)

    fig.tight_layout()
    st.source_note(fig, "Fuente: data/demo_cache.npz · exactitud por clase = aciertos sobre el total real de cada clase · "
                        "mismos valores del mapa de calor del documento · "
                        "modelos individuales por argmax, Ensemble 4M con θ=0,36 · 1 914 ventanas.")
    rutas = st.save(fig, "fig16_exactitud_por_clase_modelos")
    SALIDA_DECK.write_bytes(rutas[0].read_bytes())

    print(f"{'clase':24}" + "".join(f"{c:>10}" for c, _, _ in MODELOS))
    for i, nombre in enumerate(["Reposo", "G1", "G2", "G3", "G4", "G5", "G6"]):
        print(f"{nombre:24}" + "".join(f"{recalls[c][i]:>10.1f}" for c, _, _ in MODELOS))
    print(f"\ncopiada a {SALIDA_DECK}")


if __name__ == "__main__":
    main()
