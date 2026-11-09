"""F9. Error analysis of the Ensemble 4M (argmax): where do the 521 errors go?

Source: results/ensemble_confusion.json (confusion_matrix_counts). The three
categories are sums over the off-diagonal cells; nothing is estimated.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import data_sources as ds
import style as st


def main() -> None:
    conf = ds.load_confusion()
    cm = np.array(conf["confusion_matrix_counts"])
    total = int(cm.sum())
    errors = total - int(np.trace(cm))
    g2r = int(cm[1:, 0].sum())                      # gesture predicted as rest
    r2g = int(cm[0, 1:].sum())                      # rest predicted as gesture
    g2g = errors - g2r - r2g                        # gesture predicted as another gesture
    assert g2g == int(cm[1:, 1:].sum() - np.trace(cm[1:, 1:]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1.15, 1]})
    st.apply_style()
    fig.set_facecolor(st.SURFACE)

    parts = [("Gesto → reposo", g2r, st.MODEL_COLORS["v10"]),
             ("Gesto → otro gesto", g2g, st.MODEL_COLORS["v17"]),
             ("Reposo → gesto", r2g, st.MODEL_COLORS["v18"])]
    ax1.grid(False)
    left = 0.0
    for label, count, color in parts:
        pct = count / errors * 100
        ax1.barh([0], [pct], left=left, height=0.5, color=color, edgecolor=st.SURFACE, linewidth=2, label=label)
        ax1.text(left + pct / 2, 0, f"{st.es_number(pct, 1)} %\n({count})", ha="center", va="center",
                 fontsize=9.5, color=st.SURFACE, fontweight="semibold")
        left += pct
    ax1.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.97), handlelength=1.2, columnspacing=1.2)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(-0.6, 0.9)
    ax1.set_yticks([])
    ax1.set_xlabel(f"Porcentaje de los {errors} errores del Ensemble 4M (test, {total} ventanas)")
    st.pct_axis(ax1, "x")
    for spine in ("left",):
        ax1.spines[spine].set_visible(False)
    ax1.set_title("El modo de fallo dominante es la no detección")

    totals = cm[1:].sum(axis=1)
    absorbed = cm[1:, 0] / totals * 100
    detected = np.diag(cm)[1:] / totals * 100
    y = np.arange(6)
    ax2.grid(True, axis="x")
    ax2.grid(False, axis="y")
    ax2.barh(y, detected, height=0.6, color=st.MODEL_COLORS["ens"], zorder=3, label="Acertado")
    ax2.barh(y, absorbed, left=detected, height=0.6, color=st.MODEL_COLORS["v10"], zorder=3, label="Predicho como reposo")
    ax2.barh(y, 100 - detected - absorbed, left=detected + absorbed, height=0.6, color=st.CONTEXT, zorder=3, label="Otro gesto")
    for yi, d, a in zip(y, detected, absorbed):
        ax2.text(d / 2, yi, f"{st.es_number(d, 1)}", ha="center", va="center", fontsize=8, color=st.SURFACE)
        ax2.text(d + a / 2, yi, f"{st.es_number(a, 1)}", ha="center", va="center", fontsize=8, color=st.SURFACE)
    ax2.set_yticks(y)
    ax2.set_yticklabels(st.CLASS_NAMES[1:], fontsize=8.5)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 100)
    st.pct_axis(ax2, "x")
    ax2.set_xlabel("Ventanas de cada gesto (≈125 por gesto)")
    ax2.legend(loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.22), handlelength=1.2)
    ax2.set_title("Destino de las ventanas de cada gesto")

    fig.tight_layout(rect=(0, 0.02, 1, 1))
    st.source_note(fig, "Fuente: results/ensemble_confusion.json (conteos, Ensemble 4M sin umbral, 72,78 %).")
    st.save(fig, "fig09_analisis_errores_ensemble_4m")


if __name__ == "__main__":
    main()
