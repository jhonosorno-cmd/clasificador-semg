"""F13. From the best single model to the final result: inference-time variants.

Source: resultados.tex (v10), logs/eval_threshold.log (argmax baseline and
theta = 0.36), logs/eval_tta.log (TTA-7). Paper reference: Arcos et al. 79.72 %.
"""

from __future__ import annotations

import numpy as np

import data_sources as ds
import style as st


def main() -> None:
    names = [v[0] for v in ds.INFERENCE_VARIANTS]
    acc = np.array([v[1] for v in ds.INFERENCE_VARIANTS])
    bal = np.array([v[2] for v in ds.INFERENCE_VARIANTS])
    colors = [st.MODEL_COLORS["v10"], st.MODEL_COLORS["ens"], st.HIGHLIGHT, st.MODEL_COLORS["ens"]]

    fig = st.new_figure(8.5, 4.8)
    ax = fig.add_subplot(111)
    x = np.arange(len(names))
    bars = ax.bar(x, acc - 68, bottom=68, width=0.58, color=colors, zorder=3)
    for xi, a, b in zip(x, acc, bal):
        ax.text(xi, a + 0.12, f"{st.es_number(a)} %", ha="center", va="bottom", fontsize=10, color=st.INK, fontweight="semibold")
        ax.text(xi, 68.25, f"BalAcc\n{st.es_number(b)} %", ha="center", va="bottom", fontsize=8, color=st.SURFACE)
    ax.axhline(ds.PAPER_BASELINE, color=st.INK_MUTED, lw=1.0, ls="--")
    ax.text(len(names) - 0.5, ds.PAPER_BASELINE + 0.15, f"referente Arcos et al. 2024: {st.es_number(ds.PAPER_BASELINE)} %",
            ha="right", fontsize=8.5, color=st.INK_MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(68, 81)
    st.pct_axis(ax)
    ax.set_ylabel("Exactitud global en prueba (S1 rep. 6)")
    ax.set_title("Variantes de inferencia sobre el mismo ensemble: décimas de punto, con costo en BalAcc")
    fig.tight_layout()
    st.source_note(fig, "Fuente: resultados.tex (v10), logs/eval_threshold.log (argmax y θ=0,36 elegido en validación), "
                        "logs/eval_tta.log (TTA-7). El resultado reportado en la tesis es 72,88 %.")
    st.save(fig, "fig13_variantes_inferencia_final")


if __name__ == "__main__":
    main()
