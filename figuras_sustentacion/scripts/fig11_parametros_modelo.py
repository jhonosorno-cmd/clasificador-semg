"""F11. Where the capacity of CyborgHybrid lives, and size of each ensemble member.

Source: INFORME_COMPLETO.md §3.3 (parameters by block, v10) and
GUIA_ESTUDIO.md Módulo 05 (parameters per model); both verified by the
2026-08-22 code audit.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import data_sources as ds
import style as st


def main() -> None:
    blocks = ds.V10_PARAMS_BY_BLOCK
    total = sum(blocks.values())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1.5, 1]})
    st.apply_style()
    fig.set_facecolor(st.SURFACE)

    colors = [st.MODEL_COLORS["v17"], st.MODEL_COLORS["v10"], st.MODEL_COLORS["ens"], st.MODEL_COLORS["v18"]]
    ax1.grid(False)
    left = 0.0
    # Small segments get their label outside the bar, alternating above/below.
    outside_slot = 0
    for (name, n), color in zip(blocks.items(), colors):
        pct = n / total * 100
        ax1.barh([0], [pct], left=left, height=0.5, color=color, edgecolor=st.SURFACE, linewidth=2)
        count = f"{n:,}".replace(",", " ")
        text = f"{name}\n{count} ({st.es_number(pct, 1)} %)"
        if pct > 20:
            ax1.text(left + pct / 2, 0, text, ha="center", va="center", fontsize=8.5, color=st.SURFACE, fontweight="semibold")
        elif pct > 8:
            ax1.text(left + pct / 2, 0.36, text, ha="center", va="bottom", fontsize=8.5, color=color)
        else:
            y, va = (0.36, "bottom") if outside_slot % 2 == 0 else (-0.36, "top")
            ax1.annotate(text, xy=(left + pct / 2, 0.25 if va == "bottom" else -0.25), xytext=(left + pct / 2 + (6 if outside_slot % 2 == 0 else 0), y),
                         ha="center", va=va, fontsize=8.5, color=color,
                         arrowprops={"arrowstyle": "-", "color": st.INK_MUTED, "lw": 0.7})
            outside_slot += 1
        left += pct
    ax1.set_xlim(0, 106)
    ax1.set_ylim(-1.0, 1.0)
    ax1.set_yticks([])
    ax1.spines["left"].set_visible(False)
    st.pct_axis(ax1, "x")
    ax1.set_xlabel(f"Parámetros entrenables de CyborgHybrid v10 (total {total:,})".replace(",", " "))
    ax1.set_title("El 81 % de la capacidad está en el BiLSTM")

    names = list(ds.MODEL_PARAMS)
    vals = np.array(list(ds.MODEL_PARAMS.values())) / 1e6
    bars = ax2.bar(range(len(names)), vals, width=0.6, color=[st.MODEL_COLORS["v10"], st.MODEL_COLORS["v17"], st.MODEL_COLORS["v18"]], zorder=3)
    for b, v in zip(bars, ds.MODEL_PARAMS.values()):
        ax2.text(b.get_x() + b.get_width() / 2, v / 1e6 + 0.02, f"{v:,}".replace(",", " "), ha="center", va="bottom", fontsize=8.5, color=st.INK_2)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, fontsize=8.5)
    ax2.set_ylim(0, 1.3)
    ax2.set_ylabel("Millones de parámetros")
    ax2.set_title("Tamaño de cada miembro del ensemble")

    fig.tight_layout(rect=(0, 0.02, 1, 1))
    st.source_note(fig, "Fuente: INFORME_COMPLETO.md §3.3 y GUIA_ESTUDIO.md (Módulo 05); conteos verificados en auditoría de código.")
    st.save(fig, "fig11_parametros_cyborghybrid")


if __name__ == "__main__":
    main()
