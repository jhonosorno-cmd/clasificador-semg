"""F10. Class distribution of the test set (S1, repetition 6).

Source: results/ensemble_confusion.json (per_class.total).
"""

from __future__ import annotations

import numpy as np

import data_sources as ds
import style as st


def main() -> None:
    conf = ds.load_confusion()
    totals = np.array([conf["per_class"][str(c)]["total"] for c in range(7)])
    share = totals / totals.sum() * 100

    fig = st.new_figure(8, 4.4)
    ax = fig.add_subplot(111)
    colors = [st.MODEL_COLORS["v10"]] + [st.CONTEXT] * 6
    bars = ax.bar(range(7), totals, width=0.62, color=colors, zorder=3)
    for b, t, s in zip(bars, totals, share):
        ax.text(b.get_x() + b.get_width() / 2, t + 15, f"{t}\n{st.es_number(s, 1)} %", ha="center", va="bottom", fontsize=8.5, color=st.INK_2)
    ax.set_xticks(range(7))
    ax.set_xticklabels(st.CLASS_SHORT)
    ax.set_ylim(0, 1400)
    ax.set_ylabel("Ventanas de 250 ms")
    ax.set_title(f"Desbalance del conjunto de prueba: {totals.sum()} ventanas, {st.es_number(share[0], 1)} % de reposo")
    fig.text(0.01, 0.005, "   ".join(st.CLASS_NAMES[1:]), fontsize=7.5, color=st.INK_2, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    st.source_note(fig, "Fuente: results/ensemble_confusion.json (per_class.total) · S1 rep. 6.")
    st.save(fig, "fig10_distribucion_clases_test")


if __name__ == "__main__":
    main()
