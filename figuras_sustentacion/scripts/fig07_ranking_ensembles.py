"""F7. Ranking of the nine ensemble combinations evaluated on the test set.

Source: logs/eval_ensemble_v18.log (RESULTADOS blocks). Reference lines from
resultados.tex: v10 individual 71.32 %.
"""

from __future__ import annotations

import numpy as np

import data_sources as ds
import style as st

V10_ACC: float = 71.32


def pretty(name: str) -> str:
    name = name.replace("Ensemble ", "")
    name = name.replace("pond ", "ponderado ")
    return name


def main() -> None:
    results = sorted(ds.parse_eval_log("eval_ensemble_v18"), key=lambda r: r.accuracy)
    names = [pretty(r.name) for r in results]
    acc = np.array([r.accuracy for r in results])
    bal = np.array([r.balanced_accuracy for r in results])
    is_best = [r.name.endswith("(4M)") for r in results]

    fig = st.new_figure(9.5, 5)
    ax = fig.add_subplot(111)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    y = np.arange(len(results))
    colors = [st.MODEL_COLORS["ens"] if b else st.CONTEXT for b in is_best]
    bars = ax.barh(y, acc - 70, left=70, height=0.62, color=colors, zorder=3)
    for yi, a, b, best in zip(y, acc, bal, is_best):
        ax.text(a + 0.03, yi, f"{st.es_number(a)} %", va="center", fontsize=9,
                color=st.INK if best else st.INK_2, fontweight="semibold" if best else "normal")
        ax.text(a + 0.55, yi, f"BalAcc {st.es_number(b)} %", va="center", fontsize=7.5, color=st.INK_MUTED)
    ax.axvline(V10_ACC, color=st.MODEL_COLORS["v10"], lw=1.2, ls="--")
    ax.text(V10_ACC + 0.04, -0.75, f"v10 individual {st.es_number(V10_ACC)} %", fontsize=8, color=st.MODEL_COLORS["v10"], va="center")
    ax.set_ylim(-1.1, len(results) - 0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlim(70, 73.9)
    st.pct_axis(ax, "x", 1)
    ax.set_xlabel("Exactitud global en prueba (argmax, sin umbral)")
    ax.set_title("Las 9 combinaciones de ensemble ensayadas: todas dentro de 0,6 pp")
    fig.tight_layout()
    st.source_note(fig, "Fuente: logs/eval_ensemble_v18.log · Combinaciones comparadas sobre el conjunto de prueba "
                        "(protocolo documentado en metodologia.tex, Ensemble 4M).")
    st.save(fig, "fig07_ranking_combinaciones_ensemble")


if __name__ == "__main__":
    main()
