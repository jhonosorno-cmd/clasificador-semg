"""F4. Global accuracy vs balanced accuracy for every experiment with both metrics.

Source: latex/capitulos/anexos.tex (Tabla de resultados completos). Rows whose
metric belongs to a different task (v22a binary, v22b 6-gesture) or that lack
a balanced accuracy are excluded rather than filled in.
"""

from __future__ import annotations

import data_sources as ds
import style as st

# Manual label offsets (points) for the crowded 70-72 % region.
OFFSETS: dict[str, tuple[float, float]] = {
    "v10": (7, -9),
    "v17": (7, -8),
    "v18": (-8, 7),
    "4M": (7, 7),
    "v15": (-7, -7),
    "v23": (-7, -7),
    "v26": (-8, -4),
    "v24": (7, 3),
    "v12": (-7, 5),
    "v20": (-7, 4),
    "v21": (7, -8),
    "v19b": (7, 6),
    "v19a": (7, -8),
    "v7": (7, -7),
    "v8": (-7, 5),
    "v5-v6": (-7, 3),
    "v22 casc.": (7, 4),
    "v25": (7, -4),
    "v11": (7, 0),
    "v9": (7, 0),
    "v4": (7, -6),
}
FOCUS: dict[str, str] = {"v10": "v10", "v17": "v17", "v18": "v18", "4M": "ens"}


def main() -> None:
    rows = [r for r in ds.load_experiment_table() if r.accuracy is not None and r.balanced_accuracy is not None]

    fig = st.new_figure(8.5, 5.6)
    ax = fig.add_subplot(111)
    ax.grid(True, axis="both")

    for r in rows:
        key = FOCUS.get(r.version)
        color = st.MODEL_COLORS[key] if key else st.CONTEXT
        size = 90 if key else 45
        ax.scatter(r.accuracy, r.balanced_accuracy, s=size, color=color, edgecolor=st.SURFACE, linewidth=1.2, zorder=3)
        dx, dy = OFFSETS.get(r.version, (6, 0))
        ax.annotate(
            r.version,
            (r.accuracy, r.balanced_accuracy),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.5 if key else 7.5,
            color=st.INK if key else st.INK_MUTED,
            fontweight="semibold" if key else "normal",
            ha="left" if dx > 0 else "right",
            va="center",
        )

    best = max(rows, key=lambda r: r.accuracy)
    best_bal = max(rows, key=lambda r: r.balanced_accuracy)
    ax.axvline(best.accuracy, color=st.HIGHLIGHT, lw=0.9, ls="--")
    ax.axhline(best_bal.balanced_accuracy, color=st.HIGHLIGHT, lw=0.9, ls="--")
    ax.text(best.accuracy - 0.3, 26.5, f"máx. global {st.es_number(best.accuracy)} % ({best.version})",
            fontsize=8, color=st.HIGHLIGHT, ha="right")
    ax.text(44, best_bal.balanced_accuracy + 0.5,
            f"máx. balanceada {st.es_number(best_bal.balanced_accuracy)} % ({best_bal.version}, "
            f"{best_bal.strategy}: colapsa la exactitud global)",
            fontsize=8, color=st.HIGHLIGHT)

    ax.set_xlabel("Exactitud global en prueba")
    ax.set_ylabel("Exactitud balanceada en prueba")
    st.pct_axis(ax, "x")
    st.pct_axis(ax, "y")
    ax.set_xlim(40, 76)
    ax.set_ylim(25, 50)
    ax.set_title("Ningún experimento domina en ambas métricas: exactitud global vs balanceada")
    fig.tight_layout()
    st.source_note(fig, "Fuente: latex/capitulos/anexos.tex, Tabla de resultados completos. "
                        "Excluidos: v1–v3 (≈), v22a/v22b (otra tarea) y filas sin BalAcc.")
    st.save(fig, "fig04_tradeoff_accuracy_vs_balanced")


if __name__ == "__main__":
    main()
