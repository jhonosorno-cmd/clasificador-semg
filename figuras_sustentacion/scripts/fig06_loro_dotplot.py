"""F6. LORO-CV raw accuracy per fold as a dot plot (n = 6 per model).

Source: checkpoints/loro/loro_cv_summary.json (v10, v17, v18, ensemble_4m) and
checkpoints/loro/v15_fold_k_metrics.json (v15, ensemble component that the
thesis table omits; shown in gray as context).
"""

from __future__ import annotations

import numpy as np

import data_sources as ds
import style as st

COLUMNS: list[tuple[str, str, str]] = [
    ("v10", "v10", "v10\n48D"),
    ("v17", "v17", "v17\n132D"),
    ("v18", "v18", "v18\nAtención"),
    ("ensemble_4m", "ens", "Ensemble\n4M"),
]


def main() -> None:
    summary = ds.load_loro_summary()

    fig = st.new_figure(9, 5)
    ax = fig.add_subplot(111)

    for xi, (key, ckey, label) in enumerate(COLUMNS):
        vals = np.array(summary[key]["accuracy"]) * 100
        color = st.MODEL_COLORS[ckey]
        mean = vals.mean()
        ax.hlines(mean, xi - 0.28, xi + 0.28, color=color, lw=2.4, zorder=3)
        ax.scatter(np.full(6, xi), vals, s=70, color=color, edgecolor=st.SURFACE, linewidth=1.5, zorder=4)
        # Fold labels: push apart vertically so close folds stay legible.
        order = np.argsort(vals)
        label_y = vals[order].copy()
        for k in range(1, len(label_y)):
            label_y[k] = max(label_y[k], label_y[k - 1] + 0.33)
        for k, idx in enumerate(order):
            ax.text(xi + 0.13, label_y[k], f"R{idx + 1}", fontsize=7, color=st.INK_MUTED, va="center")
        ax.text(xi - 0.34, mean, f"{st.es_number(mean)} %", fontsize=8.5, color=color, ha="right", va="center", fontweight="semibold")

    ax.set_xticks(range(len(COLUMNS)))
    ax.set_xticklabels([c[2] for c in COLUMNS])
    ax.set_xlim(-0.9, len(COLUMNS) - 0.4)
    ax.set_ylim(62, 75)
    st.pct_axis(ax)
    ax.set_ylabel("Exactitud global por pliegue (repetición de prueba)")
    ax.set_title("LORO-CV: los 6 pliegues de cada modelo y su media (línea)")
    fig.tight_layout()
    st.source_note(fig, "Fuente: checkpoints/loro/loro_cv_summary.json · R6 = partición original (test rep. 6) · "
                        "Ensemble 4M sin umbral θ.")
    st.save(fig, "fig06_loro_dotplot_pliegues")


if __name__ == "__main__":
    main()
