"""F5. LORO-CV: mean ± std (ddof=1, n=6 folds) of the four macro metrics.

Source: checkpoints/loro/loro_cv_summary.json (values per fold).
Los cuatro modelos son los de la Tabla de validacion cruzada del documento.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import data_sources as ds
import style as st

MODELS: list[tuple[str, str, str]] = [
    ("v10", "v10", "v10\n48D"),
    ("v17", "v17", "v17\n132D"),
    ("v18", "v18", "v18\nAtención"),
    ("ensemble_4m", "ens", "Ensemble\n4M"),
]
PANELS: list[tuple[str, str]] = [
    ("accuracy", "Exactitud global"),
    ("precision_macro", "Precisión macro"),
    ("recall_macro", "Recall macro (= BalAcc)"),
    ("f1_macro", "F1 macro"),
]


def main() -> None:
    summary = ds.load_loro_summary()
    fig, axes = plt.subplots(1, 4, figsize=(12, 4.2))
    st.apply_style()
    fig.set_facecolor(st.SURFACE)

    for ax, (metric, title) in zip(axes, PANELS):
        means, stds, colors, labels = [], [], [], []
        for key, ckey, label in MODELS:
            vals = np.array(summary[key][metric]) * 100
            means.append(vals.mean())
            stds.append(vals.std(ddof=1))
            colors.append(st.MODEL_COLORS[ckey])
            labels.append(label)
        x = np.arange(len(MODELS))
        bars = ax.bar(x, means, width=0.6, color=colors, zorder=3)
        ax.errorbar(x, means, yerr=stds, fmt="none", ecolor=st.INK_2, elinewidth=1.2, capsize=4, zorder=4)
        for xi, m, s in zip(x, means, stds):
            ax.text(xi, m + s + 1.0, f"{st.es_number(m, 1)}\n±{st.es_number(s, 1)}", ha="center", va="bottom", fontsize=8, color=st.INK_2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_title(title)
        ax.set_ylim(0, 100)
        st.pct_axis(ax)
        best = int(np.argmax(means))
        bars[best].set_edgecolor(st.HIGHLIGHT)
        bars[best].set_linewidth(2)

    fig.suptitle("Validación cruzada LORO (6 pliegues): media ± desviación estándar por métrica macro",
                 fontsize=12, fontweight="semibold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    st.source_note(fig, "Fuente: checkpoints/loro/loro_cv_summary.json · σ con ddof=1 sobre 6 pliegues · "
                        "Ensemble 4M evaluado sin umbral θ (argmax) · borde dorado = mejor media del panel.")
    st.save(fig, "fig05_loro_metricas_macro_media_std")


if __name__ == "__main__":
    main()
