"""F8. Pearson correlation between the mean 48D feature profile of each class.

Source: data/demo_cache.npz (class_corr), computed by
src/build_demo_cache.py::class_profile_correlation on the test set.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import data_sources as ds
import style as st


def main() -> None:
    corr = ds.load_demo_cache()["class_corr"].astype(float)
    n = corr.shape[0]
    cmap = LinearSegmentedColormap.from_list("div", [st.MODEL_COLORS["v10"], "#F2F2F2", st.MODEL_COLORS["v17"]])

    fig = st.new_figure(7.2, 6.4)
    ax = fig.add_subplot(111)
    ax.grid(False)
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    for i in range(n):
        for j in range(n):
            v = corr[i, j]
            ax.text(j, i, st.es_number(v, 2), ha="center", va="center", fontsize=8.5,
                    color=st.SURFACE if abs(v) > 0.55 else st.INK)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(st.CLASS_SHORT)
    ax.set_yticklabels(st.CLASS_SHORT)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color=st.SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Correlación de Pearson entre perfiles medios (48D)")
    cbar.outline.set_visible(False)
    ax.set_title("Similitud entre clases en el espacio de características que ve el modelo")

    legend = "   ".join(st.CLASS_NAMES[1:])
    fig.text(0.01, 0.005, legend, fontsize=7.5, color=st.INK_2, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    st.source_note(fig, "Fuente: data/demo_cache.npz (class_corr) · test S1 rep. 6 · "
                        "Cada feature estandarizada antes de promediar por clase (src/build_demo_cache.py).")
    st.save(fig, "fig08_correlacion_entre_clases_48D")


if __name__ == "__main__":
    main()
