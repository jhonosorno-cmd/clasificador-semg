"""F2. Rest-threshold sweep on the VALIDATION set (how theta = 0.36 was chosen).

Source: data/demo_cache.npz (theta_grid, theta_val_acc, theta_val_bal), which
reproduces src/eval_threshold.py on the ensemble's validation probabilities.
"""

from __future__ import annotations

import numpy as np

import data_sources as ds
import style as st


def main() -> None:
    cache = ds.load_demo_cache()
    theta = cache["theta_grid"].astype(float)
    val_acc = cache["theta_val_acc"].astype(float) * 100
    val_bal = cache["theta_val_bal"].astype(float) * 100
    theta_opt = float(cache["theta_opt"])
    i_opt = int(np.argmin(np.abs(theta - theta_opt)))

    fig = st.new_figure(8.5, 4.6)
    ax = fig.add_subplot(111)
    ax.plot(theta, val_acc, color=st.MODEL_COLORS["v10"], lw=2.2, label="Exactitud global (val)")
    ax.plot(theta, val_bal, color=st.MODEL_COLORS["v17"], lw=2.2, label="Exactitud balanceada (val)")
    ax.axvline(theta_opt, color=st.HIGHLIGHT, lw=1.2, ls="--")

    for y, lab, dy in ((val_acc[i_opt], "acc", 2.0), (val_bal[i_opt], "bal", -4.5)):
        ax.plot(theta_opt, y, "o", ms=7, color=st.HIGHLIGHT, mec=st.SURFACE, mew=1.5)
    ax.annotate(
        f"θ óptimo = {st.es_number(theta_opt)}\n"
        f"val acc {st.es_number(val_acc[i_opt])} % · val BalAcc {st.es_number(val_bal[i_opt])} %",
        xy=(theta_opt, val_acc[i_opt]),
        xytext=(theta_opt + 0.06, 64),
        fontsize=9,
        color=st.INK_2,
        arrowprops={"arrowstyle": "-", "color": st.INK_MUTED, "lw": 0.8},
    )
    ax.text(theta[-1], val_acc[-1], "  global", va="center", fontsize=9, color=st.MODEL_COLORS["v10"])
    ax.text(theta[-1], val_bal[-1], "  balanceada", va="center", fontsize=9, color=st.MODEL_COLORS["v17"])

    ax.set_xlabel("Umbral θ sobre P(reposo)")
    ax.set_ylabel("Exactitud en validación (rep. 5)")
    ax.set_xlim(0.08, 1.02)
    ax.set_ylim(38, 75)
    st.pct_axis(ax)
    ax.legend(loc="lower left")
    ax.set_title("Selección del umbral de reposo θ sobre el conjunto de validación (Ensemble 4M)")
    fig.tight_layout()
    st.source_note(fig, "Fuente: data/demo_cache.npz (barrido de src/eval_threshold.py sobre val_probs). "
                        "Regla: reposo si P(reposo) > θ; si no, argmax de las clases activas.")
    st.save(fig, "fig02_barrido_theta_validacion")


if __name__ == "__main__":
    main()
