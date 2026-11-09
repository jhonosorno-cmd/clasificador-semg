"""F3. Effect of theta on the TEST set: global vs balanced, and rest vs gestures.

Source: data/demo_cache.npz (probs_mean, labels) evaluated with the exact rule
of src/eval_threshold.py. The script asserts that it reproduces the values
printed in logs/eval_threshold.log (72.41 % at 0.20, 72.88 % at 0.36, ...).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import data_sources as ds
import style as st

# (theta, overall %, balanced %) rows from logs/eval_threshold.log "CURVA ACCURACY vs THETA"
LOG_CHECK: list[tuple[float, float, float]] = [
    (0.20, 72.41, 42.10),
    (0.36, 72.88, 44.24),
    (0.50, 72.41, 45.14),
    (0.70, 72.05, 49.22),
    (0.90, 65.57, 50.94),
]


def main() -> None:
    cache = ds.load_demo_cache()
    probs = cache["probs_mean"].astype(float)
    labels = cache["labels"]
    theta_opt = float(cache["theta_opt"])

    grid = np.round(np.arange(0.10, 0.99, 0.02), 2)
    overall, balanced, rest_recall, gesture_recall = [], [], [], []
    for t in grid:
        pred = ds.predict_with_threshold(probs, t)
        rec = ds.per_class_recall(pred, labels)
        overall.append((pred == labels).mean() * 100)
        balanced.append(rec.mean() * 100)
        rest_recall.append(rec[0] * 100)
        gesture_recall.append(rec[1:].mean() * 100)

    for t, acc_log, bal_log in LOG_CHECK:
        pred = ds.predict_with_threshold(probs, t)
        rec = ds.per_class_recall(pred, labels)
        assert abs((pred == labels).mean() * 100 - acc_log) < 0.01, t
        assert abs(rec.mean() * 100 - bal_log) < 0.01, t

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    st.apply_style()
    fig.set_facecolor(st.SURFACE)

    ax1.plot(grid, overall, color=st.MODEL_COLORS["v10"], lw=2.2, label="Exactitud global")
    ax1.plot(grid, balanced, color=st.MODEL_COLORS["v17"], lw=2.2, label="Exactitud balanceada")
    ax1.axvline(theta_opt, color=st.HIGHLIGHT, lw=1.2, ls="--")
    i = int(np.argmin(np.abs(grid - theta_opt)))
    ax1.plot([theta_opt, theta_opt], [overall[i], balanced[i]], "o", ms=7, color=st.HIGHLIGHT, mec=st.SURFACE, mew=1.5)
    ax1.text(theta_opt + 0.02, overall[i] + 1.2, f"{st.es_number(overall[i])} %", fontsize=9, color=st.INK_2)
    ax1.text(theta_opt + 0.02, balanced[i] - 3.2, f"{st.es_number(balanced[i])} %", fontsize=9, color=st.INK_2)
    ax1.text(theta_opt, 78.5, f"θ = {st.es_number(theta_opt)}", fontsize=9, color=st.HIGHLIGHT, ha="center", va="bottom")
    ax1.set_title("Global vs balanceada en prueba")
    ax1.set_xlabel("Umbral θ sobre P(reposo)")
    ax1.set_ylabel("Exactitud (test, rep. 6)")
    ax1.set_ylim(35, 81)
    st.pct_axis(ax1)
    ax1.legend(loc="center left")

    ax2.plot(grid, rest_recall, color=st.INK_2, lw=2.2, label="Recall de reposo")
    ax2.plot(grid, gesture_recall, color=st.MODEL_COLORS["ens"], lw=2.2, label="Recall medio de los 6 gestos")
    ax2.axvline(theta_opt, color=st.HIGHLIGHT, lw=1.2, ls="--")
    ax2.text(theta_opt + 0.02, rest_recall[i] - 4, f"{st.es_number(rest_recall[i], 1)} %", fontsize=9, color=st.INK_2)
    ax2.text(theta_opt + 0.02, gesture_recall[i] + 2, f"{st.es_number(gesture_recall[i], 1)} %", fontsize=9, color=st.MODEL_COLORS["ens"])
    ax2.set_title("El reposo absorbe a los gestos")
    ax2.set_xlabel("Umbral θ sobre P(reposo)")
    ax2.set_ylabel("Recall (test, rep. 6)")
    ax2.set_ylim(0, 105)
    st.pct_axis(ax2)
    ax2.legend(loc="center left")

    fig.suptitle("Compromiso del umbral de reposo en el conjunto de prueba (Ensemble 4M)",
                 fontsize=12, fontweight="semibold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    st.source_note(fig, "Fuente: data/demo_cache.npz (probs_mean, labels) con la regla de src/eval_threshold.py; "
                        "valores verificados contra logs/eval_threshold.log.")
    st.save(fig, "fig03_barrido_theta_test")


if __name__ == "__main__":
    main()
