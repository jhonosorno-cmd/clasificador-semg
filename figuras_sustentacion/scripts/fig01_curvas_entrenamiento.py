"""F1. Training curves (validation vs training) for v15, v17 and v18.

Source: logs/train_v15.log, logs/train_v17.log, logs/train_v18.log.
There is no training log for v10, so the base model cannot be shown.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

import data_sources as ds
import style as st

MODELS: list[tuple[str, str]] = [
    ("v15", "v15 · Augmentation"),
    ("v17", "v17 · Features 132D"),
    ("v18", "v18 · Self-Attention"),
]


def main() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharex="col", sharey="row")
    st.apply_style()
    fig.set_facecolor(st.SURFACE)

    for col, (key, title) in enumerate(MODELS):
        curve = ds.parse_training_log(key)
        color = st.MODEL_COLORS[key]
        best = curve.best_epoch
        best_val = max(curve.val_acc)

        ax = axes[0, col]
        ax.plot(curve.epoch, curve.train_acc, color=color, lw=1.6, ls=(0, (3, 2)), alpha=0.55, label="Entrenamiento")
        ax.plot(curve.epoch, curve.val_acc, color=color, lw=2.0, label="Validación")
        ax.plot(best, best_val, "o", ms=7, color=color, mec=st.SURFACE, mew=1.5)
        ax.annotate(
            f"mejor: {st.es_number(best_val)} %\népoca {best}",
            xy=(best, best_val),
            xytext=(best + 1.5, best_val + 9),
            fontsize=8.5,
            color=st.INK_2,
            arrowprops={"arrowstyle": "-", "color": st.INK_MUTED, "lw": 0.8},
        )
        ax.set_title(title)
        ax.set_ylim(40, 100)
        st.pct_axis(ax)
        if col == 0:
            ax.set_ylabel("Exactitud")
            ax.legend(loc="lower right")

        ax = axes[1, col]
        ax.plot(curve.epoch, curve.train_loss, color=color, lw=1.6, ls=(0, (3, 2)), alpha=0.55, label="Entrenamiento")
        ax.plot(curve.epoch, curve.val_loss, color=color, lw=2.0, label="Validación")
        ax.axvline(best, color=st.INK_MUTED, lw=0.8, ls=":")
        ax.set_xlabel("Época")
        ax.set_ylim(0.0, 1.8)
        if col == 0:
            ax.set_ylabel("Pérdida (cross-entropy)")

    fig.suptitle(
        "Curvas de entrenamiento de los componentes del Ensemble 4M "
        "(early stopping, patience = 15)",
        fontsize=12,
        fontweight="semibold",
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    st.source_note(fig, "Fuente: logs/train_v15.log, logs/train_v17.log, logs/train_v18.log · "
                        "El punto marca la época del checkpoint guardado (máx. val acc). No existe log de v10.")
    st.save(fig, "fig01_curvas_entrenamiento_v15_v17_v18")


if __name__ == "__main__":
    main()
