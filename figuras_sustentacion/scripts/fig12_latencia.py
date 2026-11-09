"""F12. Inference latency of the Ensemble 4M (one 250 ms window, CPU).

Source: data/demo_cache.npz (latency_p50, latency_p95, latency_mean,
latency_std, latency_runs) measured by src/build_demo_cache.py with 20
warm-up runs. Only the summary statistics were stored, not the 200 samples.
The 50 ms reference is the window stride of the project (config.py).
"""

from __future__ import annotations

import data_sources as ds
import style as st

STRIDE_MS: float = 50.0


def main() -> None:
    c = ds.load_demo_cache()
    p50, p95 = float(c["latency_p50"]), float(c["latency_p95"])
    mean, std, runs = float(c["latency_mean"]), float(c["latency_std"]), int(c["latency_runs"])

    fig = st.new_figure(8.5, 3.2)
    ax = fig.add_subplot(111)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    color = st.MODEL_COLORS["ens"]
    ax.barh([0], [p95], height=0.32, color=st.CONTEXT, zorder=2)
    ax.barh([0], [p50], height=0.32, color=color, zorder=3)
    ax.errorbar(mean, 0.42, xerr=std, fmt="o", ms=5, color=st.INK_2, ecolor=st.INK_2, elinewidth=1.2, capsize=4, zorder=4)
    ax.text(mean, 0.62, f"media {st.es_number(mean, 1)} ± {st.es_number(std, 1)} ms", ha="center", fontsize=8.5, color=st.INK_2)
    ax.text(p50 / 2, 0, f"p50 {st.es_number(p50, 1)} ms", ha="center", va="center", fontsize=9, color=st.SURFACE, fontweight="semibold")
    ax.text(p95 + 0.5, 0, f"p95 {st.es_number(p95, 1)} ms", ha="left", va="center", fontsize=9, color=st.INK_2)
    ax.axvline(STRIDE_MS, color=st.HIGHLIGHT, lw=1.2, ls="--")
    ax.text(STRIDE_MS - 0.5, -0.45, f"stride entre ventanas: {int(STRIDE_MS)} ms", ha="right", fontsize=8.5, color=st.HIGHLIGHT)
    ax.set_xlim(0, 60)
    ax.set_ylim(-0.7, 0.9)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Latencia de inferencia por ventana, 4 modelos + promedio (ms, CPU)")
    ax.set_title(f"El Ensemble 4M responde antes de que llegue la siguiente ventana ({runs} corridas)")
    fig.tight_layout()
    st.source_note(fig, "Fuente: data/demo_cache.npz (latency_*), medida en src/build_demo_cache.py tras 20 corridas de calentamiento. "
                        "Solo se guardaron los estadísticos, no la distribución.")
    st.save(fig, "fig12_latencia_ensemble_4m")


if __name__ == "__main__":
    main()
