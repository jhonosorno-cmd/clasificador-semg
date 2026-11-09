"""F14. Qué explica la diferencia de desempeño entre los 11 sujetos de NinaPro DB3.

Tres paneles, en orden de fuerza explicativa:
  1. Separabilidad de Fisher contra exactitud balanceada — el predictor real.
  2. Contraste gesto/reposo contra proporción de reposo — por qué el reposo parecía la causa.
  3. Antebrazo remanente contra exactitud balanceada — la variable clínica, que solo insinúa.

Fuentes:
  - experimento_multisujeto/resultados/v10_S*_seed42.json (11 corridas propias)
  - experimento_multisujeto/resultados/calidad_senal.json (medición sin entrenar)
  - ninapro.hevs.ch/instructions/DB3.html (tabla clínica oficial)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import style as st

RES = Path(__file__).resolve().parents[2] / "experimento_multisujeto" / "resultados"

ANTEBRAZO: dict[int, int] = {
    1: 50, 2: 70, 3: 30, 4: 40, 5: 90, 6: 40, 7: 0, 8: 50, 9: 90, 10: 50, 11: 90,
}


def load() -> dict[str, np.ndarray | list[int]]:
    """Carga resultados de entrenamiento y medidas de señal, alineados por sujeto."""
    runs = {json.loads(p.read_text(encoding="utf-8"))["subject"]:
            json.loads(p.read_text(encoding="utf-8"))
            for p in RES.glob("v10_S*_seed42.json")}
    calidad = {m["subject"]: m for m in json.loads((RES / "calidad_senal.json").read_text(encoding="utf-8"))}
    sub = sorted(set(runs) & set(calidad))
    return {
        "sub": sub,
        "bal": np.array([runs[s]["balanced_accuracy"] * 100 for s in sub]),
        "rep": np.array([runs[s]["test_class_distribution"][0]
                         / sum(runs[s]["test_class_distribution"]) * 100 for s in sub]),
        "fisher": np.array([calidad[s]["fisher_7_clases"] for s in sub]),
        "contraste": np.array([calidad[s]["contraste_gesto_reposo"] for s in sub]),
        "antebrazo": np.array([ANTEBRAZO[s] for s in sub], dtype=float),
    }


def panel(ax, x, y, sub, highlight, xlabel, ylabel, title, *, pct_x=True, pct_y=True,
          offsets=None, ganador=False) -> tuple[float, float]:
    """Dibuja un panel de dispersión con recta de ajuste, etiquetas y el r anotado."""
    offsets = offsets or {}
    r, p = stats.pearsonr(x, y)
    fit = np.poly1d(np.polyfit(x, y, 1))
    span = (x.max() - x.min()) * 0.06
    xs = np.linspace(x.min() - span, x.max() + span, 50)
    ax.plot(xs, fit(xs), color=st.INK_MUTED, lw=1.2, ls="--", zorder=2)

    color_base = st.MODEL_COLORS["ens"] if ganador else st.MODEL_COLORS["v10"]
    for xi, yi, s in zip(x, y, sub):
        hl = s in highlight
        ax.scatter(xi, yi, s=125 if hl else 65,
                   color=st.HIGHLIGHT if hl else color_base,
                   edgecolor=st.SURFACE, linewidth=1.4, zorder=4)
        dx, dy = offsets.get(s, (0, 11))
        ax.annotate(f"S{s}", (xi, yi), xytext=(dx, dy), textcoords="offset points",
                    ha="center" if dx == 0 else ("left" if dx > 0 else "right"),
                    fontsize=9 if hl else 8.5,
                    color=st.HIGHLIGHT if hl else st.INK_2,
                    fontweight="semibold" if hl else "normal")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=st.INK if ganador else st.INK_2)
    if pct_x:
        st.pct_axis(ax, "x")
    if pct_y:
        st.pct_axis(ax, "y")
    ax.grid(True, axis="both")
    sig = "p < 0,001" if p < 0.001 else f"p = {st.es_number(p, 3)}"
    ax.annotate(f"r = {st.es_number(r, 3)}   ({sig})", xy=(0.04, 0.93), xycoords="axes fraction",
                fontsize=11.5, color=st.INK if ganador else st.INK_2, fontweight="semibold")
    return r, p


def main() -> None:
    d = load()
    sub = d["sub"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9))
    st.apply_style()
    fig.set_facecolor(st.SURFACE)

    r1, _ = panel(
        axes[0], d["fisher"], d["bal"], sub, {1, 7},
        "Separabilidad de las clases (razón de Fisher, 48D)",
        "Exactitud balanceada",
        "Lo que predice el desempeño",
        pct_x=False, ganador=True,
        offsets={5: (0, -18), 6: (13, -3), 11: (-12, -3), 2: (13, 1), 8: (0, -17)},
    )
    r2, _ = panel(
        axes[1], d["contraste"], d["rep"], sub, {1, 7},
        "Contraste de amplitud gesto / reposo",
        "Ventanas de reposo en prueba",
        "Y por qué el reposo lo parecía",
        pct_x=False,
        offsets={7: (0, -18), 6: (-12, -3), 10: (-12, -3), 4: (13, -3), 3: (13, -3)},
    )
    r3, p3 = panel(
        axes[2], d["antebrazo"], d["bal"], sub, {1, 7},
        "Antebrazo remanente (dato clínico de DB3)",
        "Exactitud balanceada",
        "La variable clínica solo insinúa",
        offsets={9: (-11, -3), 11: (-12, -3), 7: (14, -3)},
    )
    axes[2].set_xlim(-8, 100)

    fig.suptitle(
        "Lo que separa a los sujetos es la calidad de su señal  ·  CyborgHybrid v10, semilla 42, 11 sujetos",
        fontsize=12.5, fontweight="semibold", x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    st.source_note(
        fig,
        "Fuente: 11 corridas propias y medición de señal sin entrenar (experimento_multisujeto) · "
        "tabla clínica de ninapro.hevs.ch/instructions/DB3.html · La separabilidad se calcula sobre "
        "las 48 características sin entrenar ningún modelo · quitando S7 del panel derecho, r baja de 0,553 a 0,301.",
    )
    st.save(fig, "fig14_dispersion_entre_sujetos")
    print(f"Fisher-BalAcc r={r1:.3f} | contraste-reposo r={r2:.3f} | antebrazo-BalAcc r={r3:.3f} (p={p3:.3f})")


if __name__ == "__main__":
    main()
