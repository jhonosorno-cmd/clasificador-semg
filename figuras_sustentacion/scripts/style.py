"""Shared visual style for the defense presentation figures.

Palette validated with the dataviz checker (CVD separation, chroma floor,
contrast). Thesis identity colors are kept where they pass; the green/red pair
of the thesis TikZ figures was replaced because it fails deutan separation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

OUT_DIR: Path = Path(__file__).resolve().parents[1]

# Categorical slots, fixed order (never cycled).
MODEL_COLORS: dict[str, str] = {
    "v10": "#00539F",  # thesis uvblue
    "v17": "#D9730D",  # orange (contrast-safe variant of mlporange)
    "v18": "#8E44AD",  # purple
    "ens": "#1B9E77",  # teal, Ensemble 4M
    "v15": "#7F8C8D",  # context gray (ensemble component not in thesis tables)
}
HIGHLIGHT: str = "#C89600"  # thesis uvgold, always paired with a direct label
CONTEXT: str = "#B8BEC4"    # de-emphasized marks
INK: str = "#222222"
INK_2: str = "#555555"
INK_MUTED: str = "#8A8F94"
GRID: str = "#E3E5E8"
SURFACE: str = "#FFFFFF"

MODEL_LABELS: dict[str, str] = {
    "v10": "v10 CyborgHybrid 48D",
    "v17": "v17 CyborgHybrid 132D",
    "v18": "v18 Self-Attention",
    "ens": "Ensemble 4M",
    "v15": "v15 Augmentation",
}

CLASS_SHORT: list[str] = ["Reposo", "G1", "G2", "G3", "G4", "G5", "G6"]
# Gesture names as printed in the graded thesis (resultados.tex, Tabla 5.1).
CLASS_NAMES: list[str] = [
    "Reposo",
    "G1 Pulgar arriba",
    "G2 Ext. índice+medio",
    "G3 Flex. anular+meñique",
    "G4 Pulgar opuesto",
    "G5 Abducción de dedos",
    "G6 Dedos en puño",
]


def apply_style() -> None:
    """Set matplotlib rcParams: sans-serif, recessive axes, hairline grid."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "axes.axisbelow": True,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "text.color": INK,
            "svg.fonttype": "none",
        }
    )


def new_figure(width: float = 8.0, height: float = 4.5) -> Figure:
    """Create a figure with the shared style applied.

    Args:
        width: Figure width in inches.
        height: Figure height in inches.

    Returns:
        A new matplotlib figure.
    """
    apply_style()
    return plt.figure(figsize=(width, height))


def save(fig: Figure, name: str) -> list[Path]:
    """Save a figure as 300 dpi PNG and SVG into the output folder.

    Args:
        fig: Figure to save.
        name: Base file name without extension.

    Returns:
        Paths written.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for ext in ("png", "svg"):
        path = OUT_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.15)
        paths.append(path)
    plt.close(fig)
    return paths


def pct_axis(ax: Axes, axis: str = "y", decimals: int = 0) -> None:
    """Format an axis as percentages with a comma decimal separator.

    Args:
        ax: Target axes.
        axis: "x" or "y".
        decimals: Number of decimals to show.
    """
    fmt = mpl.ticker.FuncFormatter(lambda v, _: es_number(v, decimals) + " %")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


def es_number(value: float, decimals: int = 2) -> str:
    """Format a number with a comma as decimal separator (es-ES style).

    Args:
        value: Number to format.
        decimals: Decimals to keep.

    Returns:
        Formatted string, e.g. 72,88.
    """
    return f"{value:.{decimals}f}".replace(".", ",")


def label_bars(
    ax: Axes,
    bars: Iterable[mpl.patches.Rectangle],
    values: Iterable[float],
    decimals: int = 2,
    suffix: str = " %",
    horizontal: bool = False,
    color: str = INK_2,
    offset: float = 0.0,
) -> None:
    """Write the value at the data end of each bar.

    Args:
        ax: Axes containing the bars.
        bars: Bar patches.
        values: Values to print, aligned with bars.
        decimals: Decimals in the label.
        suffix: Text appended after the number.
        horizontal: True for barh charts.
        color: Text color.
        offset: Extra data-unit offset from the bar end.
    """
    for bar, val in zip(bars, values):
        text = es_number(val, decimals) + suffix
        if horizontal:
            x = bar.get_width() + offset
            y = bar.get_y() + bar.get_height() / 2
            ax.text(x, y, text, va="center", ha="left", fontsize=8.5, color=color)
        else:
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_height() + offset
            ax.text(x, y, text, va="bottom", ha="center", fontsize=8.5, color=color)


def source_note(fig: Figure, text: str) -> None:
    """Add a small provenance note at the bottom-left of the figure.

    Args:
        fig: Target figure.
        text: Note text (file names the data came from).
    """
    fig.text(0.01, -0.02, text, fontsize=7, color=INK_MUTED, ha="left", va="top")
