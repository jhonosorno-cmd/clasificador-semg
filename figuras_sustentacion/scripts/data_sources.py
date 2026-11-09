"""Loaders for every quantitative source used by the presentation figures.

Nothing here computes new experiments. Each function reads a file that already
exists in the project (JSON, NPZ, logs, LaTeX tables) and returns it as plain
Python structures. Derived quantities (e.g. recall per class from a confusion
matrix) are documented next to the loader that produces them.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT: Path = Path(__file__).resolve().parents[2]
LOGS: Path = ROOT / "logs"
LORO: Path = ROOT / "checkpoints" / "loro"
DEMO_CACHE: Path = ROOT / "data" / "demo_cache.npz"
CONFUSION_JSON: Path = ROOT / "results" / "ensemble_confusion.json"
ANEXOS_TEX: Path = ROOT / "latex" / "capitulos" / "anexos.tex"


# ---------------------------------------------------------------------------
# Training logs (per-epoch curves)
# ---------------------------------------------------------------------------

@dataclass
class TrainingCurve:
    """Per-epoch metrics parsed from a training log."""

    epoch: list[int]
    train_loss: list[float]
    train_acc: list[float]
    val_loss: list[float]
    val_acc: list[float]
    val_bal: list[float]

    @property
    def best_epoch(self) -> int:
        """Epoch with the highest validation accuracy (checkpoint criterion)."""
        return self.epoch[int(np.argmax(self.val_acc))]


_TRAIN_RE = re.compile(r"Train Loss:\s*([\d.]+)\s*\|\s*Train Acc:\s*([\d.]+)%")
_VAL_RE = re.compile(
    r"Val\s+Loss:\s*([\d.]+)\s*\|\s*Val Acc:\s*([\d.]+)%\s*\|\s*BalAcc:\s*([\d.]+)%"
)


def parse_training_log(version: str) -> TrainingCurve:
    """Parse ``logs/train_<version>.log`` into per-epoch curves.

    The logs mix tqdm progress (carriage returns) with summary lines. The
    carriage returns are turned into new lines before matching.

    Args:
        version: Log suffix, e.g. "v15".

    Returns:
        Parsed curve; train and val lists have the same length.
    """
    raw = (LOGS / f"train_{version}.log").read_bytes().decode("latin-1")
    lines = raw.replace("\r", "\n").splitlines()
    curve = TrainingCurve([], [], [], [], [], [])
    pending_train: tuple[float, float] | None = None
    for line in lines:
        m_train = _TRAIN_RE.search(line)
        if m_train:
            pending_train = (float(m_train.group(1)), float(m_train.group(2)))
            continue
        m_val = _VAL_RE.search(line)
        if m_val and pending_train is not None:
            curve.epoch.append(len(curve.epoch) + 1)
            curve.train_loss.append(pending_train[0])
            curve.train_acc.append(pending_train[1])
            curve.val_loss.append(float(m_val.group(1)))
            curve.val_acc.append(float(m_val.group(2)))
            curve.val_bal.append(float(m_val.group(3)))
            pending_train = None
    if not curve.epoch:
        raise ValueError(f"No epoch summaries found in train_{version}.log")
    return curve


# ---------------------------------------------------------------------------
# LORO-CV
# ---------------------------------------------------------------------------

LORO_METRICS: tuple[str, ...] = ("accuracy", "precision_macro", "recall_macro", "f1_macro")


def load_loro_summary() -> dict[str, dict[str, list[float]]]:
    """Load per-fold LORO values for v10, v17, v18 and ensemble_4m.

    Returns:
        ``{model: {metric: [6 fold values]}}`` (fractions, not percent).
    """
    data = json.loads((LORO / "loro_cv_summary.json").read_text())
    return {m: {k: v["values"] for k, v in metrics.items()} for m, metrics in data.items()}


def load_loro_folds(model: str) -> dict[str, list[float]]:
    """Load the six ``<model>_fold_k_metrics.json`` files (used for v15).

    Args:
        model: File prefix, e.g. "v15".

    Returns:
        ``{metric: [6 fold values]}`` ordered by fold.
    """
    out: dict[str, list[float]] = {k: [] for k in LORO_METRICS}
    for k in range(1, 7):
        d = json.loads((LORO / f"{model}_fold_{k}_metrics.json").read_text())
        for metric in LORO_METRICS:
            out[metric].append(float(d[metric]))
    return out


# ---------------------------------------------------------------------------
# Demo cache (probabilities, theta sweep, latency, class correlation)
# ---------------------------------------------------------------------------

def load_demo_cache() -> dict[str, np.ndarray]:
    """Load ``data/demo_cache.npz`` produced by ``src.build_demo_cache``."""
    with np.load(DEMO_CACHE) as d:
        return {k: d[k] for k in d.files}


def predict_with_threshold(probs: np.ndarray, theta: float) -> np.ndarray:
    """Decision rule of ``src/eval_threshold.py``.

    Predict rest when P(rest) > theta, otherwise the most likely active class.

    Args:
        probs: (N, 7) softmax probabilities.
        theta: Rest threshold.

    Returns:
        (N,) predicted labels.
    """
    return np.where(probs[:, 0] > theta, 0, probs[:, 1:].argmax(axis=1) + 1)


def per_class_recall(pred: np.ndarray, labels: np.ndarray, n_classes: int = 7) -> np.ndarray:
    """Recall (diagonal of the row-normalized confusion matrix) per class."""
    return np.array([(pred[labels == c] == c).mean() for c in range(n_classes)])


# ---------------------------------------------------------------------------
# Confusion matrix of the Ensemble 4M (argmax, no threshold)
# ---------------------------------------------------------------------------

def load_confusion() -> dict:
    """Load ``results/ensemble_confusion.json`` (counts, percentages, totals)."""
    return json.loads(CONFUSION_JSON.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Ensemble combinations and TTA (evaluation logs)
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """One block of an evaluation log."""

    name: str
    accuracy: float
    balanced_accuracy: float
    per_class: list[float]


_BLOCK_RE = re.compile(
    r"RESULTADOS: (?P<name>[^\n]+?)\s*\n\s*=+\s*\n"
    r"\s*Test Accuracy:\s*(?P<acc>[\d.]+)%\s*\n"
    r"\s*Test Balanced Accuracy:\s*(?P<bal>[\d.]+)%\s*\n"
    r"\s*Accuracy por clase:\s*\n(?P<classes>(?:\s*Clase \d: [\d.]+%[^\n]*\n){7})"
)


def parse_eval_log(name: str) -> list[EvalResult]:
    """Parse result blocks from ``logs/<name>.log``.

    Args:
        name: Log stem, e.g. "eval_ensemble_v18" or "eval_tta".

    Returns:
        One ``EvalResult`` per RESULTADOS block, in file order.
    """
    raw = (LOGS / f"{name}.log").read_bytes().decode("latin-1").replace("\r", "\n")
    results: list[EvalResult] = []
    for m in _BLOCK_RE.finditer(raw):
        classes = [float(x) for x in re.findall(r"Clase \d: ([\d.]+)%", m.group("classes"))]
        label = re.sub(r"\s*[^\x00-\x7f]\s*ref.*$", "", m.group("name")).strip()
        results.append(EvalResult(label, float(m.group("acc")), float(m.group("bal")), classes))
    if not results:
        raise ValueError(f"No result blocks parsed from {name}.log")
    return results


# ---------------------------------------------------------------------------
# Full experiment table (thesis annex, Tabla de resultados completos)
# ---------------------------------------------------------------------------

@dataclass
class Experiment:
    """One row of the annex table."""

    version: str
    strategy: str
    accuracy: float | None
    balanced_accuracy: float | None


def _tex_number(cell: str) -> float | None:
    cell = cell.replace("{,}", ".").replace("\\textbf{", "").replace("}", "")
    cell = cell.replace("\\%", "").replace("$\\approx$", "").strip()
    if cell in ("---", "") or cell.endswith(("*", "†", "\\dagger")):
        return None
    try:
        return float(cell)
    except ValueError:
        return None


def load_experiment_table() -> list[Experiment]:
    """Parse the longtable of all experiments from ``anexos.tex``.

    Rows whose metric is on a different task (v22a binary, v22b 6-gesture) are
    marked with * or † in the thesis and are returned with ``None`` metrics.
    """
    tex = ANEXOS_TEX.read_text(encoding="utf-8")
    start = tex.index("\\begin{longtable}")
    end = tex.index("\\end{longtable}")
    rows: list[Experiment] = []
    for line in tex[start:end].splitlines():
        stripped = line.strip()
        if "&" not in line or stripped.startswith("%"):
            continue
        if stripped.startswith("\\") and not stripped.startswith("\\textbf"):
            continue
        cells = [c.strip() for c in line.rstrip("\\ ").split("&")]
        if len(cells) != 4:
            continue
        version = re.sub(r"\\textbf\{([^}]*)\}", r"\1", cells[0]).replace("--", "-")
        strategy = re.sub(r"\\textbf\{([^}]*)\}", r"\1", cells[1])
        strategy = strategy.replace("$\\to$", "→").replace("$\\gamma=", "γ=").replace("$", "")
        strategy = strategy.replace("{,}", ",").replace("\\", "")
        rows.append(Experiment(version, strategy, _tex_number(cells[2]), _tex_number(cells[3])))
    if not rows:
        raise ValueError("No rows parsed from anexos.tex longtable")
    return rows


# ---------------------------------------------------------------------------
# Model size (INFORME_COMPLETO.md §3.3 and GUIA_ESTUDIO.md, verified in audit)
# ---------------------------------------------------------------------------

V10_PARAMS_BY_BLOCK: dict[str, int] = {
    "CNN (conv1–3)": 96_896,
    "BiLSTM": 659_456,
    "MLP features": 14_912,
    "Clasificador": 41_991,
}
MODEL_PARAMS: dict[str, int] = {
    "v10 / v15 (48D)": 813_255,
    "v17 (132D)": 824_007,
    "v18 (Attention)": 1_076_935,
}

# Inference results on the test set (logs/eval_threshold.log, logs/eval_tta.log)
INFERENCE_VARIANTS: list[tuple[str, float, float]] = [
    ("v10 individual", 71.32, 41.92),           # resultados.tex, tab:comparacion_global
    ("Ensemble 4M\nargmax", 72.78, 44.92),      # eval_threshold.log baseline
    ("Ensemble 4M\n+ umbral θ=0,36", 72.88, 44.24),
    ("Ensemble 4M\n+ TTA-7", 72.83, 44.93),     # eval_tta.log
]
PAPER_BASELINE: float = 79.72
