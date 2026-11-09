"""Aggregate leave-one-subject-out (LOSO) results for the sEMG gesture models.

Reads the per-run JSON metrics and NPZ probability files written by the LOSO
experiment for one seed. For every (subject, condition) where all four member
models have probabilities, it builds an ``ensemble4m`` row (uniform probability
average, then argmax). It then writes per-subject and summary CSVs, runs paired
Wilcoxon tests against the intra-subject v10 baseline, and prints a compact
markdown report to stdout.

Usage:
    python experimento_loso/aggregate.py [--seed 42] [--results-dir PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, f1_score

ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR: str = "experimento_loso/resultados"
BASELINE_DIR: Path = ROOT / "experimento_multisujeto" / "resultados"
BASELINE_MODEL: str = "v10"
BASELINE_SEED: int = 42
MODELS: tuple[str, ...] = ("v10", "v15", "v17", "v18")
ENSEMBLE_NAME: str = "ensemble4m"
CONDITIONS: tuple[str, ...] = ("zero", "ft1", "ft2")
SUBJECTS: tuple[int, ...] = tuple(range(1, 12))
METRICS: tuple[str, ...] = ("accuracy", "balanced_accuracy", "f1_macro")
N_CLASSES: int = 7
MIN_PAIRS: int = 5
EXPECTED_RUNS: int = len(MODELS) * len(SUBJECTS) * len(CONDITIONS)


def resolve_results_dir(raw: str) -> Path:
    """Resolves the results directory against the repository root.

    Args:
        raw: Directory given on the command line. Absolute paths are kept as-is.

    Returns:
        The absolute results directory path.
    """
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _warn(message: str) -> None:
    """Prints a warning to stderr.

    Args:
        message: Warning text.
    """
    print(f"WARNING: {message}", file=sys.stderr)


def load_loso_results(results_dir: Path, seed: int) -> pd.DataFrame:
    """Loads the LOSO JSON metrics for one seed.

    Only top-level JSON files are read; the ``probs`` subdirectory is ignored.
    Values are kept as fractions in 0..1.

    Args:
        results_dir: Directory containing the LOSO JSON files.
        seed: Seed whose runs should be loaded.

    Returns:
        A DataFrame with columns model, condition, subject, seed, accuracy,
        balanced_accuracy and f1_macro. Empty if no valid file was found.
    """
    rows: list[dict[str, object]] = []
    for json_path in sorted(results_dir.glob(f"*_seed{seed}.json")):
        try:
            with json_path.open(encoding="utf-8") as handle:
                data = json.load(handle)
            row: dict[str, object] = {
                "model": str(data["model"]),
                "condition": str(data["condition"]),
                "subject": int(data["subject"]),
                "seed": int(data["seed"]),
                "accuracy": float(data["accuracy"]),
                "balanced_accuracy": float(data["balanced_accuracy"]),
                "f1_macro": float(data["f1_macro"]),
            }
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            _warn(f"skipping {json_path.name}: {exc}")
            continue
        if row["seed"] != seed:
            _warn(f"skipping {json_path.name}: seed field {row['seed']} != {seed}")
            continue
        rows.append(row)
    return pd.DataFrame(
        rows,
        columns=[
            "model", "condition", "subject", "seed",
            "accuracy", "balanced_accuracy", "f1_macro",
        ],
    )


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Loads the probability and label arrays from one NPZ file.

    Args:
        path: Path to the NPZ file.

    Returns:
        A tuple ``(probs, labels)`` with probs as float64.

    Raises:
        KeyError: If the file lacks ``probs`` or ``labels``.
        ValueError: If the file cannot be parsed or pickles are required.
    """
    with np.load(path, allow_pickle=False) as data:
        probs = np.asarray(data["probs"], dtype=np.float64)
        labels = np.asarray(data["labels"])
    return probs, labels


def _classification_metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    """Computes accuracy, balanced accuracy and macro F1 as fractions.

    Args:
        labels: Ground-truth class indices.
        preds: Predicted class indices.

    Returns:
        A dict with keys accuracy, balanced_accuracy and f1_macro.
    """
    return {
        "accuracy": float(np.mean(preds == labels)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "f1_macro": float(
            f1_score(
                labels,
                preds,
                average="macro",
                labels=list(range(N_CLASSES)),
                zero_division=0,
            )
        ),
    }


def build_ensemble_rows(probs_dir: Path, seed: int) -> list[dict[str, object]]:
    """Builds the uniform-average ensemble rows for every complete group.

    A (subject, condition) pair is used only when the NPZ files of all four
    member models exist, their ``labels`` arrays are identical and their
    ``probs`` shapes match with N_CLASSES columns. Otherwise it is skipped.

    Args:
        probs_dir: Directory containing the NPZ probability files.
        seed: Seed of the runs to combine.

    Returns:
        One dict per ensemble row with model, condition, subject, seed and metrics.
    """
    rows: list[dict[str, object]] = []
    for condition in CONDITIONS:
        for subject in SUBJECTS:
            paths = {
                model: probs_dir / f"{model}_S{subject}_{condition}_seed{seed}.npz"
                for model in MODELS
            }
            if not all(path.is_file() for path in paths.values()):
                continue
            try:
                loaded = [_load_npz(path) for path in paths.values()]
            except (OSError, KeyError, ValueError) as exc:
                _warn(f"skipping ensemble S{subject} {condition}: {exc}")
                continue

            labels_ref = loaded[0][1]
            if any(not np.array_equal(labels_ref, labels) for _, labels in loaded[1:]):
                _warn(f"skipping ensemble S{subject} {condition}: labels differ across models")
                continue

            shape_ref = loaded[0][0].shape
            shape_ok = (
                len(shape_ref) == 2
                and shape_ref[1] == N_CLASSES
                and shape_ref[0] == labels_ref.shape[0]
            )
            if not shape_ok or any(probs.shape != shape_ref for probs, _ in loaded):
                _warn(f"skipping ensemble S{subject} {condition}: probs shapes inconsistent")
                continue

            mean_probs = np.mean(np.stack([probs for probs, _ in loaded]), axis=0)
            preds = np.argmax(mean_probs, axis=1)
            row: dict[str, object] = {
                "model": ENSEMBLE_NAME,
                "condition": condition,
                "subject": subject,
                "seed": seed,
            }
            row.update(_classification_metrics(labels_ref, preds))
            rows.append(row)
    return rows


def build_per_subject(df: pd.DataFrame) -> pd.DataFrame:
    """Converts the results to long format with percentage values.

    Args:
        df: Combined results with fractional metric columns.

    Returns:
        A DataFrame with columns model, condition, subject, metric, value_pct.
    """
    long = df.melt(
        id_vars=["model", "condition", "subject"],
        value_vars=list(METRICS),
        var_name="metric",
        value_name="value",
    )
    long["value_pct"] = (long["value"] * 100).round(2)
    long = long[["model", "condition", "subject", "metric", "value_pct"]]
    return long.sort_values(["model", "condition", "subject", "metric"]).reset_index(drop=True)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Computes mean, sample std (ddof=1) and n per (model, condition).

    Args:
        df: Combined results with fractional metric columns.

    Returns:
        A DataFrame with one row per (model, condition). Metric values are in
        percent, rounded to two decimals; std is NaN when n is 1.
    """
    rows: list[dict[str, object]] = []
    for model, condition in sorted(set(zip(df["model"], df["condition"]))):
        group = df[(df["model"] == model) & (df["condition"] == condition)]
        row: dict[str, object] = {
            "model": model,
            "condition": condition,
            "n": int(len(group)),
        }
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=np.float64) * 100
            row[f"{metric}_mean_pct"] = round(float(np.mean(values)), 2)
            if len(values) > 1:
                row[f"{metric}_std_pct"] = round(float(np.std(values, ddof=1)), 2)
            else:
                row[f"{metric}_std_pct"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def load_baseline() -> dict[int, float]:
    """Loads the intra-subject v10 baseline accuracies.

    Returns:
        A mapping from subject id to accuracy as a fraction. Missing or
        unreadable baseline files are skipped with a warning.
    """
    baseline: dict[int, float] = {}
    for subject in SUBJECTS:
        path = BASELINE_DIR / f"{BASELINE_MODEL}_S{subject}_seed{BASELINE_SEED}.json"
        if not path.is_file():
            continue
        try:
            with path.open(encoding="utf-8") as handle:
                baseline[subject] = float(json.load(handle)["accuracy"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            _warn(f"skipping baseline {path.name}: {exc}")
    return baseline


def compare_vs_baseline(df: pd.DataFrame, baseline: dict[int, float]) -> pd.DataFrame:
    """Runs paired Wilcoxon tests of LOSO accuracy against the v10 baseline.

    Compares v10 and ensemble4m for each condition. Pairs are formed by subject,
    using subjects present in both the LOSO results and the baseline. Tests run
    only when at least MIN_PAIRS pairs exist.

    Args:
        df: Combined results with fractional metric columns.
        baseline: Baseline accuracy per subject, as fractions.

    Returns:
        A DataFrame with columns model, condition, n, mean_diff_pp, statistic,
        p_value and note. Skipped rows carry only n and the note.
    """
    rows: list[dict[str, object]] = []
    for model in (BASELINE_MODEL, ENSEMBLE_NAME):
        for condition in CONDITIONS:
            subset = df[(df["model"] == model) & (df["condition"] == condition)]
            loso_by_subject = dict(zip(subset["subject"], subset["accuracy"]))
            shared = sorted(set(loso_by_subject) & set(baseline))
            n = len(shared)
            row: dict[str, object] = {
                "model": model,
                "condition": condition,
                "n": n,
                "mean_diff_pp": float("nan"),
                "statistic": float("nan"),
                "p_value": float("nan"),
                "note": "",
            }
            if n < MIN_PAIRS:
                row["note"] = "n<5, skipped"
                rows.append(row)
                continue

            loso_vals = np.array([loso_by_subject[s] for s in shared], dtype=np.float64)
            base_vals = np.array([baseline[s] for s in shared], dtype=np.float64)
            diffs = loso_vals - base_vals
            row["mean_diff_pp"] = round(float(np.mean(diffs)) * 100, 2)
            try:
                result = stats.wilcoxon(loso_vals, base_vals)
                row["statistic"] = float(result.statistic)
                row["p_value"] = float(result.pvalue)
            except ValueError as exc:
                row["note"] = f"wilcoxon not computed: {exc}"
            rows.append(row)
    return pd.DataFrame(rows)


def _fmt_mean_std(mean: float, std: float) -> str:
    """Formats a mean and std pair as ``mean ± std``.

    Args:
        mean: Mean value in percent.
        std: Sample standard deviation in percent, NaN if undefined.

    Returns:
        The formatted string, with ``-`` in place of an undefined std.
    """
    std_text = "-" if np.isnan(std) else f"{std:.2f}"
    return f"{mean:.2f} ± {std_text}"


def _fmt_optional(value: float, digits: int = 4) -> str:
    """Formats a float, or ``-`` when it is NaN.

    Args:
        value: Value to format.
        digits: Number of decimal places.

    Returns:
        The formatted string.
    """
    return "-" if np.isnan(value) else f"{value:.{digits}f}"


def print_report(summary: pd.DataFrame, comparison: pd.DataFrame, runs_present: int) -> None:
    """Prints the summary and comparison as markdown tables to stdout.

    Args:
        summary: Output of build_summary.
        comparison: Output of compare_vs_baseline.
        runs_present: Number of base-model runs found for the seed.
    """
    print("## LOSO summary (mean ± std, %)")
    print()
    print("| model | condition | n | accuracy | balanced_accuracy | f1_macro |")
    print("|---|---|---|---|---|---|")
    for _, row in summary.iterrows():
        cells = [
            _fmt_mean_std(row[f"{m}_mean_pct"], row[f"{m}_std_pct"]) for m in METRICS
        ]
        print(f"| {row['model']} | {row['condition']} | {row['n']} | " + " | ".join(cells) + " |")
    print()
    print(f"## Paired comparison vs intra-subject {BASELINE_MODEL} (accuracy)")
    print()
    print("| model | condition | n | mean diff (pp) | statistic | p-value | note |")
    print("|---|---|---|---|---|---|---|")
    for _, row in comparison.iterrows():
        print(
            f"| {row['model']} | {row['condition']} | {row['n']} | "
            f"{_fmt_optional(row['mean_diff_pp'], 2)} | "
            f"{_fmt_optional(row['statistic'], 2)} | "
            f"{_fmt_optional(row['p_value'], 4)} | {row['note'] or '-'} |"
        )
    print()
    print(
        f"Runs present: {runs_present} of {EXPECTED_RUNS} expected "
        f"({len(MODELS)} models x {len(SUBJECTS)} subjects x {len(CONDITIONS)} conditions)."
    )


def main(argv: list[str] | None = None) -> int:
    """Runs the LOSO aggregation.

    Args:
        argv: Optional argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code. Always 0, including when there is nothing to aggregate.
    """
    parser = argparse.ArgumentParser(description="Aggregate LOSO results.")
    parser.add_argument("--seed", type=int, default=42, help="Seed to aggregate (default 42).")
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help="LOSO results directory, relative to the repo root (default experimento_loso/resultados).",
    )
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    results_dir = resolve_results_dir(args.results_dir)
    if not results_dir.is_dir():
        print(f"No results directory at {results_dir}. Nothing to aggregate yet.")
        return 0

    df = load_loso_results(results_dir, args.seed)
    if df.empty:
        print(f"No LOSO JSON results for seed {args.seed} in {results_dir}. Nothing to aggregate.")
        return 0

    runs_present = int(
        df[df["model"].isin(MODELS)][["model", "subject", "condition"]].drop_duplicates().shape[0]
    )

    ensemble_rows = build_ensemble_rows(results_dir / "probs", args.seed)
    if ensemble_rows:
        df = pd.concat([df, pd.DataFrame(ensemble_rows)], ignore_index=True)
    else:
        print("No complete 4-model NPZ groups found; ensemble4m skipped.")

    per_subject = build_per_subject(df)
    summary = build_summary(df)
    comparison = compare_vs_baseline(df, load_baseline())

    per_subject_path = results_dir / "summary_per_subject.csv"
    summary_path = results_dir / "summary.csv"
    comparison_path = results_dir / "comparison_vs_intrasubject.csv"
    per_subject.to_csv(per_subject_path, index=False, encoding="utf-8")
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    comparison.to_csv(comparison_path, index=False, encoding="utf-8")

    print_report(summary, comparison, runs_present)
    print()
    print(f"Wrote {per_subject_path}, {summary_path}, {comparison_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
