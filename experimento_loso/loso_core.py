"""Pure, torch-free and data-free helpers for the LOSO protocol.

The functions here encode the leakage rules of the experiment: which subjects and
repetitions each stage may use, and how normalization statistics are computed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

ALL_SUBJECTS: tuple[int, ...] = tuple(range(1, 12))
CONDITIONS: tuple[str, ...] = ("zero", "ft1", "ft2")

_CALIB_REPS = {"zero": (1,), "ft1": (1,), "ft2": (1, 2)}
_FINETUNE_REPS = {"zero": (), "ft1": (1,), "ft2": (1, 2)}


@dataclass(frozen=True)
class FoldPlan:
    """Which subjects and repetitions each stage of one LOSO fold may use."""

    test_subject: int
    condition: str
    train_subjects: tuple[int, ...]
    train_reps: tuple[int, ...]
    val_reps: tuple[int, ...]
    test_reps: tuple[int, ...]
    calib_reps: tuple[int, ...]
    finetune_reps: tuple[int, ...]
    finetune_val_reps: tuple[int, ...]


def fold_plan(
    test_subject: int,
    condition: str,
    all_subjects: Sequence[int] = ALL_SUBJECTS,
) -> FoldPlan:
    """Builds the plan for a held-out subject and adaptation condition.

    Args:
        test_subject: Held-out subject id.
        condition: One of ``CONDITIONS``.
        all_subjects: Pool of subjects in the experiment.

    Returns:
        The immutable fold plan.

    Raises:
        ValueError: If the condition is unknown or the subject is not in the pool.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}; expected one of {CONDITIONS}")
    if test_subject not in all_subjects:
        raise ValueError(f"Subject {test_subject} not in {tuple(all_subjects)}")
    return FoldPlan(
        test_subject=test_subject,
        condition=condition,
        train_subjects=tuple(s for s in all_subjects if s != test_subject),
        train_reps=(1, 2, 3, 4),
        val_reps=(5,),
        test_reps=(6,),
        calib_reps=_CALIB_REPS[condition],
        finetune_reps=_FINETUNE_REPS[condition],
        finetune_val_reps=(5,),
    )


def norm_stats(
    signal: np.ndarray, repetition: np.ndarray, reps: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Computes per-channel mean/std using only samples of the given repetitions.

    Args:
        signal: Array of shape (N, C).
        repetition: Repetition id per sample, shape (N,) or (N, 1).
        reps: Repetitions allowed to contribute to the statistics.

    Returns:
        ``(mean, std)``, each of shape (1, C); std is floored at 1e-8.

    Raises:
        ValueError: If no sample belongs to ``reps``.
    """
    mask = np.isin(np.asarray(repetition).reshape(-1), list(reps))
    if not mask.any():
        raise ValueError(f"No samples found for repetitions {tuple(reps)}")
    selected = signal[mask]
    mean = selected.mean(axis=0, keepdims=True)
    std = selected.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-8)
    return mean, std
