"""Leakage-focused tests for the pure LOSO helpers."""

from __future__ import annotations

import numpy as np
import pytest

from experimento_loso.loso_core import ALL_SUBJECTS, CONDITIONS, fold_plan, norm_stats


@pytest.mark.parametrize("subject", ALL_SUBJECTS)
@pytest.mark.parametrize("condition", CONDITIONS)
def test_test_subject_never_in_train(subject: int, condition: str) -> None:
    plan = fold_plan(subject, condition)
    assert subject not in plan.train_subjects
    assert set(plan.train_subjects) == set(ALL_SUBJECTS) - {subject}
    assert plan.test_subject == subject


@pytest.mark.parametrize("condition", CONDITIONS)
def test_test_rep_never_used_elsewhere(condition: str) -> None:
    plan = fold_plan(3, condition)
    assert plan.test_reps == (6,)
    for reps in (plan.train_reps, plan.val_reps, plan.calib_reps,
                 plan.finetune_reps, plan.finetune_val_reps):
        assert 6 not in reps


def test_reps_per_condition() -> None:
    assert fold_plan(1, "zero").calib_reps == (1,)
    assert fold_plan(1, "zero").finetune_reps == ()
    assert fold_plan(1, "ft1").calib_reps == (1,)
    assert fold_plan(1, "ft1").finetune_reps == (1,)
    assert fold_plan(1, "ft2").calib_reps == (1, 2)
    assert fold_plan(1, "ft2").finetune_reps == (1, 2)
    plan = fold_plan(1, "ft1")
    assert plan.train_reps == (1, 2, 3, 4)
    assert plan.val_reps == (5,)
    assert plan.finetune_val_reps == (5,)


def test_fold_plan_errors() -> None:
    with pytest.raises(ValueError):
        fold_plan(1, "bogus")
    with pytest.raises(ValueError):
        fold_plan(99, "zero")
    with pytest.raises(ValueError):
        fold_plan(5, "zero", all_subjects=(1, 2, 3))


def test_custom_subject_pool() -> None:
    plan = fold_plan(2, "zero", all_subjects=(1, 2, 3))
    assert plan.train_subjects == (1, 3)


def test_norm_stats_ignores_other_reps() -> None:
    rng = np.random.default_rng(0)
    signal = rng.normal(0.0, 1.0, size=(600, 4))
    rep = np.repeat(np.arange(1, 7), 100)
    clean = signal.copy()
    signal[rep == 6] += 1e6  # huge offset on the test rep
    mean, std = norm_stats(signal, rep, (1,))
    mean_ref, std_ref = norm_stats(clean, rep, (1,))
    np.testing.assert_allclose(mean, mean_ref)
    np.testing.assert_allclose(std, std_ref)
    np.testing.assert_allclose(mean, clean[rep == 1].mean(axis=0, keepdims=True))
    assert mean.shape == (1, 4) and std.shape == (1, 4)


def test_norm_stats_accepts_column_rep_and_multiple_reps() -> None:
    signal = np.arange(40, dtype=float).reshape(20, 2)
    rep = np.repeat([1, 2], 10).reshape(-1, 1)
    mean, _ = norm_stats(signal, rep, [1, 2])
    np.testing.assert_allclose(mean, signal.mean(axis=0, keepdims=True))


def test_norm_stats_std_floor() -> None:
    signal = np.ones((10, 2))
    _, std = norm_stats(signal, np.ones(10), (1,))
    assert np.all(std == 1e-8)


def test_norm_stats_no_match_raises() -> None:
    with pytest.raises(ValueError):
        norm_stats(np.zeros((5, 2)), np.full(5, 3), (1,))
