"""Datasets for the LOSO experiment with leakage-free per-subject normalization.

Differences from ``src.data_loader.NinaProDB3Dataset``: the z-score statistics of
each subject are computed only from an explicit set of repetitions (``stats_reps``)
instead of the whole recording. Loading, decimation, band-pass filtering, label
handling, windowing and ``__getitem__`` are otherwise identical to the base.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import scipy.io
from scipy.signal import butter, filtfilt

from experimento_loso.loso_core import norm_stats
from src.config import (
    DOWNSAMPLING_FACTOR,
    RAW_DATA_PATH,
    TARGET_FREQUENCY,
)
from src.data_loader import NinaProDB3Dataset
from src.train_v17_features import NinaProExtendedFeatures


class LOSODataset(NinaProDB3Dataset):
    """NinaPro DB3 dataset whose z-score uses only ``stats_reps[subject]``."""

    def __init__(
        self,
        subjects: list,
        reps_to_use: list,
        stats_reps: Mapping[int, Sequence[int]],
        mode: str = "train",
        augment: bool = False,
        target_subject: int = 1,
    ) -> None:
        """Initializes the dataset.

        Args:
            subjects: Subjects loaded when ``mode == 'train'``.
            reps_to_use: Repetitions that produce windows.
            stats_reps: Subject id -> repetitions used for z-score statistics.
            mode: 'train', 'val' or 'test' (non-train loads only ``target_subject``).
            augment: Online augmentation (train mode only).
            target_subject: Subject loaded for non-train modes.
        """
        self.stats_reps = {int(k): tuple(v) for k, v in stats_reps.items()}
        super().__init__(subjects, reps_to_use, mode=mode, augment=augment,
                         target_subject=target_subject)

    @staticmethod
    def _bandpass(emg_signal: np.ndarray) -> np.ndarray:
        """Applies the base 20-249 Hz Butterworth (order 4) zero-phase filter."""
        nyquist = TARGET_FREQUENCY / 2.0
        b, a = butter(4, [20.0 / nyquist, (nyquist - 1.0) / nyquist], btype="band")
        filtered = np.zeros_like(emg_signal)
        for ch in range(emg_signal.shape[1]):
            filtered[:, ch] = filtfilt(b, a, emg_signal[:, ch])
        return filtered

    def _load_subject_data(self, subject_id: int, reps: Sequence[int]) -> None:
        """Loads one subject, normalizing with its allowed repetitions only."""
        if subject_id not in self.stats_reps:
            raise KeyError(f"stats_reps has no entry for subject {subject_id}")
        file_path = RAW_DATA_PATH / f"S{subject_id}_E1_A1.mat"
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        mat_data = scipy.io.loadmat(str(file_path))
        emg_signal = mat_data["emg"][::DOWNSAMPLING_FACTOR]
        stimulus = mat_data["restimulus"][::DOWNSAMPLING_FACTOR]
        repetition = mat_data["rerepetition"][::DOWNSAMPLING_FACTOR]

        filtered = self._bandpass(emg_signal)
        mean, std = norm_stats(filtered, repetition, self.stats_reps[subject_id])
        emg_signal = ((filtered - mean) / std).astype(np.float32)

        self._filter_and_remap_labels(stimulus)
        self._create_windows(emg_signal, self.filtered_stimulus, repetition, reps)


class LOSOExtendedDataset(LOSODataset, NinaProExtendedFeatures):
    """LOSO dataset with the 132D feature vector used by v17."""
