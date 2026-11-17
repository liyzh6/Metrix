"""Paderborn University bearing data loader for the CC experiments."""

from pathlib import Path

import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset


NUM_CLASSES = 14
SIGNAL_LENGTH = 1024
CLASS_NAMES = (
    "KA04", "KA15", "KA16", "KA22", "KA30", "KI04", "KI14",
    "KI16", "KI17", "KI18", "KI21", "KB27", "KB23", "KB24",
)
CONDITIONS = {
    0: "N09_M07_F10_",
    1: "N15_M01_F10_",
    2: "N15_M07_F04_",
    3: "N15_M07_F10_",
}


def _normalize(signal, normalization):
    signal = signal.astype(np.float32, copy=False)
    if normalization == "mean-std":
        std = float(signal.std())
        return (signal - signal.mean()) / max(std, np.finfo(np.float32).eps)
    if normalization == "-1-1":
        centered = signal - signal.mean()
        span = float(centered.max() - centered.min())
        return 2.0 * (centered - centered.min()) / max(span, np.finfo(np.float32).eps) - 1.0
    raise ValueError(f"Unsupported normalization: {normalization}")


class SignalDataset(Dataset):
    def __init__(self, samples, labels, normalization):
        self.samples = samples
        self.labels = labels
        self.normalization = normalization

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        signal = _normalize(self.samples[index], self.normalization)
        return torch.from_numpy(signal.T.copy()), int(self.labels[index])


def _load_condition(root, condition, samples_per_class, normalization):
    if condition not in CONDITIONS:
        raise ValueError(f"PU condition must be one of {sorted(CONDITIONS)}, got {condition}")

    samples, labels = [], []
    required_points = samples_per_class * SIGNAL_LENGTH
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = root / class_name / class_name
        if not class_dir.is_dir():
            class_dir = root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"PU class directory not found: {class_dir}")

        collected = []
        point_count = 0
        for file_index in range(1, 20):
            variable = f"{CONDITIONS[condition]}{class_name}_{file_index}"
            path = class_dir / f"{variable}.mat"
            if not path.is_file():
                raise FileNotFoundError(f"PU data file not found: {path}")
            mat = scipy.io.loadmat(path)
            if variable not in mat:
                raise KeyError(f"Variable {variable!r} not found in {path}")
            values = np.asarray(mat[variable][0][0][2][0][6][2]).reshape(-1)
            values = values[: SIGNAL_LENGTH * 100]
            collected.append(values)
            point_count += values.size
            if point_count >= required_points:
                break

        values = np.concatenate(collected)
        if values.size < required_points:
            raise ValueError(
                f"{class_name}, condition {condition}: need {required_points} points, "
                f"found {values.size}"
            )
        values = values[:required_points].reshape(samples_per_class, SIGNAL_LENGTH, 1)
        samples.extend(values)
        labels.extend([label] * samples_per_class)

    return SignalDataset(samples, labels, normalization)


def build_datasets(
    transfer_task,
    num_train_samples=200,
    num_test_samples=50,
    normalization="mean-std",
    data_root="datasets/PU",
):
    sources, target = transfer_task
    if len(sources) != 3:
        raise ValueError("CC requires exactly three source domains")
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"PU dataset root not found: {root}")
    datasets = [
        _load_condition(root, int(domain), num_train_samples, normalization)
        for domain in sources
    ]
    datasets.append(_load_condition(root, int(target), num_test_samples, normalization))
    return datasets, NUM_CLASSES
