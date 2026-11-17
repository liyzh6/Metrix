"""PHM 2009 gearbox data loader for the CC experiments."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


NUM_CLASSES = 8
SIGNAL_LENGTH = 1024
CLASS_NAMES = tuple(f"spur {index}" for index in range(1, 9))
CONDITIONS = {
    0: "30hz_High_1", 1: "30hz_High_2", 2: "30hz_Low_1", 3: "30hz_Low_2",
    4: "35hz_High_1", 5: "35hz_High_2", 6: "35hz_Low_1", 7: "35hz_Low_2",
    8: "40hz_High_1", 9: "40hz_High_2", 10: "40hz_Low_1", 11: "40hz_Low_2",
    12: "45hz_High_1", 13: "45hz_High_2", 14: "45hz_Low_1", 15: "45hz_Low_2",
    16: "50hz_High_1", 17: "50hz_High_2", 18: "50hz_Low_1", 19: "50hz_Low_2",
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
        raise ValueError(f"PHM condition must be one of {sorted(CONDITIONS)}, got {condition}")

    samples, labels = [], []
    required_points = samples_per_class * SIGNAL_LENGTH
    for label, class_name in enumerate(CLASS_NAMES):
        stem = f"{class_name}_{CONDITIONS[condition]}"
        path = root / class_name / stem / f"{stem}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"PHM data file not found: {path}")
        values = np.loadtxt(path, usecols=0, max_rows=required_points, dtype=np.float32)
        if values.size < required_points:
            raise ValueError(
                f"{class_name}, condition {condition}: need {required_points} points, "
                f"found {values.size}"
            )
        values = values.reshape(samples_per_class, SIGNAL_LENGTH, 1)
        samples.extend(values)
        labels.extend([label] * samples_per_class)

    return SignalDataset(samples, labels, normalization)


def build_datasets(
    transfer_task,
    num_train_samples=200,
    num_test_samples=50,
    normalization="mean-std",
    data_root="datasets/PHM2009/PHM2009 Gearbox/PHM_Society_2009_Competition_Expanded_txt",
):
    sources, target = transfer_task
    if len(sources) != 3:
        raise ValueError("CC requires exactly three source domains")
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"PHM dataset root not found: {root}")
    datasets = [
        _load_condition(root, int(domain), num_train_samples, normalization)
        for domain in sources
    ]
    datasets.append(_load_condition(root, int(target), num_test_samples, normalization))
    return datasets, NUM_CLASSES
