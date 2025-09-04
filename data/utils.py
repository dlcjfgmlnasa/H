# -*- coding:utf-8 -*-
import os
import numpy as np
from typing import List


def sliding_window_1d(
    data: np.ndarray,
    fs: int,
    window_sec: float,
    step_sec: float,
) -> np.ndarray:
    data = np.asarray(data).squeeze()
    assert data.ndim == 1, f"data must be 1D, got shape={data.shape}"

    window = int(round(window_sec * fs))
    step = int(round(step_sec * fs))
    n = data.shape[0]

    if window <= 0 or step <= 0:
        raise ValueError(f"window/step must be positive. window={window}, step={step}")
    if n < window:
        return np.empty((0, window), dtype=data.dtype)

    num_windows = 1 + (n - window) // step
    sw = np.lib.stride_tricks.sliding_window_view(data, window)[::step]
    sw = sw[:num_windows]
    return sw


def list_subdirs(base_path: str) -> List[str]:
    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"Base path not found or not a directory: {base_path}")
    subdirs = [os.path.join(base_path, d)
               for d in os.listdir(base_path)
               if os.path.isdir(os.path.join(base_path, d))]
    subdirs.sort()
    return subdirs


def get_dataset(
    name: str,
    base_path: str,
    train: bool = True,
    fs: int = 125,
    second: float = 10.0,
    sliding_window_sec: float = 10.0,
    train_ratio: float = 0.8,
):
    name = name.lower()

    if name == "heartbeat":
        from data.data_loader import HeartbeatDataset
        return HeartbeatDataset(
            base_path=base_path,
            train=train,
            fs=fs,
            second=second,
            sliding_window_sec=sliding_window_sec,
            train_ratio=train_ratio,
        )
    elif name == "ahi":
        from data.data_loader import AHIDataset
        return AHIDataset(
            base_path=base_path,
            train=train,
            fs=fs,
            second=second,
            sliding_window_sec=sliding_window_sec,
            train_ratio=train_ratio,
        )
    else:
        raise ValueError(f"Invalid dataset name provided: {name}")
