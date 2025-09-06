# -*- coding:utf-8 -*-
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import polars as pl
import torch
from data import utils
from torch.utils.data import Dataset, DataLoader


class SlidingWindowDataset(Dataset):
    def __init__(
        self,
        base_path: str,
        signal_cols: Sequence[str],
        *,
        train: bool = True,
        train_ratio: float = 0.8,
        fs: int = 125,
        window_sec: float = 10.0,
        step_sec: float = 10.0,
    ):
        super().__init__()
        self.base_path = base_path
        self.signal_cols = list(signal_cols)
        self.fs = fs
        self.window_sec = window_sec
        self.step_sec = step_sec
        self.train = train

        self.paths = utils.list_subdirs(base_path)
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No subdirectories under: {base_path}")

        split = int(len(self.paths) * float(train_ratio))
        self.paths_train = self.paths[:split]
        self.paths_test = self.paths[split:]

        target_paths = self.paths_train if train else self.paths_test

        self.data_dict, self.mask_arr = self._load_and_window_all(target_paths)

    def _load_one(self, path: str) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        data_path = os.path.join(path, "data.parquet")
        mask_path = os.path.join(path, "mask.parquet")

        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"Missing data.parquet at: {path}")
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(f"Missing mask.parquet at: {path}")

        data_df = pl.read_parquet(data_path)
        mask_df = pl.read_parquet(mask_path)

        for col in self.signal_cols:
            if col not in data_df.columns:
                raise KeyError(f"Column '{col}' not found in {data_path}. "
                               f"Available: {data_df.columns}")

        if "MASK" not in mask_df.columns:
            raise KeyError(f"'MASK' column not found in {mask_path}. "
                           f"Available: {mask_df.columns}")
        mask_1d = mask_df["MASK"].to_numpy().squeeze()
        signal_1d = {col: data_df[col].to_numpy().squeeze() for col in self.signal_cols}
        return signal_1d, mask_1d

    def _window_one(self, sig_1d: Dict[str, np.ndarray], mask_1d: np.ndarray
                    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:

        win_signals = {
            col: utils.sliding_window_1d(arr, self.fs, self.window_sec, self.step_sec)
            for col, arr in sig_1d.items()
        }
        win_mask = utils.sliding_window_1d(mask_1d, self.fs, self.window_sec, self.step_sec)

        lengths = [v.shape[0] for v in win_signals.values()] + [win_mask.shape[0]]
        n = min(lengths) if lengths else 0
        if n == 0:
            w = int(round(self.window_sec * self.fs))
            win_signals = {k: np.empty((0, w), dtype=float) for k in self.signal_cols}
            win_mask = np.empty((0, w), dtype=float)
        else:
            win_signals = {k: v[:n] for k, v in win_signals.items()}
            win_mask = win_mask[:n]
        return win_signals, win_mask

    def _load_and_window_all(self, paths: Sequence[str]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        data_chunks: Dict[str, List[np.ndarray]] = {k: [] for k in self.signal_cols}
        mask_chunks: List[np.ndarray] = []

        for p in paths:
            sig_1d, m_1d = self._load_one(p)
            sig_w, m_w = self._window_one(sig_1d, m_1d)

            if self.train:
                m_w_expend = np.sum(m_w, axis=-1).astype(np.bool_)
                for k in self.signal_cols:
                    data_chunks[k].append(sig_w[k][m_w_expend])
                mask_chunks.append(m_w[m_w_expend])
            else:
                for k in self.signal_cols:
                    data_chunks[k].append(sig_w[k])
                mask_chunks.append(m_w)

        cat_dict = {k: np.concatenate(v, axis=0) if len(v) else np.empty((0, 0), dtype=float)
                    for k, v in data_chunks.items()}
        cat_mask = np.concatenate(mask_chunks, axis=0) if len(mask_chunks) else np.empty((0, 0), dtype=float)
        return cat_dict, cat_mask

    # ---- Dataset API ----
    def __len__(self) -> int:
        return self.mask_arr.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        data_np = {k: self.data_dict[k][idx] for k in self.signal_cols}
        mask_np = self.mask_arr[idx]

        data = {k: torch.tensor(v, dtype=torch.float32) for k, v in data_np.items()}
        mask = torch.tensor(mask_np, dtype=torch.long)

        # --- #
        mask[mask > 0] = 1
        # --- #
        return data, mask


# -----------------------------------------------------------------------------
# Concrete Datasets
# -----------------------------------------------------------------------------
class HeartbeatDataset(SlidingWindowDataset):
    """MIT-BIH Arrhythmia Database"""
    def __init__(
        self,
        base_path: str = "/data/segmentation/mit_bit",
        train: bool = True,
        fs: int = 125,
        second: float = 30.0,
        sliding_window_sec: float = 30.0,
        train_ratio: float = 0.8,
    ):
        super().__init__(
            base_path=base_path,
            signal_cols=("ECG_1", "ECG_2"),
            train=train,
            train_ratio=train_ratio,
            fs=fs,
            window_sec=second,
            step_sec=sliding_window_sec,
        )


class AHIDataset(SlidingWindowDataset):
    """Sleep Heart Health Study"""
    def __init__(
        self,
        base_path: str = "/data/segmentation/shhs",
        train: bool = True,
        fs: int = 125,
        second: float = 10.0,
        sliding_window_sec: float = 10.0,
        train_ratio: float = 0.8,
    ):
        super().__init__(
            base_path=base_path,
            signal_cols=("AIRFLOW", "THOR RES", "ABDO RES", "SaO2"),
            train=train,
            train_ratio=train_ratio,
            fs=fs,
            window_sec=second,
            step_sec=sliding_window_sec,

        )


if __name__ == "__main__":
    ds = HeartbeatDataset(
        base_path="/data/segmentation/mit_bit",
        train=True,
        fs=125,
        second=5.0,
        sliding_window_sec=5.0,
        train_ratio=0.8,
    )
    print(ds.mask_arr.shape)

    ds2 = HeartbeatDataset(
        base_path="/data/segmentation/mit_bit",
        train=False,
        fs=125,
        second=5.0,
        sliding_window_sec=5.0,
        train_ratio=0.8,
    )
    print(ds2.mask_arr.shape)
    # import matplotlib.pyplot as plt
    # for d in ds:
    #     print(d)
    #     d, m = d
    #     # # plt.title(f'{set(m.numpy())}')
    #     plt.plot(d['ECG_1'])
    #     plt.plot(m)
    #     plt.show()
    #     # exit()
    # # dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
    # # data, mask = next(iter(dl))
    # # print(data)
    # # print(mask)

