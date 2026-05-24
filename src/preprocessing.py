"""
Utilities for loading and preprocessing IMU signals.
"""
import numpy as np
import pandas as pd
from pathlib import Path


def load_signal_from_csv(path) -> np.ndarray:
    """
    Load a CSV with 6 columns (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z).
    Returns a (T, 6) float32 array. Handles files with or without a header.
    """
    df = pd.read_csv(path)

    # Auto-detect: if first row looks numeric assume no header
    if df.shape[1] >= 6:
        arr = df.iloc[:, :6].to_numpy(dtype=np.float32)
    else:
        raise ValueError(
            f"Expected at least 6 columns, got {df.shape[1]}. "
            "Columns must be: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z"
        )

    if arr.shape[0] < 50:
        raise ValueError(f"Signal too short: {arr.shape[0]} samples (need ≥ 50)")

    return arr


def get_center_window(signal: np.ndarray, win_size: int = 300) -> np.ndarray:
    """Return the center `win_size` samples of a signal, padding if necessary."""
    T = len(signal)
    if T >= win_size:
        start = (T - win_size) // 2
        return signal[start : start + win_size]
    # Pad symmetrically
    pad = win_size - T
    return np.pad(signal, ((pad // 2, pad - pad // 2), (0, 0)), mode="edge")


def normalize_signal(signal: np.ndarray, mean=None, std=None):
    """Channel-wise z-score normalization."""
    if mean is None:
        mean = signal.mean(axis=0)
    if std is None:
        std = signal.std(axis=0) + 1e-8
    return (signal - mean) / std, mean, std
