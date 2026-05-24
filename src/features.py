"""
Signal feature extraction for IMU-based fall detection.
Extracts 112 statistical features from a (T, 6) window of
[acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z].
"""
import numpy as np
from scipy.stats import skew, kurtosis, iqr

CHANNEL_NAMES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
FALL_ACTIVITIES = ["FOL", "FKL", "BSC", "SDL"]
ADL_ACTIVITIES  = ["STD", "WAL", "JOG", "JUM", "STU", "STN", "SCH", "CSI", "CSO"]
ALL_ACTIVITIES  = FALL_ACTIVITIES + ADL_ACTIVITIES

FALL_DESCRIPTIONS = {
    "FOL": "Forward Fall",
    "FKL": "Fall on Knees",
    "BSC": "Back Stumble",
    "SDL": "Sideways Fall",
}

ADL_DESCRIPTIONS = {
    "WAL": "Walking",
    "JOG": "Jogging",
    "STD": "Standing",
    "STN": "Sitting (Normal)",
    "STU": "Sitting/Standing Up",
    "JUM": "Jumping",
    "SCH": "Stairs (up)",
    "CSI": "Car Sitting In",
    "CSO": "Car Sitting Out",
}

_STAT_NAMES = [
    "mean", "std", "min", "max", "median",
    "p25", "p75", "iqr", "rms", "mav",
    "energy", "skew", "kurtosis", "zcr",
]


def _zero_crossing_rate(x: np.ndarray) -> float:
    return float(np.mean(np.diff(np.signbit(x)).astype(np.float32)))


def _signal_features_1d(x: np.ndarray) -> list:
    x = np.asarray(x, dtype=np.float32)
    eps = 1e-8
    return [
        float(np.mean(x)),
        float(np.std(x)),
        float(np.min(x)),
        float(np.max(x)),
        float(np.median(x)),
        float(np.percentile(x, 25)),
        float(np.percentile(x, 75)),
        float(iqr(x)),
        float(np.sqrt(np.mean(x ** 2))),
        float(np.mean(np.abs(x))),
        float(np.sum(x ** 2) / len(x)),
        float(skew(x, bias=False)) if float(np.std(x)) > eps else 0.0,
        float(kurtosis(x, bias=False)) if float(np.std(x)) > eps else 0.0,
        _zero_crossing_rate(x),
    ]


def extract_features(window: np.ndarray):
    """
    Extract 112 hand-crafted features from a (T, 6) IMU window.

    Returns
    -------
    features : np.ndarray, shape (112,)
    feature_names : list[str]
    """
    window = np.asarray(window, dtype=np.float32)
    if window.ndim != 2 or window.shape[1] != 6:
        raise ValueError(f"Expected (T, 6) array, got {window.shape}")

    feats, feat_names = [], []

    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        vals = _signal_features_1d(window[:, ch_idx])
        feats.extend(vals)
        feat_names.extend([f"{ch_name}_{n}" for n in _STAT_NAMES])

    acc_mag  = np.linalg.norm(window[:, 0:3], axis=1)
    gyro_mag = np.linalg.norm(window[:, 3:6], axis=1)

    for mag_name, mag_vals in [("acc_mag", acc_mag), ("gyro_mag", gyro_mag)]:
        vals = _signal_features_1d(mag_vals)
        feats.extend(vals)
        feat_names.extend([f"{mag_name}_{n}" for n in _STAT_NAMES])

    return np.array(feats, dtype=np.float32), feat_names
