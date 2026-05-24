"""
Generate synthetic but plausible IMU sample signals for the Streamlit demo.
Each file is 3 seconds at 100 Hz = 300 rows, 6 columns.
Run once: python scripts/generate_samples.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

FS = 100
DURATION = 3.0
T = int(FS * DURATION)
COLUMNS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
GRAVITY = 9.81

rng = np.random.default_rng(42)


def _noise(shape, scale=0.05):
    return rng.normal(0, scale, shape).astype(np.float32)


def _smooth(x, k=5):
    return np.convolve(x, np.ones(k) / k, mode="same")


def generate_fol():
    """Forward Fall (FOL): pre-lean → sharp forward impact → rest on front."""
    t = np.linspace(0, DURATION, T)
    acc = np.zeros((T, 3), np.float32)
    gyro = np.zeros((T, 3), np.float32)

    # Static gravity baseline (upright)
    acc[:, 2] = GRAVITY

    # Pre-fall lean phase (0 → 1.4 s)
    lean_end = int(1.4 * FS)
    acc[:lean_end, 0] += np.linspace(0, 3.0, lean_end)          # forward lean
    acc[:lean_end, 2] -= np.linspace(0, 1.5, lean_end)           # gravity shifting
    gyro[:lean_end, 1] -= np.linspace(0, 4.0, lean_end)          # pitch forward

    # Impact (1.4 → 1.7 s)
    imp_start, imp_end = int(1.4 * FS), int(1.7 * FS)
    imp_len = imp_end - imp_start
    spike = np.exp(-np.linspace(0, 5, imp_len))
    acc[imp_start:imp_end, 0] += 18.0 * spike
    acc[imp_start:imp_end, 2] += 22.0 * spike
    gyro[imp_start:imp_end, 1] -= 8.0 * spike

    # Post-fall (lying face down, gravity shifts to z≈0, x≈+g)
    post_start = imp_end
    acc[post_start:, 0] = np.linspace(acc[post_start - 1, 0], GRAVITY * 0.9, T - post_start)
    acc[post_start:, 2] = np.linspace(acc[post_start - 1, 2], 1.5, T - post_start)
    gyro[post_start:, 1] = np.linspace(gyro[post_start - 1, 1], 0, T - post_start)

    acc  += _noise((T, 3), 0.12)
    gyro += _noise((T, 3), 0.04)
    return np.concatenate([acc, gyro], axis=1)


def generate_fkl():
    """Fall on Knees (FKL): forward stumble → knee impact → body impact."""
    acc = np.zeros((T, 3), np.float32)
    gyro = np.zeros((T, 3), np.float32)
    acc[:, 2] = GRAVITY

    # Pre-fall stumble
    lean_end = int(1.3 * FS)
    acc[:lean_end, 0] += np.linspace(0, 4.0, lean_end)
    gyro[:lean_end, 0] += np.linspace(0, 3.0, lean_end)

    # Knee impact (1.3 s)
    k1 = int(1.3 * FS)
    k1_end = k1 + 12
    knee_spike = np.exp(-np.linspace(0, 4, k1_end - k1))
    acc[k1:k1_end, 2] += 14.0 * knee_spike
    acc[k1:k1_end, 1] += 8.0 * knee_spike

    # Body/hip impact (1.65 s)
    k2 = int(1.65 * FS)
    k2_end = k2 + 15
    body_spike = np.exp(-np.linspace(0, 4, k2_end - k2))
    acc[k2:k2_end, 2] += 18.0 * body_spike
    acc[k2:k2_end, 0] += 12.0 * body_spike

    # Post-fall
    acc[k2_end:, 0] = np.linspace(acc[k2_end - 1, 0], GRAVITY * 0.85, T - k2_end)
    acc[k2_end:, 2] = np.linspace(acc[k2_end - 1, 2], 2.0, T - k2_end)

    acc  += _noise((T, 3), 0.12)
    gyro += _noise((T, 3), 0.04)
    return np.concatenate([acc, gyro], axis=1)


def generate_bsc():
    """Back Stumble (BSC): backward lean → backward impact → lying on back."""
    acc = np.zeros((T, 3), np.float32)
    gyro = np.zeros((T, 3), np.float32)
    acc[:, 2] = GRAVITY

    lean_end = int(1.5 * FS)
    acc[:lean_end, 0] -= np.linspace(0, 3.5, lean_end)          # backward
    gyro[:lean_end, 1] += np.linspace(0, 4.5, lean_end)          # pitch backward

    imp_start, imp_end = int(1.5 * FS), int(1.8 * FS)
    imp_len = imp_end - imp_start
    spike = np.exp(-np.linspace(0, 5, imp_len))
    acc[imp_start:imp_end, 0] -= 16.0 * spike
    acc[imp_start:imp_end, 2] += 20.0 * spike
    gyro[imp_start:imp_end, 1] += 7.0 * spike

    post = imp_end
    acc[post:, 0] = np.linspace(acc[post - 1, 0], -GRAVITY * 0.9, T - post)
    acc[post:, 2] = np.linspace(acc[post - 1, 2], 1.5, T - post)

    acc  += _noise((T, 3), 0.12)
    gyro += _noise((T, 3), 0.04)
    return np.concatenate([acc, gyro], axis=1)


def generate_sdl():
    """Sideways Fall (SDL): lateral lean → side impact → lying on side."""
    acc = np.zeros((T, 3), np.float32)
    gyro = np.zeros((T, 3), np.float32)
    acc[:, 2] = GRAVITY

    lean_end = int(1.4 * FS)
    acc[:lean_end, 1] += np.linspace(0, 4.0, lean_end)           # lateral lean
    gyro[:lean_end, 2] += np.linspace(0, 5.0, lean_end)           # roll

    imp_start, imp_end = int(1.4 * FS), int(1.7 * FS)
    imp_len = imp_end - imp_start
    spike = np.exp(-np.linspace(0, 5, imp_len))
    acc[imp_start:imp_end, 1] += 20.0 * spike
    acc[imp_start:imp_end, 2] += 18.0 * spike
    gyro[imp_start:imp_end, 2] += 9.0 * spike

    post = imp_end
    acc[post:, 1] = np.linspace(acc[post - 1, 1], GRAVITY * 0.9, T - post)
    acc[post:, 2] = np.linspace(acc[post - 1, 2], 1.5, T - post)

    acc  += _noise((T, 3), 0.12)
    gyro += _noise((T, 3), 0.04)
    return np.concatenate([acc, gyro], axis=1)


def generate_wal():
    """Walking (WAL): regular periodic gait pattern."""
    t = np.linspace(0, DURATION, T)
    freq = 1.85  # Hz
    acc = np.zeros((T, 3), np.float32)
    gyro = np.zeros((T, 3), np.float32)

    # Gravity + vertical gait oscillation
    acc[:, 2] = GRAVITY + 1.4 * np.sin(2 * np.pi * freq * t)
    # Forward/backward arm swing
    acc[:, 0] = 0.5 * np.sin(2 * np.pi * freq * t + np.pi / 4)
    # Lateral weight shift
    acc[:, 1] = 0.3 * np.sin(2 * np.pi * freq * 0.5 * t)
    # Pitch oscillation
    gyro[:, 1] = 0.9 * np.sin(2 * np.pi * freq * t)
    gyro[:, 0] = 0.2 * np.sin(2 * np.pi * freq * t + np.pi / 3)

    acc  += _noise((T, 3), 0.08)
    gyro += _noise((T, 3), 0.03)
    return np.concatenate([acc, gyro], axis=1)


def generate_jog():
    """Jogging (JOG): faster periodic motion with higher amplitude."""
    t = np.linspace(0, DURATION, T)
    freq = 2.6
    acc = np.zeros((T, 3), np.float32)
    gyro = np.zeros((T, 3), np.float32)

    acc[:, 2] = GRAVITY + 3.0 * np.sin(2 * np.pi * freq * t)
    acc[:, 0] = 1.2 * np.sin(2 * np.pi * freq * t + np.pi / 4)
    acc[:, 1] = 0.5 * np.sin(2 * np.pi * freq * 0.5 * t)
    gyro[:, 1] = 2.0 * np.sin(2 * np.pi * freq * t)
    gyro[:, 0] = 0.4 * np.sin(2 * np.pi * freq * t + np.pi / 3)

    acc  += _noise((T, 3), 0.1)
    gyro += _noise((T, 3), 0.05)
    return np.concatenate([acc, gyro], axis=1)


def generate_std():
    """Standing still (STD): minimal motion, gravity on z-axis."""
    acc = np.zeros((T, 3), np.float32)
    gyro = np.zeros((T, 3), np.float32)
    acc[:, 2] = GRAVITY
    acc  += _noise((T, 3), 0.05)
    gyro += _noise((T, 3), 0.02)
    return np.concatenate([acc, gyro], axis=1)


def generate_stn():
    """Sitting (STN): gravity mostly on z, slight tilt on y."""
    acc = np.zeros((T, 3), np.float32)
    gyro = np.zeros((T, 3), np.float32)
    acc[:, 2] = GRAVITY * 0.92
    acc[:, 1] = GRAVITY * 0.15
    acc  += _noise((T, 3), 0.05)
    gyro += _noise((T, 3), 0.02)
    return np.concatenate([acc, gyro], axis=1)


GENERATORS = {
    "fol": generate_fol,
    "fkl": generate_fkl,
    "bsc": generate_bsc,
    "sdl": generate_sdl,
    "wal": generate_wal,
    "jog": generate_jog,
    "std": generate_std,
    "stn": generate_stn,
}

if __name__ == "__main__":
    out_dir = Path(__file__).parent.parent / "sample_data"
    out_dir.mkdir(exist_ok=True)

    for name, fn in GENERATORS.items():
        signal = fn()
        df = pd.DataFrame(signal, columns=COLUMNS)
        path = out_dir / f"{name}_sample.csv"
        df.to_csv(path, index=False, float_format="%.6f")
        print(f"  Wrote {path}  shape={signal.shape}")

    print("Done.")
