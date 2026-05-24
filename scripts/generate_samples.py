"""
Generate synthetic but plausible IMU sample signals for the Streamlit demo.
Each file is 3 seconds at 100 Hz = 300 rows, 6 columns.
Run once: python scripts/generate_samples.py

MobiFall sensor placement: phone in right front thigh pocket, top pointing up.
Axis convention (matches training data statistics):
  acc_y  ≈ primary gravity axis (~9.5 m/s² when upright)
  acc_x  ≈ lateral axis            (~0 m/s² when upright)
  acc_z  ≈ forward/outward axis    (~1-2 m/s² when upright, slight tilt)
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


# ── Fall activities ──────────────────────────────────────────────────────────
# All start upright (acc_y ≈ GRAVITY) then transition to post-fall orientation.

def generate_fol():
    """Forward Fall (FOL): upright → forward lean → chest impact → lying face down."""
    t = np.linspace(0, DURATION, T)
    acc = np.zeros((T, 3), np.float32)
    gyro = np.zeros((T, 3), np.float32)

    # Upright: gravity on Y (phone in right thigh pocket)
    acc[:, 1] = GRAVITY

    # Pre-fall lean forward (0 → 1.4 s): body tilts, Z gains gravity component
    lean_end = int(1.4 * FS)
    acc[:lean_end, 2] += np.linspace(0, 4.0, lean_end)
    acc[:lean_end, 1] -= np.linspace(0, 2.5, lean_end)
    gyro[:lean_end, 0] += np.linspace(0, 3.5, lean_end)   # pitch forward

    # Impact (1.4 → 1.7 s): sharp deceleration spike
    imp_start, imp_end = int(1.4 * FS), int(1.7 * FS)
    imp_len = imp_end - imp_start
    spike = np.exp(-np.linspace(0, 5, imp_len))
    acc[imp_start:imp_end, 1] += 22.0 * spike
    acc[imp_start:imp_end, 2] += 18.0 * spike
    acc[imp_start:imp_end, 0] +=  8.0 * spike
    gyro[imp_start:imp_end, 0] -=  7.0 * spike

    # Post-fall: lying face down — gravity now mostly on Z
    post = imp_end
    acc[post:, 2] = np.linspace(acc[post - 1, 2], GRAVITY * 0.92, T - post)
    acc[post:, 1] = np.linspace(acc[post - 1, 1], 1.5, T - post)
    gyro[post:, 0] = np.linspace(gyro[post - 1, 0], 0, T - post)

    acc  += _noise((T, 3), 0.12)
    gyro += _noise((T, 3), 0.05)
    return np.concatenate([acc, gyro], axis=1)


def generate_fkl():
    """Fall on Knees (FKL): forward stumble → combined knee/torso impact → lying front."""
    acc = np.zeros((T, 3), np.float32)
    gyro = np.zeros((T, 3), np.float32)
    acc[:, 1] = GRAVITY

    # Pre-fall lean forward (0 → 1.4 s)
    lean_end = int(1.4 * FS)
    acc[:lean_end, 2] += np.linspace(0, 4.0, lean_end)
    acc[:lean_end, 1] -= np.linspace(0, 2.5, lean_end)
    gyro[:lean_end, 0] += np.linspace(0, 3.0, lean_end)

    # Single impulsive impact (1.4–1.65 s): knee hits then body falls forward
    imp_start, imp_end = int(1.4 * FS), int(1.65 * FS)
    imp_len = imp_end - imp_start
    spike = np.exp(-np.linspace(0, 5, imp_len))
    acc[imp_start:imp_end, 1] += 24.0 * spike    # primary Y (vertical deceleration)
    acc[imp_start:imp_end, 2] += 14.0 * spike    # Z (forward component)
    acc[imp_start:imp_end, 0] += 10.0 * spike    # X (lateral — provides high kurtosis)
    gyro[imp_start:imp_end, 0] -=  6.0 * spike

    # Post-fall: lying face down, gravity on Z
    post = imp_end
    acc[post:, 2] = np.linspace(acc[post - 1, 2], GRAVITY * 0.90, T - post)
    acc[post:, 1] = np.linspace(acc[post - 1, 1], 1.5, T - post)
    gyro[post:, 0] = np.linspace(gyro[post - 1, 0], 0, T - post)

    acc  += _noise((T, 3), 0.12)
    gyro += _noise((T, 3), 0.05)
    return np.concatenate([acc, gyro], axis=1)


def generate_bsc():
    """Back Stumble (BSC): backward lean → back/buttock impact → lying on back."""
    acc = np.zeros((T, 3), np.float32)
    gyro = np.zeros((T, 3), np.float32)
    acc[:, 1] = GRAVITY

    # Backward lean
    lean_end = int(1.5 * FS)
    acc[:lean_end, 2] -= np.linspace(0, 4.0, lean_end)    # Z decreases (leaning back)
    acc[:lean_end, 1] -= np.linspace(0, 2.0, lean_end)
    gyro[:lean_end, 0] -= np.linspace(0, 4.0, lean_end)   # pitch backward

    # Impact (1.5 → 1.8 s)
    imp_start, imp_end = int(1.5 * FS), int(1.8 * FS)
    imp_len = imp_end - imp_start
    spike = np.exp(-np.linspace(0, 5, imp_len))
    acc[imp_start:imp_end, 1] += 20.0 * spike
    acc[imp_start:imp_end, 2] -= 18.0 * spike             # negative Z spike (backward)
    acc[imp_start:imp_end, 0] +=  5.0 * spike
    gyro[imp_start:imp_end, 0] -=  8.0 * spike

    # Post-fall: lying on back — gravity on -Z
    post = imp_end
    acc[post:, 2] = np.linspace(acc[post - 1, 2], -GRAVITY * 0.92, T - post)
    acc[post:, 1] = np.linspace(acc[post - 1, 1], 1.5, T - post)
    gyro[post:, 0] = np.linspace(gyro[post - 1, 0], 0, T - post)

    acc  += _noise((T, 3), 0.12)
    gyro += _noise((T, 3), 0.05)
    return np.concatenate([acc, gyro], axis=1)


def generate_sdl():
    """Sideways Fall (SDL): lateral lean → side impact → lying on side."""
    acc = np.zeros((T, 3), np.float32)
    gyro = np.zeros((T, 3), np.float32)
    acc[:, 1] = GRAVITY

    # Lateral lean (right side)
    lean_end = int(1.4 * FS)
    acc[:lean_end, 0] += np.linspace(0, 4.5, lean_end)    # X increases (lateral lean)
    acc[:lean_end, 1] -= np.linspace(0, 2.5, lean_end)
    gyro[:lean_end, 2] += np.linspace(0, 4.5, lean_end)   # roll

    # Impact (1.4 → 1.7 s)
    imp_start, imp_end = int(1.4 * FS), int(1.7 * FS)
    imp_len = imp_end - imp_start
    spike = np.exp(-np.linspace(0, 5, imp_len))
    acc[imp_start:imp_end, 0] += 22.0 * spike             # main lateral impact
    acc[imp_start:imp_end, 1] += 16.0 * spike
    acc[imp_start:imp_end, 2] +=  8.0 * spike
    gyro[imp_start:imp_end, 2] +=  9.0 * spike

    # Post-fall: lying on right side — gravity on +X
    post = imp_end
    acc[post:, 0] = np.linspace(acc[post - 1, 0], GRAVITY * 0.90, T - post)
    acc[post:, 1] = np.linspace(acc[post - 1, 1], 1.5, T - post)
    gyro[post:, 2] = np.linspace(gyro[post - 1, 2], 0, T - post)

    acc  += _noise((T, 3), 0.12)
    gyro += _noise((T, 3), 0.05)
    return np.concatenate([acc, gyro], axis=1)


# ── ADL activities ───────────────────────────────────────────────────────────
# Gravity on Y axis; small Z tilt; periodic motion appropriate to activity.

def generate_wal():
    """Walking (WAL): gravity on Y, periodic step oscillation."""
    t = np.linspace(0, DURATION, T)
    freq = 1.85  # Hz (typical walking cadence)
    acc = np.zeros((T, 3), np.float32)
    gyro = np.zeros((T, 3), np.float32)

    # Gravity baseline (Y-axis primary)
    acc[:, 1] = GRAVITY * 0.96 + 2.2 * np.sin(2 * np.pi * freq * t)
    # Forward/backward motion
    acc[:, 2] = 1.5 + 1.0 * np.sin(2 * np.pi * freq * t + np.pi / 4)
    # Lateral weight shift (half freq)
    acc[:, 0] = 0.6 * np.sin(2 * np.pi * freq * 0.5 * t)
    # Gyro: pitch and roll oscillation
    gyro[:, 0] = 1.2 * np.sin(2 * np.pi * freq * t)
    gyro[:, 2] = 0.5 * np.sin(2 * np.pi * freq * 0.5 * t + np.pi / 3)

    acc  += _noise((T, 3), 0.10)
    gyro += _noise((T, 3), 0.04)
    return np.concatenate([acc, gyro], axis=1)


def generate_jog():
    """Jogging (JOG): gravity on Y, higher-amplitude step oscillation."""
    t = np.linspace(0, DURATION, T)
    freq = 2.6  # Hz (typical jogging cadence)
    acc = np.zeros((T, 3), np.float32)
    gyro = np.zeros((T, 3), np.float32)

    acc[:, 1] = GRAVITY * 0.95 + 4.5 * np.sin(2 * np.pi * freq * t)
    acc[:, 2] = 1.8 + 1.8 * np.sin(2 * np.pi * freq * t + np.pi / 4)
    acc[:, 0] = 1.0 * np.sin(2 * np.pi * freq * 0.5 * t)
    gyro[:, 0] = 2.5 * np.sin(2 * np.pi * freq * t)
    gyro[:, 2] = 0.8 * np.sin(2 * np.pi * freq * 0.5 * t + np.pi / 3)

    acc  += _noise((T, 3), 0.12)
    gyro += _noise((T, 3), 0.06)
    return np.concatenate([acc, gyro], axis=1)


def generate_std():
    """Standing still (STD): gravity on Y, small body sway."""
    t = np.linspace(0, DURATION, T)
    acc = np.zeros((T, 3), np.float32)
    gyro = np.zeros((T, 3), np.float32)

    # Gravity on Y; slight forward tilt (phone in pocket)
    acc[:, 1] = GRAVITY * 0.97
    acc[:, 2] = GRAVITY * 0.12
    # Slow postural sway (breathing + balance)
    acc[:, 1] += 0.35 * np.sin(2 * np.pi * 0.22 * t)
    acc[:, 0] += 0.18 * np.sin(2 * np.pi * 0.17 * t + 0.5)
    gyro[:, 0] += 0.15 * np.sin(2 * np.pi * 0.20 * t)

    acc  += _noise((T, 3), 0.08)
    gyro += _noise((T, 3), 0.03)
    return np.concatenate([acc, gyro], axis=1)


def generate_stn():
    """Sitting (STN): gravity on Y (phone stays in pocket, slightly more tilted)."""
    t = np.linspace(0, DURATION, T)
    acc = np.zeros((T, 3), np.float32)
    gyro = np.zeros((T, 3), np.float32)

    # Y is still dominant (phone in pocket); slightly more Z due to hip angle
    # Keep acc_x ≈ 0 so X-kurtosis stays in ADL range
    acc[:, 1] = GRAVITY * 0.94
    acc[:, 2] = GRAVITY * 0.18
    # Slow upper-body sway (breathing, weight shifts)
    acc[:, 1] += 0.45 * np.sin(2 * np.pi * 0.20 * t)
    acc[:, 2] += 0.20 * np.sin(2 * np.pi * 0.23 * t + 0.8)
    gyro[:, 0] += 0.12 * np.sin(2 * np.pi * 0.19 * t)

    acc  += _noise((T, 3), 0.08)
    gyro += _noise((T, 3), 0.03)
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
