"""
Train lightweight XGBoost models on MobiFall v2.0 and save as .joblib files.

Requirements:
    pip install kagglehub xgboost scikit-learn scipy joblib tqdm

Usage:
    # Set Kaggle credentials first (kaggle.com → Account → Create API Token)
    export KAGGLE_USERNAME=your_username
    export KAGGLE_KEY=your_api_key

    python scripts/export_models.py

Output:
    models/xgboost_binary.joblib   — binary classifier (fall vs ADL)
    models/xgboost_fall4.joblib    — 4-class fall type classifier
    models/feature_names.txt       — list of 112 feature names
"""
import os
import sys
import glob
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

import kagglehub
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier

# Add project root to path so src.* imports work
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features import extract_features, FALL_ACTIVITIES

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

FS        = 100
WIN_SEC   = 3
WIN_SIZE  = WIN_SEC * FS
STRIDE    = WIN_SIZE // 2
MAX_WIN   = 30


# ── Data loading (copied from the classical ML notebook) ───────────────────

def load_mobifall_file(path):
    with open(path, errors="ignore") as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines) if l.strip() == "@DATA") + 1
    rows = []
    for line in lines[start:]:
        line = line.strip().rstrip(",")
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            rows.append([int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])])
        except ValueError:
            continue
    return pd.DataFrame(rows, columns=["t_ns", "x", "y", "z"])


def enumerate_trials(root):
    files = glob.glob(f"{root}/**/*.txt", recursive=True)
    by_key = defaultdict(lambda: {"sensors": set(), "paths": {}})
    pat = re.compile(r"([A-Z]{3})_(acc|gyro|ori)_(\d+)_(\d+)\.txt$")
    for f in files:
        m = pat.match(os.path.basename(f))
        if not m:
            continue
        act, sensor, _, trial = m.groups()
        parts = f.split(os.sep)
        category, subject = parts[-3], parts[-4]
        key = (subject, category, act, int(trial))
        by_key[key]["sensors"].add(sensor)
        by_key[key]["paths"][sensor] = f
    trials, paths_map = [], {}
    for k, info in by_key.items():
        if {"acc", "gyro"}.issubset(info["sensors"]):
            trials.append(k)
            paths_map[k] = info["paths"]
    trials.sort(key=lambda t: (int(t[0][3:]), t[1], t[2], t[3]))
    return trials, paths_map


def load_trial(paths, target_hz=100):
    acc  = load_mobifall_file(paths["acc"])
    gyro = load_mobifall_file(paths["gyro"])
    t0 = max(acc["t_ns"].iloc[0], gyro["t_ns"].iloc[0])
    t1 = min(acc["t_ns"].iloc[-1], gyro["t_ns"].iloc[-1])
    step_ns = int(1e9 / target_hz)
    grid = np.arange(t0, t1, step_ns)
    def interp(df):
        return np.stack([
            np.interp(grid, df["t_ns"], df["x"]),
            np.interp(grid, df["t_ns"], df["y"]),
            np.interp(grid, df["t_ns"], df["z"]),
        ], axis=1)
    X = np.concatenate([interp(acc), interp(gyro)], axis=1)
    return X.astype(np.float32)


def window_trial(X):
    T = X.shape[0]
    if T < WIN_SIZE:
        return np.empty((0, WIN_SIZE, 6), dtype=np.float32)
    starts = list(range(0, T - WIN_SIZE + 1, STRIDE))
    if len(starts) > MAX_WIN:
        idx = np.linspace(0, len(starts) - 1, MAX_WIN).astype(int)
        starts = [starts[i] for i in idx]
    return np.stack([X[s : s + WIN_SIZE] for s in starts])


def build_dataset(root):
    trials, paths_map = enumerate_trials(root)
    print(f"Found {len(trials)} valid trials")

    X_feat, y_binary, y_fall4, y_subject = [], [], [], []
    feat_names = None
    act_to_idx4 = {a: i for i, a in enumerate(FALL_ACTIVITIES)}

    for key in tqdm(trials, desc="Extracting features"):
        subj, cat, act, _ = key
        try:
            sig = load_trial(paths_map[key])
        except Exception as e:
            print(f"  skip {key}: {e}")
            continue

        windows = window_trial(sig)
        for w in windows:
            feats, names = extract_features(w)
            X_feat.append(feats)
            y_binary.append(1 if cat == "FALLS" else 0)
            y_fall4.append(act_to_idx4.get(act, -1))
            y_subject.append(int(subj[3:]))
            if feat_names is None:
                feat_names = names

    return (
        np.vstack(X_feat).astype(np.float32),
        np.array(y_binary, dtype=np.int64),
        np.array(y_fall4,  dtype=np.int64),
        np.array(y_subject, dtype=np.int64),
        feat_names,
    )


def make_split(X, y, groups, test_size=0.2):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    tr, te = next(gss.split(X, y, groups=groups))
    return tr, te


def build_pipeline(n_classes):
    objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
    params = dict(n_estimators=200, max_depth=6, learning_rate=0.05,
                  subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1)
    if n_classes > 2:
        params["num_class"] = n_classes
        params["eval_metric"] = "mlogloss"
    else:
        params["eval_metric"] = "logloss"
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     XGBClassifier(objective=objective, **params)),
    ])


def main():
    print("Downloading MobiFall v2.0 from Kaggle…")
    path = kagglehub.dataset_download("kmknation/mobifall-dataset-v20")
    root = os.path.join(path, "MobiFall_Dataset_v2.0")
    if not os.path.exists(root):
        # Try one level up
        for entry in os.listdir(path):
            candidate = os.path.join(path, entry)
            if os.path.isdir(candidate):
                root = candidate
                break
    print(f"Dataset root: {root}")

    X, y_bin, y_fall4, subjects, feat_names = build_dataset(root)
    print(f"Dataset: {X.shape[0]} windows, {X.shape[1]} features")

    # ── Binary classifier ───────────────────────────────────────────────────
    print("\n[1/2] Training binary classifier (fall vs ADL)…")
    tr, te = make_split(X, y_bin, subjects)
    pipe_bin = build_pipeline(n_classes=2)
    pipe_bin.fit(X[tr], y_bin[tr])
    y_pred = pipe_bin.predict(X[te])
    print(f"  Accuracy: {accuracy_score(y_bin[te], y_pred):.4f}  "
          f"F1: {f1_score(y_bin[te], y_pred, average='binary'):.4f}")

    out = MODELS_DIR / "xgboost_binary.joblib"
    joblib.dump(pipe_bin, out, compress=3)
    print(f"  Saved → {out}  ({out.stat().st_size/1e6:.1f} MB)")

    # ── Fall-4 classifier ───────────────────────────────────────────────────
    print("\n[2/2] Training fall-type classifier (4-class)…")
    mask = y_fall4 >= 0
    X4, y4, g4 = X[mask], y_fall4[mask], subjects[mask]
    tr4, te4 = make_split(X4, y4, g4)
    pipe_fall4 = build_pipeline(n_classes=4)
    pipe_fall4.fit(X4[tr4], y4[tr4])
    y_pred4 = pipe_fall4.predict(X4[te4])
    print(f"  Accuracy: {accuracy_score(y4[te4], y_pred4):.4f}  "
          f"F1 (macro): {f1_score(y4[te4], y_pred4, average='macro'):.4f}")
    print(classification_report(y4[te4], y_pred4,
          target_names=FALL_ACTIVITIES, digits=3))

    out4 = MODELS_DIR / "xgboost_fall4.joblib"
    joblib.dump(pipe_fall4, out4, compress=3)
    print(f"  Saved → {out4}  ({out4.stat().st_size/1e6:.1f} MB)")

    # ── Feature names ───────────────────────────────────────────────────────
    feat_path = MODELS_DIR / "feature_names.txt"
    feat_path.write_text("\n".join(feat_names))
    print(f"\nFeature names saved → {feat_path}")
    print("\nAll done. Place the .joblib files in models/ and restart the app.")


if __name__ == "__main__":
    main()
