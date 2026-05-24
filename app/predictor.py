"""
Model loading and inference for the Streamlit demo.

Supports two model formats (tries new format first):
  NEW (version-independent): xgboost_binary.json + xgboost_binary_meta.npz
  OLD (legacy joblib):       xgboost_binary.joblib  ← may fail on sklearn mismatch
"""
import numpy as np
import xgboost as xgb
import joblib
from pathlib import Path

from src.features import extract_features
from src.preprocessing import get_center_window

MODELS_DIR = Path(__file__).parent.parent / "models"

# ---------------------------------------------------------------------------
# Pre-computed results (from notebooks) — shown even without model files
# ---------------------------------------------------------------------------

BINARY_RESULTS = [
    {"Model": "Deep CNN (from scratch)",            "Accuracy": 0.9935, "F1": 0.9931, "Precision": None,   "Recall": None},
    {"Model": "Transfer Learning (UCI HAR→MobiFall)","Accuracy": 0.9908, "F1": 0.9903, "Precision": None,   "Recall": None},
    {"Model": "XGBoost",                            "Accuracy": 0.9509, "F1": 0.9562, "Precision": 0.9361, "Recall": 0.9771},
    {"Model": "SVM",                                "Accuracy": 0.9452, "F1": 0.9513, "Precision": 0.9269, "Recall": 0.9771},
    {"Model": "Random Forest",                      "Accuracy": 0.9292, "F1": 0.9381, "Precision": 0.9004, "Recall": 0.9792},
    {"Model": "Logistic Regression",                "Accuracy": 0.9007, "F1": 0.9133, "Precision": 0.8757, "Recall": 0.9542},
    {"Model": "Decision Tree",                      "Accuracy": 0.8607, "F1": 0.8806, "Precision": 0.8303, "Recall": 0.9375},
    {"Model": "Naive Bayes",                        "Accuracy": 0.8196, "F1": 0.8364, "Precision": 0.8313, "Recall": 0.8417},
]

FALL4_RESULTS = [
    {"Model": "Transfer Learning (3 s window)",  "Accuracy": 0.6055, "F1": 0.6107},
    {"Model": "Transfer Learning (2 s window)",  "Accuracy": 0.6263, "F1": 0.6169},
    {"Model": "Transfer Learning (1 s window)",  "Accuracy": 0.6183, "F1": 0.6049},
    {"Model": "XGBoost (classical)",             "Accuracy": None,   "F1": None},
    {"Model": "Random Forest (classical)",       "Accuracy": None,   "F1": None},
]

TRANSFER_RESULTS = [
    {"Window": "1 s", "Binary": 0.9304, "Multi-13": 0.6690},
    {"Window": "2 s", "Binary": 0.9651, "Multi-13": 0.7925},
    {"Window": "3 s", "Binary": 0.9908, "Multi-13": 0.8418},
]

FALL_CODE_TO_NAME = {
    0: ("FOL", "Forward Fall"),
    1: ("FKL", "Fall on Knees"),
    2: ("BSC", "Back Stumble"),
    3: ("SDL", "Sideways Fall"),
}

# ---------------------------------------------------------------------------

class FallDetector:
    """
    Loads XGBoost models and runs inference on raw IMU windows.

    Tries the version-independent JSON format first, then falls back to
    the legacy sklearn Pipeline joblib (which may fail on version mismatch).
    """

    def __init__(self):
        self._binary_booster, self._binary_meta = self._load_json("xgboost_binary")
        self._fall4_booster,  self._fall4_meta  = self._load_json("xgboost_fall4")

        # Legacy fallback — only used if JSON files are absent
        if self._binary_booster is None:
            self._binary_pipe = self._load_joblib("xgboost_binary.joblib")
        else:
            self._binary_pipe = None

        if self._fall4_booster is None:
            self._fall4_pipe = self._load_joblib("xgboost_fall4.joblib")
        else:
            self._fall4_pipe = None

    @staticmethod
    def _load_json(name: str):
        json_path = MODELS_DIR / f"{name}.json"
        meta_path = MODELS_DIR / f"{name}_meta.npz"
        if json_path.exists() and meta_path.exists():
            try:
                booster = xgb.Booster()
                booster.load_model(str(json_path))
                meta = np.load(str(meta_path))
                return booster, meta
            except Exception:
                pass
        return None, None

    @staticmethod
    def _load_joblib(filename: str):
        path = MODELS_DIR / filename
        if path.exists():
            try:
                return joblib.load(path)
            except Exception:
                return None
        return None

    def is_loaded(self) -> bool:
        return (self._binary_booster is not None) or (self._binary_pipe is not None)

    def _predict_proba(self, X: np.ndarray, booster, meta, pipe) -> np.ndarray:
        """Apply preprocessing + predict, returning a (1, n_classes) probability array."""
        if booster is not None:
            X_imp = np.where(np.isnan(X), meta["imputer_stats"], X)
            if "scaler_mean" in meta.files:
                X_proc = (X_imp - meta["scaler_mean"]) / meta["scaler_scale"]
            else:
                X_proc = X_imp
            raw = booster.predict(xgb.DMatrix(X_proc))
            # XGBoost output shape varies by version and objective:
            #   binary:logistic  → (n_samples,)       e.g. array([0.97])
            #   multi:softprob   → (n_samples, n_cls)  or flat (n_samples*n_cls,)
            if raw.ndim == 2:
                return raw                              # already (1, n_classes)
            if raw.shape == (1,):
                return np.column_stack([1 - raw, raw]) # binary → (1, 2)
            return raw.reshape(1, -1)                  # flat multiclass → (1, n_classes)
        else:
            return pipe.predict_proba(X)

    def predict(self, signal: np.ndarray, task: str = "binary") -> dict:
        window   = get_center_window(signal, win_size=300)
        features, _ = extract_features(window)
        X = features.reshape(1, -1)

        if task == "binary":
            booster, meta, pipe = self._binary_booster, self._binary_meta, self._binary_pipe
        else:
            booster, meta, pipe = self._fall4_booster, self._fall4_meta, self._fall4_pipe

        if booster is None and pipe is None:
            return {"prediction": -1, "confidence": 0.0, "probabilities": {}}

        proba = self._predict_proba(X, booster, meta, pipe)[0]
        pred  = int(np.argmax(proba))
        conf  = float(proba[pred])

        if task == "binary":
            result = {
                "prediction": pred,
                "confidence": conf,
                "probabilities": {"Normal (ADL)": float(proba[0]), "Fall": float(proba[1])},
            }
            if pred == 1 and (self._fall4_booster is not None or self._fall4_pipe is not None):
                p4 = self._predict_proba(
                    X, self._fall4_booster, self._fall4_meta, self._fall4_pipe
                )[0]
                fc = int(np.argmax(p4))
                result["fall_type_code"], result["fall_type_name"] = FALL_CODE_TO_NAME.get(fc, ("?", "Unknown"))
                result["fall_type_confidence"] = float(p4[fc])
        else:
            probs = {FALL_CODE_TO_NAME[i][0]: float(p) for i, p in enumerate(proba) if i in FALL_CODE_TO_NAME}
            result = {
                "prediction": pred,
                "confidence": conf,
                "probabilities": probs,
                "fall_type_code": FALL_CODE_TO_NAME.get(pred, ("?",))[0],
                "fall_type_name": FALL_CODE_TO_NAME.get(pred, ("?", "Unknown"))[1],
            }

        return result
