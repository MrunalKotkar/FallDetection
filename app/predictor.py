"""
Model loading and inference for the Streamlit demo.
Falls back gracefully when model files are absent.
"""
import numpy as np
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
    """Loads trained sklearn models and runs inference on raw IMU windows."""

    def __init__(self):
        self._binary  = self._try_load("xgboost_binary.joblib")
        self._fall4   = self._try_load("xgboost_fall4.joblib")

    @staticmethod
    def _try_load(filename: str):
        path = MODELS_DIR / filename
        if path.exists():
            try:
                return joblib.load(path)
            except Exception:
                return None
        return None

    def is_loaded(self) -> bool:
        return self._binary is not None

    def predict(self, signal: np.ndarray, task: str = "binary") -> dict:
        """
        Parameters
        ----------
        signal : np.ndarray, shape (T, 6)
        task   : "binary" | "fall4"

        Returns dict with keys: prediction, confidence, probabilities
        """
        window = get_center_window(signal, win_size=300)
        features, _ = extract_features(window)
        X = features.reshape(1, -1)

        model = self._binary if task == "binary" else self._fall4
        if model is None:
            return {"prediction": -1, "confidence": 0.0, "probabilities": {}}

        proba = model.predict_proba(X)[0]
        pred  = int(np.argmax(proba))
        conf  = float(proba[pred])

        if task == "binary":
            result = {
                "prediction": pred,
                "confidence": conf,
                "probabilities": {"Normal (ADL)": float(proba[0]), "Fall": float(proba[1])},
            }
            # If fall detected and fall4 model exists, also run fall type classification
            if pred == 1 and self._fall4 is not None:
                p4 = self._fall4.predict_proba(X)[0]
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
