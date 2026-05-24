"""Quick smoke test — run: python _test_app.py"""
import sys, traceback
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

results = []

def check(label, fn):
    try:
        fn()
        results.append(f"  OK  {label}")
    except Exception as e:
        results.append(f"FAIL  {label}\n        {e}\n{traceback.format_exc()}")

# 1. Core imports
check("import numpy",    lambda: __import__("numpy"))
check("import pandas",   lambda: __import__("pandas"))
check("import plotly",   lambda: __import__("plotly"))
check("import sklearn",  lambda: __import__("sklearn"))
check("import xgboost",  lambda: __import__("xgboost"))
check("import scipy",    lambda: __import__("scipy"))
check("import joblib",   lambda: __import__("joblib"))
check("import streamlit",lambda: __import__("streamlit"))

# 2. Project imports
check("src.features",        lambda: __import__("src.features",       fromlist=["*"]))
check("src.preprocessing",   lambda: __import__("src.preprocessing",  fromlist=["*"]))
check("app.predictor",       lambda: __import__("app.predictor",      fromlist=["*"]))
check("app.visualizations",  lambda: __import__("app.visualizations", fromlist=["*"]))

# 3. Feature extraction on sample data
def _test_features():
    import pandas as pd, numpy as np
    from src.features import extract_features
    sig = pd.read_csv(ROOT / "sample_data" / "fol_sample.csv").values.astype("float32")
    feats, names = extract_features(sig[:300])
    assert feats.shape == (112,), f"expected (112,), got {feats.shape}"
check("feature extraction (fol_sample)", _test_features)

# 4. Model loading
def _test_model_load():
    from app.predictor import FallDetector
    det = FallDetector()
    loaded = det.is_loaded()
    print(f"\n        detector.is_loaded() = {loaded}")
    return loaded
check("FallDetector loads", _test_model_load)

# 5. Full prediction if model loaded
def _test_predict():
    import pandas as pd, numpy as np
    from app.predictor import FallDetector
    det = FallDetector()
    if not det.is_loaded():
        print("\n        (skipped — no model files)")
        return
    sig = pd.read_csv(ROOT / "sample_data" / "fol_sample.csv").values.astype("float32")
    result = det.predict(sig, task="binary")
    print(f"\n        result = {result}")
    assert "prediction" in result
    assert "confidence" in result
check("end-to-end prediction", _test_predict)

# 6. Charts render without error
def _test_charts():
    import pandas as pd, numpy as np
    from app.visualizations import plot_signal, plot_model_comparison
    sig = pd.read_csv(ROOT / "sample_data" / "wal_sample.csv").values.astype("float32")
    plot_signal(sig)
    from app.predictor import BINARY_RESULTS
    plot_model_comparison(BINARY_RESULTS)
check("chart rendering", _test_charts)

print("\n=== Test Results ===")
for r in results:
    print(r)
print("\nDone.")
