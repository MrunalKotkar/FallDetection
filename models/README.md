# Model Files

The trained model files are **not committed to git** (they are listed in `.gitignore`).

## What files are expected

| Filename | Task | Algorithm | Size (approx) |
|---|---|---|---|
| `xgboost_binary.joblib` | Fall vs. Normal (binary) | XGBoost | ~5–15 MB |
| `xgboost_fall4.joblib`  | 4-class fall type | XGBoost | ~5–15 MB |
| `rf_binary.joblib`      | Fall vs. Normal (binary) | Random Forest | ~20–50 MB |

At minimum you need `xgboost_binary.joblib` for the demo to run live predictions.

---

## Option A — Export from your Google Drive Colab models

Your Colab sessions saved files to Google Drive under `MyDrive/`. If the saved format
is a `.pkl` (pickle) file you can convert it:

```python
import pickle, joblib

with open("your_model.pkl", "rb") as f:
    model = pickle.load(f)

joblib.dump(model, "models/xgboost_binary.joblib", compress=3)
```

Then place the `.joblib` file in this `models/` directory.

---

## Option B — Re-train from scratch (recommended for fresh export)

Run the export script — it downloads the MobiFall dataset from Kaggle, trains
lightweight XGBoost models, and saves `.joblib` files here:

```bash
# 1. Set Kaggle credentials (get from kaggle.com → Account → API)
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

# 2. Run the export script (takes ~5 min on a laptop)
python scripts/export_models.py
```

On Windows PowerShell:
```powershell
$env:KAGGLE_USERNAME = "your_username"
$env:KAGGLE_KEY      = "your_api_key"
python scripts/export_models.py
```

---

## Option C — Skip live inference for deployment

The Streamlit app works **without** model files. The Results tab shows all
pre-computed metrics, charts, and confusion matrices. The Demo tab displays
a friendly "model not loaded" notice with a link to this file.

This is fine for a portfolio demo — the visualisations and analysis are the
main value, not the inference endpoint.

---

## Committing model files to git

If the `.joblib` files are under 100 MB each, you can commit them directly:

```bash
git add models/*.joblib
git commit -m "add trained XGBoost model files"
```

For larger files (Random Forest can be 50–100 MB) use Git LFS:

```bash
git lfs install
git lfs track "models/*.joblib"
git add .gitattributes models/*.joblib
git commit -m "add model files via Git LFS"
```
