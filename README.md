# Fall Detection & Classification

> **IMU-based fall detection using Deep CNN, Transfer Learning, and Classical ML**  

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-667eea?style=for-the-badge&logo=streamlit)](https://falldetection1.streamlit.app/)
[![Visitors](https://visitor-badge.laobi.icu/badge?page_id=mrunalsuhas.fall-detection)](https://github.com/mrunalsuhas/fall-detection)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)](https://python.org)

---

## What it does

Automatically detects whether a 3-second wearable IMU recording contains a **fall**, and classifies the fall type (forward, backward, sideways, on knees).

Different ML and DL pipelines were evaluated and compared:

| Approach | Accuracy | F1 |
|---|---|---|
| Deep CNN (from scratch) | **99.35 %** | **99.31 %** |
| Transfer Learning (UCI HAR → MobiFall) | **99.08 %** | **99.03 %** |
| XGBoost (classical ML) | 95.09 % | 95.62 % |
| SVM | 94.52 % | 95.13 % |

---

## Dataset

**MobiFall v2.0**: accelerometer + gyroscope recordings from 24 subjects wearing a smartphone.

| Property | Value |
|---|---|
| Subjects | 24 |
| Trials | 630 (288 falls, 342 ADL) |
| Signal | 6-axis IMU @ 100 Hz |
| Fall types | FOL · FKL · BSC · SDL |
| ADL types | 9 (walking, jogging, standing, sitting, …) |

**Fall types:**
- **FOL**: Forward Fall (trips forward, impact on hands/chest)
- **FKL**: Fall on Knees (stumbles forward, knee then body impact)
- **BSC**: Back Stumble (steps backward, buttocks/back impact)
- **SDL**: Sideways Fall (loses lateral balance, side impact)

---

## Architecture

### 1. Classical ML Pipeline
```
Raw IMU signal (T, 6)
  └─ Sliding window (3 s, 50 % overlap)
       └─ Feature extraction: 112 hand-crafted features
            (mean, std, min, max, RMS, MAV, energy, skewness,
             kurtosis, zero-crossing rate — per channel + magnitude)
          └─ StandardScaler → XGBoost / SVM / Random Forest / …
```

### 2. Deep CNN (from scratch)
```
Raw window (300, 6)
  └─ Conv1D(64, k=7) + BN + ReLU
  └─ Conv1D(128, k=5) + BN + ReLU + MaxPool
  └─ ResBlock × 2 (128 filters)  + MaxPool
  └─ ResBlock × 2 (256 filters)  + MaxPool
  └─ ResBlock     (256 filters)
  └─ GlobalAveragePooling1D
  └─ Dense(128) + Dropout(0.3)
  └─ Dense(n_classes)
```

### 3. Transfer Learning
```
Pre-train on UCI HAR (6 activities, 7 352 windows)
  → 94.10 % test accuracy

Fine-tune on MobiFall (two-stage):
  Stage 1 — frozen backbone, only head trained (15 epochs)
  Stage 2 — full fine-tune at 1e-4 LR             (40 epochs)
  → 99.08 % accuracy (binary, 3 s window)
```

---

## Notebooks

| Notebook | Description |
|---|---|
| [`01_mobifall_task3.ipynb`](notebooks/01_mobifall_task3.ipynb) | Deep CNN trained end-to-end on MobiFall |
| [`02_uci_har_pretrain.ipynb`](notebooks/02_uci_har_pretrain.ipynb) | CNN pre-training on UCI HAR dataset |
| [`03_transfer_B_ipynb.ipynb`](notebooks/03_transfer_B_ipynb.ipynb) | Transfer learning fine-tuning + evaluation |
| [`mobifall_task3_Evaluation.ipynb`](notebooks/mobifall_task3_Evaluation.ipynb) | Comprehensive evaluation & confusion matrices |
| [`mobifall_v2_classical_ml_clean.ipynb`](notebooks/mobifall_v2_classical_ml_clean.ipynb) | Classical ML pipeline (XGBoost, SVM, RF, …) |

All notebooks were developed on Google Colab with GPU.

---

## Project Structure

```
fall-detection/
├── streamlit_app.py         ← Streamlit demo app
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── features.py          ← 112-feature signal extractor
│   └── preprocessing.py     ← Signal loading & windowing utilities
│
├── app/
│   ├── predictor.py         ← Model loading + inference
│   └── visualizations.py    ← Plotly chart helpers
│
├── notebooks/               ← Jupyter notebooks (Google Colab)
│   └── (5 notebooks)
│
├── models/
│   ├── README.md            ← Instructions for downloading model files
│   └── *.joblib             ← Trained models (not in git — see README)
│
├── sample_data/             ← Synthetic IMU samples for demo (8 activities)
│   └── *.csv
│
└── scripts/
    ├── export_models.py     ← Train + save XGBoost models from Kaggle data
    └── generate_samples.py  ← Generate synthetic sample CSVs
```

---

## Quick Start

### 1. Clone & install
```bash
git clone https://github.com/YOUR_USERNAME/fall-detection.git
cd fall-detection
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run streamlit_app.py
```
Open [http://localhost:8501](http://localhost:8501)

---

### Deploy your own fork

1. Fork this repo on GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, branch `main`, file `streamlit_app.py`
4. (Optional) Add secret `GOATCOUNTER_URL` for visitor analytics
5. Click **Deploy**: live in ~2 minutes

### Visitor analytics

This project uses **[GoatCounter](https://www.goatcounter.com)**: free, open-source,
no cookie banner needed. Set it up:

1. Sign up at [goatcounter.com](https://www.goatcounter.com) (free)
2. Create a site, get your tracking URL (e.g. `https://yourname.goatcounter.com`)
3. On Streamlit Cloud → App settings → **Secrets**:
   ```toml
   GOATCOUNTER_URL = "https://yourname.goatcounter.com"
   ```

---

## Results

### Binary classification (Fall vs. Normal Activity)

| Model | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|
| Deep CNN (scratch) | 99.35 % | 99.31 % | - | - |
| Transfer Learning (3 s) | 99.08 % | 99.03 % | - | - |
| XGBoost | 95.09 % | 95.62 % | 93.61 % | 97.71 % |
| SVM | 94.52 % | 95.13 % | 92.69 % | 97.71 % |
| Random Forest | 92.92 % | 93.81 % | 90.04 % | 97.92 % |
| Logistic Regression | 90.07 % | 91.33 % | 87.57 % | 95.42 % |

### Transfer learning — accuracy vs window size

| Window | Binary | Multi-13 |
|---|---|---|
| 1 s | 93.04 % | 66.90 % |
| 2 s | 96.51 % | 79.25 % |
| 3 s | **99.08 %** | **84.18 %** |
