"""
Fall Detection & Classification — Streamlit demo
SJSU CS Project | MobiFall Dataset v2.0
"""
import os
import io
import sys
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# Ensure project root is on sys.path regardless of where streamlit is launched from
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="Fall Detection & Classification",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.predictor import FallDetector, BINARY_RESULTS, FALL4_RESULTS, TRANSFER_RESULTS
from app.visualizations import (
    plot_signal,
    plot_model_comparison,
    plot_probability_bars,
    plot_transfer_learning,
)

# ── Visitor analytics (GoatCounter — optional) ──────────────────────────────
_gc_url = os.getenv("GOATCOUNTER_URL", "")
if _gc_url:
    st.components.v1.html(
        f'<script data-goatcounter="{_gc_url}/count" '
        f'async src="//gc.zgo.at/count.js"></script>',
        height=0,
    )

# ── Session state ────────────────────────────────────────────────────────────
if "predictions_made" not in st.session_state:
    st.session_state.predictions_made = 0
if "last_signal" not in st.session_state:
    st.session_state.last_signal = None

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.big-title   { font-size:2.2rem; font-weight:700; color:#667eea; }
.sub-title   { color:#555; margin-top:-0.5rem; margin-bottom:1rem; }
.result-fall {
    background:linear-gradient(135deg,#f5576c,#f093fb);
    color:#fff; padding:1.2rem 2rem; border-radius:10px;
    font-size:1.6rem; font-weight:700; text-align:center; margin:0.5rem 0;
}
.result-safe {
    background:linear-gradient(135deg,#4facfe,#00f2fe);
    color:#fff; padding:1.2rem 2rem; border-radius:10px;
    font-size:1.6rem; font-weight:700; text-align:center; margin:0.5rem 0;
}
.info-card {
    background:#f8f9fa; border-radius:8px; padding:1rem 1.2rem;
    border-left:4px solid #667eea; margin-bottom:0.5rem;
}
.stat-label  { font-size:0.75rem; color:#888; text-transform:uppercase; }
.stat-value  { font-size:1.5rem; font-weight:700; color:#333; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<p class="big-title">🛡️ Fall Detection & Classification</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Detects falls from 6-axis wearable IMU data using three ML pipelines — '
    'Deep CNN (99.35% F1), Transfer Learning from UCI HAR (99.03% F1), and XGBoost with handcrafted features (95.62% F1) — '
    'evaluated on 630 trials across 24 subjects from the MobiFall v2.0 dataset.</p>',
    unsafe_allow_html=True,
)
st.divider()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Controls")

    task_label = st.selectbox(
        "Classification task",
        ["Binary — Fall vs Normal Activity", "Fall Type — 4-class (FOL / FKL / BSC / SDL)"],
        help="Binary: Is this a fall? | Fall Type: Which fall type?",
    )
    is_binary = task_label.startswith("Binary")

    st.divider()
    st.markdown("### Dataset")
    st.markdown(
        "**MobiFall v2.0**  \n"
        "24 subjects · 630 trials  \n"
        "6-axis IMU @ 100 Hz  \n"
        "4 fall types · 9 daily activities"
    )

    st.divider()
    st.markdown("### Best results")
    st.markdown(
        "🏆 Deep CNN: **99.35 %** F1  \n"
        "🔄 Transfer: **99.03 %** F1  \n"
        "🌲 XGBoost:  **95.62 %** F1"
    )

    if st.session_state.predictions_made:
        st.divider()
        st.metric("Predictions this session", st.session_state.predictions_made)


# ── Load model (cached) ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def get_detector():
    return FallDetector()


detector = get_detector()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_demo, tab_results, tab_about = st.tabs(
    ["🔍  Try the Demo", "📊  Results & Analysis", "📖  About the Project"]
)


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Demo
# ════════════════════════════════════════════════════════════════════════════
with tab_demo:
    col_in, col_out = st.columns([1, 1], gap="large")

    SAMPLE_OPTIONS = {
        "⚠️ Forward Fall (FOL)":       ("fol",  True),
        "⚠️ Fall on Knees (FKL)":      ("fkl",  True),
        "⚠️ Back Stumble (BSC)":       ("bsc",  True),
        "⚠️ Sideways Fall (SDL)":      ("sdl",  True),
        "✅ Walking (WAL)":            ("wal",  False),
        "✅ Jogging (JOG)":            ("jog",  False),
        "✅ Standing Still (STD)":     ("std",  False),
        "✅ Sitting (STN)":            ("stn",  False),
    }

    with col_in:
        st.subheader("Input")
        mode = st.radio("Input mode", ["Sample data", "Upload CSV"], horizontal=True)

        signal_data   = None
        signal_label  = "signal"
        expected_fall = None

        if mode == "Sample data":
            choice = st.selectbox("Choose an activity", list(SAMPLE_OPTIONS.keys()))
            code, expected_fall = SAMPLE_OPTIONS[choice]
            sample_path = _ROOT / "sample_data" / f"{code}_sample.csv"
            if sample_path.exists():
                signal_data = pd.read_csv(sample_path).values.astype(np.float32)
                signal_label = choice.split("(")[0].strip().lstrip("⚠️✅ ")
                st.success(f"Loaded **{choice}** — {len(signal_data)} samples @ 100 Hz")
            else:
                st.error(f"Sample file not found: {sample_path}")

        else:
            st.markdown(
                "Upload a CSV with **6 columns**:  \n"
                "`acc_x  acc_y  acc_z  gyro_x  gyro_y  gyro_z`  \n"
                "_(units: m/s² and rad/s, ≥ 100 rows)_"
            )
            uploaded = st.file_uploader("Choose file", type=["csv", "txt"])
            if uploaded:
                try:
                    df = pd.read_csv(io.StringIO(uploaded.read().decode("utf-8")))
                    if df.shape[1] < 6:
                        st.error(f"Need ≥ 6 columns, got {df.shape[1]}.")
                    elif len(df) < 50:
                        st.error("Signal too short (need ≥ 50 rows).")
                    else:
                        signal_data = df.iloc[:, :6].values.astype(np.float32)
                        signal_label = uploaded.name
                        st.success(f"Loaded **{uploaded.name}** — {len(signal_data)} samples")
                except Exception as e:
                    st.error(f"Could not read file: {e}")

    # ── Prediction ──────────────────────────────────────────────────────────
    with col_out:
        st.subheader("Prediction")

        if signal_data is not None:
            if detector.is_loaded():
                result = detector.predict(signal_data, task="binary" if is_binary else "fall4")
                st.session_state.predictions_made += 1

                if is_binary:
                    is_fall = result["prediction"] == 1
                    if is_fall:
                        st.markdown('<div class="result-fall">⚠️ FALL DETECTED</div>',
                                    unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="result-safe">✅ NORMAL ACTIVITY</div>',
                                    unsafe_allow_html=True)

                    conf = result["confidence"]
                    st.metric("Confidence", f"{conf:.1%}")

                    if is_fall and "fall_type_name" in result:
                        st.info(
                            f"**Fall type:** {result['fall_type_name']} "
                            f"({result['fall_type_code']})  \n"
                            f"Confidence: {result.get('fall_type_confidence', 0):.1%}"
                        )

                    if expected_fall is not None:
                        if is_fall == expected_fall:
                            st.success("Matches expected label ✓")
                        else:
                            st.warning("Does not match expected label — synthetic data may vary")

                else:
                    ft_name = result.get("fall_type_name", "Unknown")
                    ft_code = result.get("fall_type_code", "?")
                    st.markdown(f'<div class="result-fall">⚠️ {ft_name} ({ft_code})</div>',
                                unsafe_allow_html=True)
                    st.metric("Confidence", f"{result['confidence']:.1%}")

                # Probability breakdown
                if result.get("probabilities"):
                    st.plotly_chart(
                        plot_probability_bars(result["probabilities"], "Class Probabilities"),
                        use_container_width=True,
                    )

            else:
                st.warning(
                    "**Model files not loaded.**  \n"
                    "See `models/README.md` for download instructions.  \n"
                    "The Results tab still shows pre-computed metrics.",
                    icon="ℹ️",
                )

        else:
            st.info("Select a sample or upload a CSV to run prediction.", icon="👆")

    # ── Signal visualisation ─────────────────────────────────────────────────
    if signal_data is not None:
        st.divider()
        st.subheader("Signal Preview")
        st.plotly_chart(
            plot_signal(signal_data, title=f"IMU Signal — {signal_label}"),
            use_container_width=True,
        )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Results
# ════════════════════════════════════════════════════════════════════════════
with tab_results:
    st.subheader("Model Performance on MobiFall v2.0")

    sub_binary, sub_fall4, sub_transfer = st.tabs(
        ["Binary Classification", "Fall Type (4-class)", "Transfer Learning"]
    )

    # ── Binary ───────────────────────────────────────────────────────────────
    with sub_binary:
        st.markdown("**Task:** Detect whether a 3-second IMU window contains a fall.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Deep CNN (best)", "99.35 % acc", "99.31 % F1")
        c2.metric("Transfer Learning", "99.08 % acc", "99.03 % F1")
        c3.metric("XGBoost (classical)", "95.09 % acc", "95.62 % F1")

        st.plotly_chart(
            plot_model_comparison(
                BINARY_RESULTS,
                "Binary Classification: Fall vs. Normal Activity (MobiFall v2.0)",
            ),
            use_container_width=True,
        )

        df_b = pd.DataFrame(BINARY_RESULTS)
        for col in ["Accuracy", "F1", "Precision", "Recall"]:
            if col in df_b.columns:
                df_b[col] = df_b[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "—")
        st.dataframe(df_b, use_container_width=True, hide_index=True)

    # ── Fall4 ────────────────────────────────────────────────────────────────
    with sub_fall4:
        st.markdown("**Task:** Classify the *type* of fall — FOL, FKL, BSC, or SDL.")

        c1, c2 = st.columns(2)
        c1.metric("Best accuracy", "~62.6 %", "XGBoost / Transfer (2 s)")
        c2.metric("Challenge", "4 classes, 72 trials each")

        st.markdown(
            "Fall-type classification is significantly harder than binary detection.  \n"
            "The four fall types (forward, knees, backward, sideways) share similar "
            "biomechanical trajectories, making inter-class separation difficult with "
            "only 72 trials per class."
        )

        fall4_plot_data = [
            {"Model": "Transfer (3 s)", "Accuracy": 0.6055, "F1": 0.6107},
            {"Model": "Transfer (2 s)", "Accuracy": 0.6263, "F1": 0.6169},
            {"Model": "Transfer (1 s)", "Accuracy": 0.6183, "F1": 0.6049},
        ]
        st.plotly_chart(
            plot_model_comparison(fall4_plot_data, "Fall Type Classification (4-class)"),
            use_container_width=True,
        )

    # ── Transfer ─────────────────────────────────────────────────────────────
    with sub_transfer:
        st.markdown(
            "**Approach:** Pre-train a residual CNN on UCI HAR (6 activities, 7352 windows), "
            "then fine-tune on MobiFall in two stages — frozen backbone first, then full "
            "end-to-end fine-tuning."
        )

        c1, c2 = st.columns(2)
        c1.metric("Pre-training accuracy (UCI HAR)", "94.10 %")
        c2.metric("Best transfer accuracy (binary, 3 s)", "99.08 %")

        st.plotly_chart(
            plot_transfer_learning(TRANSFER_RESULTS),
            use_container_width=True,
        )

        st.markdown("""
        | Stage | Details |
        |---|---|
        | **Stage 1 — frozen** | Only head layers trained; backbone weights locked; 15 epochs |
        | **Stage 2 — full fine-tune** | All layers unfrozen; lower LR (1e-4); up to 40 epochs |
        | **Architecture** | Residual 1D-CNN: 64→128→256 filters, global avg pooling |
        | **Layers transferred** | 25 out of 27 weight-bearing layers |
        """)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — About
# ════════════════════════════════════════════════════════════════════════════
with tab_about:
    col_main, col_side = st.columns([2, 1], gap="large")

    with col_main:
        st.subheader("What this project does")
        st.markdown("""
        This project builds an **IMU-based fall detection system** for wearable devices.
        Falls are one of the leading causes of injury-related death in people over 65.
        Automatic detection enables immediate emergency alerts, potentially saving lives.

        We evaluate three ML pipelines on the **MobiFall Dataset v2.0**:

        #### 1. Classical ML
        Each 3-second IMU window → 112 hand-crafted features (mean, std, RMS, kurtosis,
        zero-crossing rate, …) → trained with XGBoost, SVM, Random Forest, Logistic
        Regression, Naive Bayes, Decision Tree.
        **Best: XGBoost — 95.09 % accuracy, 95.62 % F1.**

        #### 2. Deep CNN (from scratch)
        Raw (T, 6) windows → Residual 1D-CNN with 4 residual blocks (64 → 128 → 256 filters),
        global average pooling, dropout, dense head.  Trained end-to-end on MobiFall.
        **Best: 99.35 % accuracy, 99.31 % F1.**

        #### 3. Transfer Learning (UCI HAR → MobiFall)
        The same CNN architecture is pre-trained on the UCI HAR dataset (7352 training
        windows across 6 activity classes), then fine-tuned on MobiFall in two stages.
        The UCI HAR pre-training acts as a general activity-recognition backbone.
        **Best: 99.08 % accuracy, 99.03 % F1 (3 s window).**
        """)

    with col_side:
        st.subheader("Key numbers")

        stats = [
            ("Dataset", "MobiFall v2.0"),
            ("Subjects", "24"),
            ("Trials", "630"),
            ("Fall types", "FOL · FKL · BSC · SDL"),
            ("IMU channels", "6 (acc + gyro)"),
            ("Sample rate", "100 Hz"),
            ("Window size", "3 seconds"),
            ("Best F1 (binary)", "99.31 %"),
        ]
        for label, val in stats:
            st.markdown(
                f'<div class="info-card">'
                f'<span class="stat-label">{label}</span><br>'
                f'<span class="stat-value">{val}</span></div>',
                unsafe_allow_html=True,
            )

    st.divider()
    st.subheader("Fall Types")
    c1, c2, c3, c4 = st.columns(4)
    descriptions = [
        ("⬆ → ⬇", "FOL", "Forward Fall", "Trips forward; impact on hands/chest"),
        ("⬆ → 🦵", "FKL", "Fall on Knees", "Stumbles forward; knee and body impacts"),
        ("⬆ → ←",  "BSC", "Back Stumble",  "Steps backward; impact on buttocks/back"),
        ("⬆ → ↙",  "SDL", "Sideways Fall", "Loses lateral balance; side impact"),
    ]
    for col, (icon, code, name, desc) in zip([c1, c2, c3, c4], descriptions):
        with col:
            st.markdown(
                f'<div class="info-card"><b>{icon} {name}</b><br>'
                f'<span style="font-size:0.85rem;color:#555">{desc}</span></div>',
                unsafe_allow_html=True,
            )

    st.divider()
    st.subheader("Links & References")
    st.markdown("""
    - **Notebooks:** [View on GitHub](https://github.com/mrunalsuhas/fall-detection/tree/main/notebooks) — full training code on Google Colab
    - **Pre-training source:** [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
    - **Report:** [View Report (PDF)](https://github.com/mrunalsuhas/fall-detection/blob/main/Report.pdf)

    > Vavoulas, G. et al. (2014). *The MobiFall Dataset: Fall Detection and Classification with a Smartphone.*
    """)
