"""
Plotly chart helpers for the Streamlit demo.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

_ACC_COLORS  = ["#e74c3c", "#2ecc71", "#3498db"]
_GYRO_COLORS = ["#e67e22", "#9b59b6", "#1abc9c"]
_BRAND       = "#667eea"


def plot_signal(signal: np.ndarray, title: str = "IMU Signal", fs: int = 100) -> go.Figure:
    """Two-row subplot: accelerometer on top, gyroscope below."""
    T = len(signal)
    t = np.arange(T) / fs

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Accelerometer  (m/s²)", "Gyroscope  (rad/s)"),
        shared_xaxes=True,
        vertical_spacing=0.12,
    )

    for i, (color, axis) in enumerate(zip(_ACC_COLORS, ["x", "y", "z"])):
        fig.add_trace(
            go.Scatter(x=t, y=signal[:, i], name=f"acc_{axis}",
                       line=dict(color=color, width=1.5)),
            row=1, col=1,
        )

    # Acceleration magnitude
    acc_mag = np.linalg.norm(signal[:, :3], axis=1)
    fig.add_trace(
        go.Scatter(x=t, y=acc_mag, name="acc_magnitude",
                   line=dict(color="#2c3e50", width=2, dash="dot"),
                   opacity=0.8),
        row=1, col=1,
    )

    for i, (color, axis) in enumerate(zip(_GYRO_COLORS, ["x", "y", "z"])):
        fig.add_trace(
            go.Scatter(x=t, y=signal[:, 3 + i], name=f"gyro_{axis}",
                       line=dict(color=color, width=1.5)),
            row=2, col=1,
        )

    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="m/s²",  row=1, col=1)
    fig.update_yaxes(title_text="rad/s", row=2, col=1)
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        height=480,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        margin=dict(t=60, b=60),
    )
    return fig


def plot_model_comparison(results, title: str = "") -> go.Figure:
    """Grouped bar chart comparing model accuracies and F1 scores."""
    df = pd.DataFrame(results).dropna(subset=["Accuracy"])
    fig = px.bar(
        df,
        x="Model", y=["Accuracy", "F1"] if "F1" in df.columns else ["Accuracy"],
        barmode="group",
        title=title,
        color_discrete_sequence=[_BRAND, "#764ba2"],
        text_auto=".1%",
    )
    fig.update_yaxes(range=[0.5, 1.02], title="Score", tickformat=".0%")
    fig.update_layout(
        height=380,
        legend_title="Metric",
        xaxis_tickangle=-20,
        margin=dict(b=80),
    )
    return fig


def plot_probability_bars(probabilities: dict, title: str = "Class Probabilities") -> go.Figure:
    """Horizontal bar chart for prediction confidence breakdown."""
    labels = list(probabilities.keys())
    values = [probabilities[k] for k in labels]
    colors = [_BRAND if v == max(values) else "#b0bec5" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1%}" for v in values],
        textposition="outside",
    ))
    fig.update_xaxes(range=[0, 1.15], tickformat=".0%", title="Probability")
    fig.update_layout(
        title=title,
        height=max(150, 60 * len(labels)),
        margin=dict(l=10, r=60, t=40, b=20),
    )
    return fig


def plot_transfer_learning(transfer_results: list) -> go.Figure:
    """Line chart: transfer learning accuracy vs window size."""
    df = pd.DataFrame(transfer_results)
    fig = go.Figure()
    for col, color, name in [
        ("Binary", _BRAND, "Binary (Fall vs ADL)"),
        ("Multi-13", "#764ba2", "Multi-13 (all activities)"),
    ]:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["Window"], y=df[col], name=name,
                mode="lines+markers+text",
                text=[f"{v:.1%}" for v in df[col]],
                textposition="top center",
                line=dict(color=color, width=2),
                marker=dict(size=8),
            ))
    fig.update_yaxes(range=[0.5, 1.05], tickformat=".0%", title="Test Accuracy")
    fig.update_xaxes(title="Window Size")
    fig.update_layout(
        title="Transfer Learning (UCI HAR → MobiFall): Accuracy vs Window Size",
        height=350,
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
    )
    return fig
