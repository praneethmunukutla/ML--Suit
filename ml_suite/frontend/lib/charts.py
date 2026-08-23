"""Chart builders for the dashboard.

Colour roles come from a validated palette: one blue for single-series
magnitude, a single-hue blue ramp for continuous magnitude, and a fixed
categorical order that is assigned by slot and never cycled. Dark mode uses
its own steps chosen for the dark surface rather than an automatic inversion.
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

# Categorical slots, in the fixed order they must be assigned.
_CATEGORICAL_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                      "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
_CATEGORICAL_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500",
                     "#d55181", "#008300", "#9085e9", "#e66767"]

# Single-hue blue ramp, light -> dark, for continuous magnitude.
_SEQUENTIAL = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
               "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
               "#184f95", "#104281", "#0d366b"]

_TOKENS = {
    "light": {"primary": "#2a78d6", "text": "#0b0b0b", "muted": "#52514e",
              "grid": "#e6e5e1", "surface": "#ffffff", "good": "#008300",
              "bad": "#e34948"},
    "dark": {"primary": "#3987e5", "text": "#ffffff", "muted": "#c3c2b7",
             "grid": "#333330", "surface": "#0e1117", "good": "#008300",
             "bad": "#e66767"},
}


def _mode() -> str:
    try:
        return "dark" if st.get_option("theme.base") == "dark" else "light"
    except Exception:
        return "light"


def tokens() -> dict:
    return _TOKENS[_mode()]


def categorical(n: int) -> list[str]:
    """Fixed-order slots. Past eight series, fold into 'Other' instead."""
    palette = _CATEGORICAL_DARK if _mode() == "dark" else _CATEGORICAL_LIGHT
    return [palette[i % len(palette)] for i in range(n)]


def _base_layout(fig: go.Figure, height: int, title: str | None = None,
                 legend: bool = False) -> go.Figure:
    """Recessive axes, transparent surface, ink-coloured text.

    `legend=True` reserves a band under the title so a horizontal legend does
    not overlap it.
    """
    t = tokens()
    top = 84 if legend else (44 if title else 12)
    fig.update_layout(
        height=height + (top - 44 if legend else 0),
        margin=dict(l=8, r=16, t=top, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=t["muted"], size=12,
                  family="-apple-system, Segoe UI, Helvetica, sans-serif"),
        title=dict(text=title, font=dict(color=t["text"], size=14),
                   x=0, xanchor="left", yref="container", y=0.97,
                   yanchor="top") if title else None,
        hoverlabel=dict(bgcolor=t["surface"], font_color=t["text"],
                        bordercolor=t["grid"]),
        showlegend=False,
    )
    # automargin lets the axis claim the room its tick labels need; without it
    # category names on horizontal bars are clipped by the tight margins.
    fig.update_xaxes(gridcolor=t["grid"], zerolinecolor=t["grid"],
                     linecolor=t["grid"], tickfont=dict(color=t["muted"]),
                     automargin=True)
    fig.update_yaxes(gridcolor=t["grid"], zerolinecolor=t["grid"],
                     linecolor=t["grid"], tickfont=dict(color=t["muted"]),
                     automargin=True)
    return fig


def leaderboard(entries: list[dict], metric: str) -> go.Figure:
    """Ranked model scores. One series, so no legend — the title names it.
    The winner is the only mark that differs, and it is labelled."""
    ok = [e for e in entries if e.get("status") == "ok"
          and e.get("primary_score") is not None]
    if not ok:
        return _base_layout(go.Figure(), 200, "No successful models")
    ok = sorted(ok, key=lambda e: e["primary_score"])
    t = tokens()
    best_index = len(ok) - 1
    colors = [t["primary"] if i == best_index else _SEQUENTIAL[3]
              for i in range(len(ok))]

    fig = go.Figure(go.Bar(
        x=[e["primary_score"] for e in ok],
        y=[e["model"] for e in ok],
        orientation="h",
        marker=dict(color=colors, cornerradius=4),
        text=[f"{e['primary_score']:.4f}" for e in ok],
        textposition="outside",
        textfont=dict(color=t["muted"], size=11),
        hovertemplate="<b>%{y}</b><br>" + metric + ": %{x:.4f}<extra></extra>",
    ))
    fig.update_xaxes(range=[0, max(e["primary_score"] for e in ok) * 1.18])
    fig = _base_layout(fig, max(220, 42 * len(ok)), f"Model ranking by {metric}")
    fig.update_layout(bargap=0.35)
    return fig


def confusion_matrix(matrix: list[list[int]], labels: list[str]) -> go.Figure:
    """Counts are magnitude, so a single-hue sequential ramp, not a rainbow."""
    t = tokens()
    ramp = _SEQUENTIAL[:8]
    total = sum(sum(row) for row in matrix) or 1
    text = [[f"{v}<br><span style='font-size:10px'>{100*v/total:.1f}%</span>"
             for v in row] for row in matrix]
    peak = max(max(row) for row in matrix) or 1
    fig = go.Figure(go.Heatmap(
        z=matrix, x=labels, y=labels, text=text, texttemplate="%{text}",
        # Capped at the mid-steps of the ramp: the full range runs dark enough
        # that the cell labels lose contrast against the deepest cells.
        colorscale=[[i / (len(ramp) - 1), c] for i, c in enumerate(ramp)],
        showscale=False, xgap=2, ygap=2,
        # Keep label ink readable against both ends of the ramp.
        textfont=dict(size=13, color=t["text"]),
        hovertemplate="predicted <b>%{x}</b><br>actual <b>%{y}</b>"
                      "<br>%{z} rows<extra></extra>",
        zmin=0, zmax=peak,
    ))
    fig = _base_layout(fig, 340, "Confusion matrix")
    fig.update_xaxes(title_text="predicted", side="bottom", showgrid=False)
    fig.update_yaxes(title_text="actual", autorange="reversed", showgrid=False)
    return fig


def predicted_vs_actual(actual: list[float], predicted: list[float]) -> go.Figure:
    """Points against the perfect-prediction diagonal."""
    t = tokens()
    lo, hi = min(*actual, *predicted), max(*actual, *predicted)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines", name="perfect",
        line=dict(color=t["muted"], width=2, dash="dash"),
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=actual, y=predicted, mode="markers", name="predictions",
        marker=dict(color=t["primary"], size=8, opacity=0.65,
                    line=dict(width=2, color=t["surface"])),
        hovertemplate="actual %{x:,.2f}<br>predicted %{y:,.2f}<extra></extra>",
    ))
    fig = _base_layout(fig, 360, "Predicted vs actual")
    fig.update_xaxes(title_text="actual")
    fig.update_yaxes(title_text="predicted")
    return fig


def residuals(predicted: list[float], residual_values: list[float]) -> go.Figure:
    """Residuals against fitted values; structure here means missed signal."""
    t = tokens()
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=t["muted"], width=2, dash="dash"))
    fig.add_trace(go.Scatter(
        x=predicted, y=residual_values, mode="markers",
        marker=dict(color=t["primary"], size=8, opacity=0.65,
                    line=dict(width=2, color=t["surface"])),
        hovertemplate="predicted %{x:,.2f}<br>residual %{y:,.2f}<extra></extra>",
    ))
    fig = _base_layout(fig, 320, "Residuals")
    fig.update_xaxes(title_text="predicted")
    fig.update_yaxes(title_text="actual − predicted")
    return fig


def feature_importance(items: list[dict], top_n: int = 15) -> go.Figure:
    """Single-series magnitude: one hue, ranked, labelled."""
    if not items:
        return _base_layout(go.Figure(), 160, "This model exposes no feature weights")
    top = sorted(items, key=lambda d: d["importance"], reverse=True)[:top_n][::-1]
    t = tokens()
    peak = max(d["importance"] for d in top) or 1
    fig = go.Figure(go.Bar(
        x=[d["importance"] for d in top],
        y=[d["feature"] for d in top],
        orientation="h",
        marker=dict(color=t["primary"], cornerradius=4),
        hovertemplate="<b>%{y}</b><br>weight %{x:.4f}<extra></extra>",
    ))
    fig.update_xaxes(range=[0, peak * 1.05])
    fig = _base_layout(fig, max(240, 30 * len(top)), "Feature importance")
    fig.update_layout(bargap=0.3)
    return fig


def metric_comparison(entries: list[dict], metric_names: list[str]) -> go.Figure:
    """Grouped bars across models. Colour follows the metric, not the rank,
    so filtering models never repaints a series."""
    ok = [e for e in entries if e.get("status") == "ok"]
    if not ok:
        return _base_layout(go.Figure(), 200, "No successful models")
    shown = metric_names[:4]  # past four, the eye cannot track groups
    colors = categorical(len(shown))
    t = tokens()
    fig = go.Figure()
    for name, color in zip(shown, colors):
        fig.add_trace(go.Bar(
            name=name,
            x=[e["model"] for e in ok],
            y=[e["metrics"].get(name) for e in ok],
            marker=dict(color=color, cornerradius=4),
            hovertemplate="<b>%{x}</b><br>" + name + ": %{y:.4f}<extra></extra>",
        ))
    fig = _base_layout(fig, 360, "Metrics by model", legend=True)
    fig.update_layout(
        barmode="group", bargap=0.3, bargroupgap=0.08,
        showlegend=True,  # more than one series, so identity is never colour-alone
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="left", x=0, font=dict(color=t["muted"])),
    )
    return fig


def class_distribution(counts: dict[str, int]) -> go.Figure:
    """Class balance — a single series, so one hue."""
    t = tokens()
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    fig = go.Figure(go.Bar(
        x=[k for k, _ in items], y=[v for _, v in items],
        marker=dict(color=t["primary"], cornerradius=4),
        text=[str(v) for _, v in items], textposition="outside",
        textfont=dict(color=t["muted"], size=11),
        hovertemplate="<b>%{x}</b><br>%{y} rows<extra></extra>",
    ))
    fig = _base_layout(fig, 260, "Target distribution")
    fig.update_layout(bargap=0.4)
    return fig


def missing_values(profiles: list[dict], top_n: int = 12) -> go.Figure:
    """Columns ranked by how much data they are missing."""
    have_missing = [p for p in profiles if p["missing_pct"] > 0]
    if not have_missing:
        return _base_layout(go.Figure(), 140, "No missing values")
    top = sorted(have_missing, key=lambda p: p["missing_pct"],
                 reverse=True)[:top_n][::-1]
    t = tokens()
    fig = go.Figure(go.Bar(
        x=[p["missing_pct"] for p in top], y=[p["name"] for p in top],
        orientation="h",
        marker=dict(color=t["primary"], cornerradius=4),
        text=[f"{p['missing_pct']:.1f}%" for p in top], textposition="outside",
        textfont=dict(color=t["muted"], size=11),
        hovertemplate="<b>%{y}</b><br>%{x:.1f}% missing<extra></extra>",
    ))
    fig.update_xaxes(range=[0, max(p["missing_pct"] for p in top) * 1.2])
    fig = _base_layout(fig, max(200, 32 * len(top)), "Missing values by column")
    fig.update_layout(bargap=0.35)
    return fig
