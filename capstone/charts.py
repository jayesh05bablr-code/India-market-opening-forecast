"""Market-style charts for the dashboard (Plotly): candlesticks with volume,
moving averages, a range selector, and the next-open forecast marked."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

UP, DOWN = "#16a34a", "#dc2626"   # market convention: green up, red down
MA_COLORS = {20: "#2a78d6", 50: "#eda100", 200: "#6250d6"}
RANGES = [dict(count=1, label="1M", step="month", stepmode="backward"),
          dict(count=3, label="3M", step="month", stepmode="backward"),
          dict(count=6, label="6M", step="month", stepmode="backward"),
          dict(count=1, label="1Y", step="year", stepmode="backward"),
          dict(step="all", label="All")]


def candles(hist: pd.DataFrame, title: str, mas=(20, 50), forecast: dict | None = None,
            levels: dict | None = None, show_days: int = 120) -> go.Figure:
    """hist: date, open, high, low, close, volume. forecast: {date, open, low, high}."""
    h = hist.sort_values("date").copy()
    has_vol = "volume" in h and h["volume"].notna().any()
    fig = make_subplots(rows=2 if has_vol else 1, cols=1, shared_xaxes=True,
                        row_heights=[0.78, 0.22] if has_vol else [1], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=h["date"], open=h["open"], high=h["high"], low=h["low"], close=h["close"],
                                 name="Price", increasing_line_color=UP, decreasing_line_color=DOWN,
                                 increasing_fillcolor=UP, decreasing_fillcolor=DOWN), row=1, col=1)
    for n in mas:
        if len(h) > n:
            fig.add_trace(go.Scatter(x=h["date"], y=h["close"].rolling(n).mean(), name=f"SMA {n}",
                                     line=dict(width=1.5, color=MA_COLORS.get(n))), row=1, col=1)
    if forecast:
        fig.add_trace(go.Scatter(
            x=[forecast["date"]], y=[forecast["open"]], mode="markers", name="Expected open",
            marker=dict(symbol="diamond", size=11, color="#6250d6", line=dict(width=1, color="white")),
            error_y=dict(type="data", symmetric=False, array=[forecast["high"] - forecast["open"]],
                         arrayminus=[forecast["open"] - forecast["low"]], color="#6250d6", width=6),
            hovertemplate="Expected open ₹%{y:,.2f}<extra></extra>"), row=1, col=1)
    for label, (value, color) in (levels or {}).items():
        fig.add_hline(y=value, line=dict(color=color, width=1, dash="dash"), row=1, col=1,
                      annotation_text=f"{label} ₹{value:,.2f}", annotation_position="right")
    if has_vol:
        colors = [UP if c >= o else DOWN for o, c in zip(h["open"], h["close"])]
        fig.add_trace(go.Bar(x=h["date"], y=h["volume"], name="Volume", marker_color=colors,
                             opacity=0.6, showlegend=False), row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1, showgrid=False)
    end = forecast["date"] if forecast else h["date"].max()
    start = h["date"].iloc[max(0, len(h) - show_days)]
    fig.update_xaxes(range=[start, end + pd.Timedelta(days=2)],
                     rangebreaks=[dict(bounds=["sat", "mon"])], showspikes=True, spikemode="across",
                     spikethickness=1, spikedash="dot")
    fig.update_xaxes(rangeselector=dict(buttons=RANGES), row=1, col=1)
    fig.update_yaxes(title_text="₹", row=1, col=1, showspikes=True, spikethickness=1, spikedash="dot")
    fig.update_layout(title=title, height=560, hovermode="x unified", xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=60, b=10),
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"))
    return fig


def line(df: pd.DataFrame, x: str, ys: dict[str, str], title: str, y_title: str,
         show_days: int | None = None) -> go.Figure:
    """Clean line chart with crosshair; ys maps column -> legend name."""
    fig = go.Figure()
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
    for (col, name), color in zip(ys.items(), palette):
        fig.add_trace(go.Scatter(x=df[x], y=df[col], name=name, mode="lines", line=dict(width=2, color=color)))
    if show_days and len(df) > show_days:
        fig.update_xaxes(range=[df[x].iloc[-show_days], df[x].iloc[-1]])
    fig.update_xaxes(rangeselector=dict(buttons=RANGES), showspikes=True, spikemode="across",
                     spikethickness=1, spikedash="dot")
    fig.update_yaxes(title_text=y_title, showspikes=True, spikethickness=1, spikedash="dot")
    fig.update_layout(title=title, height=420, hovermode="x unified", margin=dict(l=10, r=10, t=60, b=10),
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"))
    return fig
