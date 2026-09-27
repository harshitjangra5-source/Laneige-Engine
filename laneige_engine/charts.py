"""Plotly figures for the web app (validated palette, one y-axis per chart, hover on every mark)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .calendar import short_name

PALETTES = {
    "light": {"s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a", "pos": "#2a78d6", "neg": "#e34948",
              "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
              "band": "rgba(235,104,52,0.14)", "event": "rgba(137,135,129,0.10)", "surface": "#fcfcfb"},
    "dark": {"s1": "#3987e5", "s2": "#d95926", "s3": "#199e70", "pos": "#3987e5", "neg": "#e66767",
             "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
             "band": "rgba(217,89,38,0.20)", "event": "rgba(195,194,183,0.10)", "surface": "#1a1a19"},
}
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _layout(fig: go.Figure, p: dict, title: str | None = None, height: int = 380, legend: bool = True,
            yfmt: str = ",.0f", legend_below: bool = False) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=p["ink"]), x=0, xanchor="left") if title else None,
        height=height, margin=dict(l=8, r=8, t=48 if title else 16, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=p["ink2"], size=12),
        showlegend=legend,
        legend=(dict(orientation="h", yanchor="top", y=-0.08, xanchor="left", x=0, font=dict(color=p["ink2"]))
                if legend_below else
                dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, font=dict(color=p["ink2"]))),
        hoverlabel=dict(font=dict(family=FONT)),
        barcornerradius=4,
    )
    fig.update_xaxes(showgrid=False, linecolor=p["axis"], tickfont=dict(color=p["muted"]), zeroline=False)
    fig.update_yaxes(gridcolor=p["grid"], gridwidth=1, linecolor=p["axis"], tickfont=dict(color=p["muted"]),
                     zeroline=False, tickformat=yfmt, rangemode="tozero")
    return fig


def forecast_chart(history: pd.DataFrame, forecast: pd.DataFrame, bridge: pd.DataFrame, calendar: pd.DataFrame,
                   theme: str = "light", title: str | None = None, history_buckets: int = 104) -> go.Figure:
    """Actuals, forecast with 80% band, last year re-aligned to this year's festival dates, event windows."""
    p = PALETTES[theme]
    h = history.groupby("bucket")["qty"].sum().sort_index().tail(history_buckets)
    f = forecast.groupby("bucket")[["forecast", "p10", "p90"]].sum().sort_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(f.index) + list(f.index[::-1]), y=list(f["p90"]) + list(f["p10"][::-1]),
                             fill="toself", fillcolor=p["band"], line=dict(width=0), hoverinfo="skip",
                             name="80% range", showlegend=True))
    fig.add_trace(go.Scatter(x=h.index, y=h.values, name="Actual sell-out", mode="lines",
                             line=dict(color=p["s1"], width=2), hovertemplate="%{y:,.0f}<extra>Actual</extra>"))
    if not bridge.empty:
        b = bridge.groupby("bucket")["ly_aligned"].sum().sort_index()
        fig.add_trace(go.Scatter(x=b.index, y=b.values, name="Last year, festival-aligned", mode="lines",
                                 line=dict(color=p["s3"], width=2),
                                 hovertemplate="%{y:,.0f}<extra>LY aligned</extra>"))
    fig.add_trace(go.Scatter(x=f.index, y=f["forecast"], name="Forecast", mode="lines",
                             line=dict(color=p["s2"], width=2),
                             customdata=f[["p10", "p90"]].values,
                             hovertemplate="%{y:,.0f} (80%: %{customdata[0]:,.0f}-%{customdata[1]:,.0f})"
                                           "<extra>Forecast</extra>"))
    if len(f):
        lo, hi = f.index.min(), f.index.max() + pd.Timedelta(days=6)
        ev = calendar[(calendar["end"] >= lo) & (calendar["start"] <= hi)
                      & calendar["category"].isin(["festival", "sale", "promo"])]
        for r in ev.sort_values("start").itertuples():
            fig.add_vrect(x0=r.start, x1=r.end + pd.Timedelta(days=1), fillcolor=p["event"], line_width=0,
                          layer="below")
            fig.add_annotation(x=r.start, y=1.0, yref="paper", yanchor="top", xanchor="right", textangle=-90,
                               text=short_name(r.event, r.name), showarrow=False,
                               font=dict(size=10, color=p["muted"]))
        fig.add_vline(x=lo, line_width=1, line_color=p["axis"])
        fig.add_annotation(x=lo, y=0, yref="paper", yanchor="bottom", xanchor="right", text="forecast →  ",
                           showarrow=False, font=dict(size=10, color=p["muted"]))
    fig.update_layout(hovermode="x unified")
    return _layout(fig, p, title, height=440, legend_below=True)


def monthly_bridge_chart(mb: pd.DataFrame, theme: str = "light", title: str | None = None) -> go.Figure:
    p = PALETTES[theme]
    fig = go.Figure()
    if "ly_actual" in mb:
        fig.add_trace(go.Bar(x=mb["month"], y=mb["ly_actual"], name="Last year (actual dates)", marker_color=p["s1"],
                             hovertemplate="%{y:,.0f}<extra>LY actual</extra>"))
        fig.add_trace(go.Bar(x=mb["month"], y=mb["ly_aligned"], name="Last year, festival-aligned",
                             marker_color=p["s3"], hovertemplate="%{y:,.0f}<extra>LY aligned</extra>"))
    fig.add_trace(go.Bar(x=mb["month"], y=mb["forecast"], name="Forecast", marker_color=p["s2"],
                         customdata=mb[["vs_aligned_ly_pct"]].values if "vs_aligned_ly_pct" in mb else None,
                         hovertemplate="%{y:,.0f}" + (" (%{customdata[0]:+.1f}% vs aligned LY)"
                                                       if "vs_aligned_ly_pct" in mb else "")
                                       + "<extra>Forecast</extra>"))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.06, hovermode="x unified")
    fig.update_traces(marker_line_width=0)
    return _layout(fig, p, title, height=380)


def seasonal_index_chart(index_by_month: dict, theme: str = "light", title: str | None = None) -> go.Figure:
    p = PALETTES[theme]
    months = list(index_by_month)
    vals = [index_by_month[m] for m in months]
    fig = go.Figure(go.Bar(x=months, y=vals, marker_color=p["s1"], marker_line_width=0,
                           hovertemplate="%{x}: %{y:.2f}x average<extra></extra>"))
    fig.add_hline(y=1.0, line_width=1, line_color=p["muted"],
                  annotation_text="average", annotation_font_color=p["muted"], annotation_position="right")
    fig = _layout(fig, p, title, height=300, legend=False, yfmt=".2f")
    return fig


def event_lift_chart(lifts: pd.DataFrame, theme: str = "light", title: str | None = None) -> go.Figure:
    """Best-phase daily uplift per event (blue = uplift, red = dip)."""
    p = PALETTES[theme]
    d = lifts.copy()
    d["lift"] = d[["pre_lift_pct", "main_lift_pct"]].max(axis=1)
    d = d.dropna(subset=["lift"]).sort_values("lift")
    fig = go.Figure(go.Bar(
        y=[short_name(k, n) for k, n in zip(d["event"], d["event_name"])], x=d["lift"], orientation="h",
        marker_color=[p["pos"] if v >= 0 else p["neg"] for v in d["lift"]], marker_line_width=0,
        customdata=d[["incremental_last_12m"]].values,
        hovertemplate="%{y}: %{x:+.0f}% per day in window<br>%{customdata[0]:+,.0f} units in last 12 months"
                      "<extra></extra>"))
    fig = _layout(fig, p, title, height=max(260, 26 * len(d) + 60), legend=False, yfmt=",.0f")
    fig.update_xaxes(ticksuffix="%", showgrid=True, gridcolor=p["grid"])
    fig.update_yaxes(showgrid=False, rangemode="normal", tickfont=dict(color=p["ink2"]))
    return fig


def festival_timeline_chart(shifts: pd.DataFrame, years: list[int], theme: str = "light",
                            title: str | None = None) -> go.Figure:
    """Each event's window per year on a common Jan-Dec axis, so year-on-year date moves are visible."""
    p = PALETTES[theme]
    colors = [p["s1"], p["s3"], p["s2"]]
    d = shifts[shifts["year"].isin(years)].copy()
    d["start"] = pd.to_datetime(d["start"])
    d["end"] = pd.to_datetime(d["end"])
    d["label"] = [short_name(k, e) for k, e in zip(d["key"], d["event"])]
    latest = d[d["year"] == years[-1]].sort_values("start")["label"].drop_duplicates().tolist()
    order = latest + sorted(set(d["label"]) - set(latest))
    fig = go.Figure()
    for yi, y in enumerate(years):
        dy = d[d["year"] == y]
        xs, ys, texts = [], [], []
        for r in dy.itertuples():
            a = r.start.replace(year=2000)
            b = (r.end.replace(year=2000) if r.end.year == r.start.year else pd.Timestamp("2000-12-31"))
            label = r.label
            hover = (f"{label} {y}: {r.start:%d %b} - {r.end:%d %b} ({r.weekday})"
                     + (f"<br>{r.shift_vs_ly_days:+.0f} days vs LY" if pd.notna(r.shift_vs_ly_days) else "")
                     + f"<br>{r.source}")
            xs += [a, b + pd.Timedelta(hours=20), None]
            ys += [label, label, None]
            texts += [hover, hover, None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=str(y),
                                 line=dict(color=colors[yi % 3], width=6), marker=dict(size=8),
                                 text=texts, hovertemplate="%{text}<extra></extra>"))
    fig = _layout(fig, p, title, height=max(320, 28 * len(order) + 80), yfmt="")
    fig.update_xaxes(tickformat="%b", dtick="M1", range=["2000-01-01", "2000-12-31"], showgrid=True,
                     gridcolor=p["grid"])
    fig.update_yaxes(categoryorder="array", categoryarray=order[::-1], showgrid=False, rangemode="normal",
                     tickfont=dict(color=p["ink2"]))
    return fig


def supply_chart(sp: pd.DataFrame, theme: str = "light", title: str | None = None) -> go.Figure:
    p = PALETTES[theme]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=sp["bucket"], y=sp["planned_receipts"] + sp["scheduled_receipts"],
                         name="Receipts (inbound + planned)", marker_color=p["s3"], opacity=0.55,
                         marker_line_width=0, hovertemplate="%{y:,.0f}<extra>Receipts</extra>"))
    fig.add_trace(go.Scatter(x=sp["bucket"], y=sp["projected_if_no_new_orders"], name="Projected stock, no new POs",
                             mode="lines", line=dict(color=p["s2"], width=2),
                             hovertemplate="%{y:,.0f}<extra>No new POs</extra>"))
    fig.add_trace(go.Scatter(x=sp["bucket"], y=sp["projected_on_hand"], name="Projected stock with plan",
                             mode="lines", line=dict(color=p["s1"], width=2),
                             hovertemplate="%{y:,.0f}<extra>With plan</extra>"))
    fig.add_trace(go.Scatter(x=sp["bucket"], y=sp["safety_stock"], name="Safety stock", mode="lines",
                             line=dict(color=p["muted"], width=1),
                             hovertemplate="%{y:,.0f}<extra>Safety stock</extra>"))
    fig.add_hline(y=0, line_width=1, line_color=p["axis"])
    fig.update_layout(hovermode="x unified")
    fig = _layout(fig, p, title, height=380)
    fig.update_yaxes(rangemode="normal")
    return fig


def sellin_chart(si: pd.DataFrame, theme: str = "light", title: str | None = None) -> go.Figure:
    p = PALETTES[theme]
    g = si.groupby("bucket")[["sell_out_forecast", "sell_in_plan", "channel_stock_end"]].sum().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=g["bucket"], y=g["sell_in_plan"], name="Sell-in plan", marker_color=p["s1"],
                         marker_line_width=0, hovertemplate="%{y:,.0f}<extra>Sell-in</extra>"))
    fig.add_trace(go.Scatter(x=g["bucket"], y=g["sell_out_forecast"], name="Sell-out forecast", mode="lines",
                             line=dict(color=p["s2"], width=2), hovertemplate="%{y:,.0f}<extra>Sell-out</extra>"))
    fig.add_trace(go.Scatter(x=g["bucket"], y=g["channel_stock_end"], name="Partner stock", mode="lines",
                             line=dict(color=p["s3"], width=2), hovertemplate="%{y:,.0f}<extra>Partner stock</extra>"))
    fig.update_layout(hovermode="x unified")
    return _layout(fig, p, title, height=360)
