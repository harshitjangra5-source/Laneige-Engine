"""Compact, planner-friendly views of a PlanningResult.

Shared by the Streamlit app, the Excel export, the Claude agent's tools and the offline brief,
so every surface reports the same numbers.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .pipeline import PlanningResult


def _r(x, nd=1):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return round(float(x), nd)


def _growth(row) -> float:
    """Festival-adjusted YoY growth, falling back to current trend momentum for short histories."""
    g = row.yoy_growth
    return row.momentum if g is None or (isinstance(g, float) and math.isnan(g)) else g


def _pct(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.1f}%"


def monthly_bridge(res: PlanningResult, sku: str | None = None) -> pd.DataFrame:
    """Forecast vs last year vs last year re-aligned to this year's festival dates, by calendar month."""
    b = res.shift_bridge
    fc = res.forecast
    if sku:
        b = b[b["sku"] == sku]
        fc = fc[fc["sku"] == sku]
    if fc.empty:
        return pd.DataFrame()
    fc = fc.assign(month=pd.to_datetime(fc["bucket"]).dt.to_period("M"))
    out = fc.groupby("month")[["forecast", "p10", "p90"]].sum()
    if not b.empty:
        b = b.assign(month=pd.to_datetime(b["bucket"]).dt.to_period("M"))
        out = out.join(b.groupby("month")[["ly_actual", "ly_aligned", "festival_shift_impact"]].sum(), how="left")
        out["vs_ly_pct"] = (out["forecast"] / out["ly_actual"].replace(0, np.nan) - 1) * 100
        out["vs_aligned_ly_pct"] = (out["forecast"] / out["ly_aligned"].replace(0, np.nan) - 1) * 100
    events = (res.calendar.assign(month=res.calendar["start"].dt.to_period("M"))
              .groupby("month")["name"].agg(lambda s: ", ".join(n.split(" (")[0] for n in s)))
    out["events_this_year"] = events.reindex(out.index).fillna("")
    out.index = out.index.astype(str)
    return out.reset_index()


def portfolio_overview(res: PlanningResult) -> dict:
    s = dict(res.summary)
    mb = monthly_bridge(res)
    s["monthly"] = [{k: (_r(v, 0) if isinstance(v, (float, np.floating)) and "pct" not in k else
                         _r(v, 1) if isinstance(v, (float, np.floating)) else v)
                     for k, v in row.items()} for row in mb.to_dict("records")]
    cat = res.forecast.groupby("category")["forecast"].sum().sort_values(ascending=False)
    s["forecast_by_category"] = {k or "Uncategorised": _r(v, 0) for k, v in cat.items()}
    return _clean(s)


def top_skus(res: PlanningResult, by: str = "volume", n: int = 10) -> list[dict]:
    fc = res.forecast.groupby("sku")["forecast"].sum().rename("forecast_units")
    p = res.profiles.set_index("sku")
    a = res.accuracy.set_index("sku")
    df = p[["product_name", "category", "trend_label", "yoy_growth", "seasonality_label", "peak_months",
            "demand_class", "status"]].join(fc).join(a[["model", "accuracy", "bias"]])
    if not res.alerts.empty:
        worst = res.alerts.groupby("sku")["severity"].first()
        df = df.join(worst.rename("worst_alert"))
    key = {"volume": "forecast_units", "growth": "yoy_growth", "decline": "yoy_growth",
           "accuracy": "accuracy", "worst_accuracy": "accuracy"}.get(by, "forecast_units")
    asc = by in {"decline", "worst_accuracy"}
    df = df.sort_values(key, ascending=asc, na_position="last").head(n).reset_index()
    return _clean(df.to_dict("records"))


def sku_card(res: PlanningResult, sku: str) -> dict:
    if sku not in set(res.profiles["sku"]):
        matches = res.profiles[res.profiles["product_name"].str.contains(sku, case=False, na=False) |
                               res.profiles["sku"].str.contains(sku, case=False, na=False)]
        if matches.empty:
            return {"error": f"SKU '{sku}' not found. Known SKUs: {sorted(res.profiles['sku'])[:30]}"}
        sku = matches["sku"].iloc[0]
    prof = res.profiles[res.profiles["sku"] == sku].iloc[0].to_dict()
    acc = res.accuracy[res.accuracy["sku"] == sku].iloc[0].to_dict()
    scores = res.model_scores[res.model_scores["sku"] == sku][["model", "wape", "bias", "selected"]]
    lifts = res.event_lifts[res.event_lifts["sku"] == sku] if not res.event_lifts.empty else pd.DataFrame()
    if not lifts.empty:
        lifts = lifts.assign(abs_inc=lifts["incremental_last_12m"].abs()).sort_values("abs_inc", ascending=False)
        lifts = lifts[["event_name", "pre_lift_pct", "main_lift_pct", "post_lift_pct", "main_pool_pct",
                       "incremental_last_12m", "incremental_horizon"]].head(8)
    fc = res.forecast[res.forecast["sku"] == sku].groupby("bucket")[["forecast", "p10", "p90", "baseline"]].sum()
    ev = res.forecast[res.forecast["sku"] == sku].groupby("bucket")["events"].first()
    fc = fc.join(ev).reset_index()
    fc["bucket"] = fc["bucket"].dt.strftime("%Y-%m-%d")
    alerts = res.alerts[res.alerts["sku"] == sku][["severity", "alert", "detail"]] if not res.alerts.empty else []
    orders = res.orders[res.orders["sku"] == sku] if not res.orders.empty else pd.DataFrame()
    hist = res.history[res.history["sku"] == sku].groupby("bucket")["qty"].sum().tail(8)
    card = {
        "sku": sku, "product_name": prof.get("product_name"), "category": prof.get("category"),
        "profile": {k: prof.get(k) for k in ("status", "demand_class", "history_buckets", "avg_recent", "yoy_growth",
                                              "momentum", "trend_label", "seasonality_strength",
                                              "seasonality_label", "peak_months", "low_months", "top_events")},
        "seasonal_index_by_month": {k[4:]: v for k, v in prof.items() if k.startswith("idx_")},
        "model": {k: acc.get(k) for k in ("model", "ensemble_of", "accuracy", "wape", "bias", "naive_wape", "fva",
                                           "backtest_folds", "notes")},
        "model_tournament": scores.to_dict("records"),
        "event_lifts": lifts.to_dict("records") if not isinstance(lifts, list) else [],
        "recent_actuals": {d.strftime("%Y-%m-%d"): v for d, v in hist.items()},
        "forecast": fc.to_dict("records"),
        "monthly_vs_ly": monthly_bridge(res, sku).to_dict("records"),
        "alerts": alerts.to_dict("records") if hasattr(alerts, "to_dict") else [],
        "orders": orders.to_dict("records") if not orders.empty else [],
    }
    return _clean(card)


def calendar_view(res: PlanningResult, year: int | None = None, upcoming_only: bool = False) -> list[dict]:
    t = res.calendar_shifts
    if year:
        t = t[t["year"] == year]
    if upcoming_only:
        t = t[pd.to_datetime(t["start"]) >= pd.Timestamp(res.plan_start)]
    return _clean(t.to_dict("records"))


def alerts_view(res: PlanningResult, limit: int = 25) -> dict:
    a = res.alerts.head(limit) if not res.alerts.empty else pd.DataFrame()
    o = res.orders
    now = o[o["status"].isin(["EXPEDITE", "PLACE NOW"])] if not o.empty else pd.DataFrame()
    return _clean({
        "alerts": a.to_dict("records") if not a.empty else [],
        "orders_to_act_on_now": now.to_dict("records") if not now.empty else [],
        "upcoming_planned_orders": (o[o["status"] == "PLANNED"].head(limit).to_dict("records")
                                    if not o.empty else []),
    })


def data_quality_view(res: PlanningResult) -> dict:
    anomalies = res.anomalies.head(12) if not res.anomalies.empty else pd.DataFrame()
    return _clean({
        "issues": res.data_quality.to_dict("records"),
        "engine_log": res.log,
        "calendar_adjustments": res.calendar_log,
        "unexplained_spikes_or_dips": (anomalies[["sku", "bucket", "actual", "expected", "deviation_pct",
                                                  "events_in_bucket"]].to_dict("records")
                                       if not anomalies.empty else []),
    })


def _clean(obj):
    """Make results JSON-safe (dates -> str, NaN -> None, numpy -> python, round floats)."""
    if obj is pd.NA or obj is pd.NaT:
        return None
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (pd.Timestamp,)):
        return obj.strftime("%Y-%m-%d")
    if hasattr(obj, "isoformat") and not isinstance(obj, str):
        return obj.isoformat()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        v = float(obj)
        return round(v, 3) if abs(v) < 10 else round(v, 1)
    if obj is pd.NaT or obj is pd.NA:
        return None
    return obj


# ------------------------------------------------------------------------------------------------
# Offline brief (used when no Claude API key is configured)
# ------------------------------------------------------------------------------------------------
def build_brief(res: PlanningResult) -> str:
    s = res.summary
    near = 13 if res.freq == "W" else 3
    unit = res.bucket_label
    L = [f"# S&OP brief: plan from {s['plan_start']:%d %b %Y}" if hasattr(s["plan_start"], "strftime")
         else f"# S&OP brief: plan from {s['plan_start']}", ""]
    fc_near = s.get(f"forecast_next_{near}")
    line = f"**Next {near} {unit}s:** {fc_near:,.0f} units"
    if s.get("vs_ly_pct") is not None:
        line += f" ({_pct(s['vs_ly_pct'])} vs last year"
        if s.get("vs_aligned_ly_pct") is not None:
            line += f", {_pct(s['vs_aligned_ly_pct'])} vs last year re-aligned to this year's festival dates"
        line += ")"
    L += [line + ".", ""]
    if s.get("portfolio_accuracy") is not None:
        L.append(f"**Forecast accuracy (backtest, volume-weighted):** {s['portfolio_accuracy']:.0%} "
                 f"(bias {s['portfolio_bias']:+.1%}).")
    L.append(f"**Supply:** {s['critical_alerts']} critical and {s['high_alerts']} high alerts.")
    L.append("")

    if s["festival_moves_in_horizon"]:
        L.append("## Festival & sale shifts in the horizon")
        L += [f"- {m}" for m in s["festival_moves_in_horizon"]]
        mb = monthly_bridge(res)
        if not mb.empty and "festival_shift_impact" in mb:
            big = mb[mb["festival_shift_impact"].abs() > 0.05 * mb["ly_actual"].abs().clip(lower=1)]
            for r in big.itertuples():
                L.append(f"- {r.month}: festival timing moves {r.festival_shift_impact:+,.0f} units vs last year's "
                         f"calendar ({_pct(r.vs_aligned_ly_pct)} vs aligned LY, {_pct(r.vs_ly_pct)} vs raw LY).")
        L.append("")

    L.append("## Trend & seasonality")
    p = res.profiles
    grow = p[p["trend_label"] == "Growing"].sort_values("yoy_growth", ascending=False)
    dec = p[p["trend_label"] == "Declining"].sort_values("yoy_growth")
    if not grow.empty:
        L.append("- Growing: " + ", ".join(f"{r.product_name or r.sku} ({_growth(r):+.0%})"
                                          for r in grow.head(5).itertuples()))
    if not dec.empty:
        L.append("- Declining: " + ", ".join(f"{r.product_name or r.sku} ({_growth(r):+.0%})"
                                            for r in dec.head(5).itertuples()))
    strong = p[p["seasonality_label"].str.startswith("Strong")]
    for r in strong.head(5).itertuples():
        L.append(f"- {r.product_name or r.sku}: strong seasonality, peaks {r.peak_months or 'n/a'}, lows {r.low_months or 'n/a'}.")
    if not res.portfolio_events.empty:
        pe = res.portfolio_events.copy()
        pe["best"] = pe[["pre_lift_pct", "main_lift_pct"]].max(axis=1)
        top = pe.sort_values("best", ascending=False).drop_duplicates("event").head(5)
        L.append("- Biggest event uplifts (portfolio, per day in window): " +
                 ", ".join(f"{r.event} {r.best:+.0f}%" for r in top.itertuples()))
    L.append("")

    if not res.alerts.empty:
        L.append("## Supply actions")
        for r in res.alerts.head(10).itertuples():
            L.append(f"- **{r.severity.upper()}** {r.product_name or r.sku}: {r.detail}")
        L.append("")

    acc = res.accuracy.dropna(subset=["accuracy"]).sort_values("accuracy")
    if not acc.empty:
        L.append("## Forecast exceptions")
        for r in acc.head(3).itertuples():
            L.append(f"- {r.product_name or r.sku}: accuracy {r.accuracy:.0%} ({r.model}, {r.demand_class}). "
                     f"Review drivers or add promo dates.")
        L.append("")

    if s["dates_to_confirm"]:
        L.append("## Calendar dates to confirm")
        L += [f"- {d}" for d in s["dates_to_confirm"]]
        L.append("")
    dq = res.data_quality
    if not dq.empty or res.calendar_log:
        L.append("## Data notes")
        L += [f"- {x}" for x in res.calendar_log[:6]]
        L += [f"- [{r.severity}] {r.issue}" for r in dq.itertuples() if r.severity != "info"][:8]
    return "\n".join(L)
