"""Formatted Excel workbook with the full plan."""
from __future__ import annotations

import io
import re

import pandas as pd

from . import insights as ins
from .pipeline import PlanningResult

PCT_COLS = re.compile(r"(^|_)(pct|accuracy|wape|bias|fva|growth|momentum|strength)($|_)")
INT_COLS = re.compile(r"forecast|p10|p90|baseline|qty|units|sell_in|sell_out|stock|receipt|demand|actual|aligned|"
                      r"impact|incremental|target|projected|safety|value|order")


def _sheet(writer, name: str, df: pd.DataFrame, fmt: dict, note: str | None = None):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.date
    start = 2 if note else 0
    df.to_excel(writer, sheet_name=name, index=False, startrow=start)
    ws = writer.sheets[name]
    if note:
        ws.write(0, 0, note, fmt["note"])
    for i, col in enumerate(df.columns):
        ws.write(start, i, col, fmt["head"])
        sample = df[col].astype(str).head(200)
        width = min(60, max(10, len(str(col)) + 2, int(sample.str.len().quantile(0.9)) + 2 if len(sample) else 10))
        cf = None
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            if "pct" in col:
                cf = fmt["dec1"]
            elif PCT_COLS.search(col) and df[col].abs().max(skipna=True) is not None and \
                    (df[col].abs().max(skipna=True) or 0) <= 5:
                cf = fmt["pct"]
            elif INT_COLS.search(col):
                cf = fmt["int"]
            else:
                cf = fmt["dec2"]
        ws.set_column(i, i, width, cf)
    if len(df):
        ws.autofilter(start, 0, start + len(df), len(df.columns) - 1)
    ws.freeze_panes(start + 1, 1)
    return ws


def to_excel(res: PlanningResult, brief: str | None = None) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        wb = writer.book
        fmt = {
            "head": wb.add_format({"bold": True, "bg_color": "#1F3A5F", "font_color": "#FFFFFF", "border": 1,
                                   "text_wrap": True, "valign": "top"}),
            "title": wb.add_format({"bold": True, "font_size": 16, "font_color": "#1F3A5F"}),
            "h2": wb.add_format({"bold": True, "font_size": 12, "font_color": "#1F3A5F"}),
            "note": wb.add_format({"italic": True, "font_color": "#555555"}),
            "int": wb.add_format({"num_format": "#,##0"}),
            "dec1": wb.add_format({"num_format": "0.0"}),
            "dec2": wb.add_format({"num_format": "0.00"}),
            "pct": wb.add_format({"num_format": "0.0%"}),
            "wrap": wb.add_format({"text_wrap": True, "valign": "top"}),
            "bold": wb.add_format({"bold": True}),
            "crit": wb.add_format({"bg_color": "#F8D7DA", "font_color": "#842029"}),
            "high": wb.add_format({"bg_color": "#FFF3CD", "font_color": "#664D03"}),
            "med": wb.add_format({"bg_color": "#E2E3E5"}),
        }

        # ---- Summary ----
        ws = wb.add_worksheet("Summary")
        writer.sheets["Summary"] = ws
        ws.set_column(0, 0, 44)
        ws.set_column(1, 1, 110)
        ws.write(0, 0, "Laneige Demand & Supply Plan", fmt["title"])
        s = res.summary
        near = 13 if res.freq == "W" else 3
        kpis = [
            ("Plan start", str(s["plan_start"])),
            ("Planning bucket / horizon", f"{s['bucket']} / {s['horizon_buckets']}"),
            ("Series forecast", s["series"]),
            (f"Forecast next {near} {s['bucket']}s (units)", round(s.get(f"forecast_next_{near}") or 0)),
            ("vs last year", f"{s['vs_ly_pct']:+.1f}%" if s.get("vs_ly_pct") is not None else "n/a"),
            ("vs last year aligned to this year's festivals",
             f"{s['vs_aligned_ly_pct']:+.1f}%" if s.get("vs_aligned_ly_pct") is not None else "n/a"),
            ("Backtest accuracy (volume-weighted)",
             f"{s['portfolio_accuracy']:.1%}" if s.get("portfolio_accuracy") is not None else "n/a"),
            ("Forecast bias", f"{s['portfolio_bias']:+.1%}" if s.get("portfolio_bias") is not None else "n/a"),
            ("Critical / high supply alerts", f"{s['critical_alerts']} / {s['high_alerts']}"),
        ]
        for i, (k, v) in enumerate(kpis):
            ws.write(2 + i, 0, k, fmt["bold"])
            ws.write(2 + i, 1, v)
        row = 3 + len(kpis)
        ws.write(row, 0, "AI planner brief" if brief else "Engine brief", fmt["h2"])
        row += 1
        for line in (brief or ins.build_brief(res)).splitlines():
            clean = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
            if clean.startswith("#"):
                ws.write(row, 0, clean.lstrip("# "), fmt["h2"])
            else:
                ws.write(row, 1, clean, fmt["wrap"])
            row += 1

        mb = ins.monthly_bridge(res)
        _sheet(writer, "Monthly vs LY", mb, fmt,
               "ly_aligned = last year's actuals with festival/sale volume moved to this year's dates; "
               "festival_shift_impact = ly_aligned - ly_actual.")

        fc = res.forecast.copy()
        _sheet(writer, "Forecast", fc, fmt, "Sell-out forecast per bucket with 80% interval (p10-p90). "
               "baseline = without festival/sale effects; event_factor = uplift multiplier.")
        piv = fc.pivot_table(index=[c for c in res.key_cols] + ["product_name"], columns="bucket",
                             values="forecast", aggfunc="sum")
        piv.columns = [c.strftime("%d-%b-%y") for c in piv.columns]
        piv = piv.reset_index()
        _sheet(writer, "Forecast pivot", piv, fmt)

        if not res.sellin_plan.empty:
            _sheet(writer, "Sell-in plan", res.sellin_plan, fmt,
                   f"Sell-in = sell-out forecast + change needed to hold {res.config.channel_cover_weeks:g} weeks of "
                   f"forward cover at partners (pre-loads ahead of festivals).")
        if not res.supply_plan.empty:
            _sheet(writer, "DC supply plan", res.supply_plan, fmt,
                   "Projected DC stock from WMS on-hand + inbound - sell-in plan; planned receipts keep stock above "
                   "safety stock. 'expedite' = needed inside the lead time.")
        if not res.orders.empty:
            _sheet(writer, "Orders", res.orders, fmt)
        ws = _sheet(writer, "Alerts", res.alerts, fmt)
        if not res.alerts.empty:
            n = len(res.alerts)
            for sev, f in (("critical", "crit"), ("high", "high"), ("medium", "med")):
                ws.conditional_format(1, 0, n, len(res.alerts.columns) - 1,
                                      {"type": "formula", "criteria": f'=$C2="{sev}"', "format": fmt[f]})

        _sheet(writer, "Trend & seasonality", res.profiles, fmt,
               "yoy_growth = festival-adjusted growth, last 52 weeks vs prior 52; momentum = current trend slope "
               "(annualised); idx_* = seasonal index by month (1.0 = average).")
        if not res.event_lifts.empty:
            _sheet(writer, "Event lifts", res.event_lifts, fmt,
                   "*_lift_pct = extra sales per day inside the event phase (+80 = 1.8x); *_pool_pct = portfolio "
                   "estimate the SKU is shrunk toward; incremental = units attributable to the event.")
        if not res.portfolio_events.empty:
            _sheet(writer, "Portfolio lifts", res.portfolio_events, fmt)
        _sheet(writer, "Festival calendar", res.calendar_shifts, fmt,
               "shift_vs_ly_days = how many days later (+) or earlier (-) than last year. Sources: lunar table, "
               "fixed date, reported, rule estimate, auto-aligned (snapped to your data), your calendar.")
        if not res.shift_bridge.empty:
            _sheet(writer, "Festival shift bridge", res.shift_bridge, fmt)
        acc = res.accuracy.copy()
        _sheet(writer, "Model accuracy", acc, fmt,
               "Rolling-origin backtest. accuracy = 1 - WAPE; fva = improvement over naive last-year forecast.")
        _sheet(writer, "Model tournament", res.model_scores, fmt)
        if not res.anomalies.empty:
            _sheet(writer, "Anomalies", res.anomalies, fmt,
                   "Unexplained spikes / dips (not covered by the calendar). Add missing promotions to your calendar.")
        dq = res.data_quality.copy()
        extra = pd.DataFrame({"severity": "info", "dataset": "engine", "issue": res.log + res.calendar_log})
        _sheet(writer, "Data quality", pd.concat([dq, extra], ignore_index=True), fmt)
        cfg = pd.DataFrame(list(res.config.to_dict().items()), columns=["setting", "value"])
        cfg["value"] = cfg["value"].astype(str)
        _sheet(writer, "Settings", cfg, fmt)
    return buf.getvalue()
