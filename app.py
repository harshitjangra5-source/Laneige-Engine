"""Laneige Demand & Supply Engine: web app.

Run:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import os

import pandas as pd
import streamlit as st

from laneige_engine import charts
from laneige_engine import insights as ins
from laneige_engine.agent import DEFAULT_MODEL, ClaudePlanner, PlanningSession, make_planner
from laneige_engine.config import EngineConfig
from laneige_engine.excel import to_excel
from laneige_engine.io import SCHEMAS, read_table, standardize
from laneige_engine.pipeline import PlanningInputs
from laneige_engine.sample_data import generate

st.set_page_config(page_title="Laneige Demand & Supply Engine", layout="wide")

KINDS = {"sell_out": "Sell-out (required)", "sell_in": "Sell-in", "wms": "WMS stock",
         "sku_master": "SKU master (lead time, MOQ, price)", "events": "Your promo / event calendar"}

ss = st.session_state
for k, v in {"raw": {}, "mapping": {}, "std_cache": {}, "session": None, "inputs_key": None, "overrides": [],
             "brief": None, "brief_by": None, "chat": [], "planner": None, "planner_key": None,
             "excel": (None, None), "upload_ids": {}}.items():
    ss.setdefault(k, v)


def theme() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def reset_results():
    ss.session, ss.brief, ss.brief_by, ss.planner, ss.planner_key = None, None, None, None, None


def standardized(kind: str):
    raw = ss.raw.get(kind)
    if raw is None:
        return None, None, None
    key = (id(raw), tuple(sorted((ss.mapping.get(kind) or {}).items())))
    cached = ss.std_cache.get(kind)
    if cached is None or cached[0] != key:
        try:
            df, rep = standardize(raw, kind, ss.mapping.get(kind))
            ss.std_cache[kind] = (key, df, rep, None)
        except Exception as e:  # shown to the user in the Data tab
            ss.std_cache[kind] = (key, None, None, str(e))
    return ss.std_cache[kind][1:]


def fmt_pct(x, signed=True):
    return "n/a" if x is None else (f"{x:+.1f}%" if signed else f"{x:.1f}%")


# ------------------------------------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------------------------------------
with st.sidebar:
    st.header("1 · Data")
    for kind, label in KINDS.items():
        up = st.file_uploader(label, type=["csv", "xlsx", "xls"], key=f"up_{kind}")
        if up is not None:
            uid = f"{up.name}:{up.size}"
            if ss.upload_ids.get(kind) != uid:
                try:
                    ss.raw[kind] = read_table(up)
                    ss.upload_ids[kind] = uid
                    ss.mapping.pop(kind, None)
                    reset_results()
                except Exception as e:
                    st.error(f"Could not read {up.name}: {e}")
    c1, c2 = st.columns(2)
    if c1.button("Load demo data", width="stretch"):
        ss.raw = generate()
        ss.mapping, ss.upload_ids, ss.overrides, ss.chat = {}, {}, [], []
        reset_results()
    if c2.button("Clear", width="stretch"):
        ss.raw, ss.mapping, ss.upload_ids, ss.overrides, ss.chat = {}, {}, {}, [], []
        reset_results()

    st.header("2 · Settings")
    freq = st.selectbox("Planning bucket", ["auto", "W", "M"],
                        format_func={"auto": "Auto", "W": "Weekly", "M": "Monthly"}.get)
    horizon = st.number_input("Horizon in buckets (0 = 26 weeks / 12 months)", 0, 104, 0)
    level = st.selectbox("Forecast level", ["sku", "sku_channel"],
                         format_func={"sku": "SKU (all channels)", "sku_channel": "SKU x channel"}.get)
    service = st.slider("Service level", 0.80, 0.995, 0.95, 0.005)
    lead = st.number_input("Default lead time, days (if not in SKU master/WMS)", 7, 365, 60)
    cover = st.number_input("Partner stock target, weeks of forward sell-out", 0.0, 26.0, 4.0, 0.5)
    cycle = st.number_input("PO cycle, weeks of demand per order", 1.0, 26.0, 4.0, 1.0)
    excess = st.number_input("Excess threshold, weeks of cover", 4.0, 104.0, 26.0, 1.0)
    align = st.toggle("Auto-align sale dates to spikes in the data", value=True)
    config = EngineConfig(freq=freq, horizon=int(horizon) or None, level=level, service_level=service,
                          default_lead_time_days=int(lead), channel_cover_weeks=cover, order_cycle_weeks=cycle,
                          excess_cover_weeks=excess, auto_align_sale_dates=align)
    run_clicked = st.button("Run forecast", type="primary", width="stretch")

    st.header("3 · AI planner")
    env_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    api_key = st.text_input("Anthropic API key", type="password",
                            placeholder="Using key from environment" if env_key else "sk-ant-...",
                            help="Optional. Without a key the planner runs in rule-based offline mode.")
    model = st.text_input("Claude model", DEFAULT_MODEL)
    effort = st.selectbox("Reasoning effort", ["low", "medium", "high", "xhigh", "max"], index=2)

# ------------------------------------------------------------------------------------------------
# Header / empty state
# ------------------------------------------------------------------------------------------------
st.title("Laneige Demand & Supply Engine")
st.caption("Upload WMS, sell-in and sell-out data. The engine learns trend, seasonality and festival/sale uplifts, "
           "shifts them to this year's dates, picks the best model per SKU, and turns the forecast into a sell-in "
           "and supply plan. The AI planner explains it and answers questions.")

if "sell_out" not in ss.raw:
    st.info("Upload your sell-out file in the sidebar (CSV or Excel), or click **Load demo data** to try it.")
    with st.expander("What columns does the engine look for?", expanded=True):
        for kind, label in KINDS.items():
            sch = SCHEMAS[kind]
            st.markdown(f"**{label}**: required `{', '.join(sch['required'])}`; optional "
                        f"`{', '.join(sch['optional'])}`")
        st.markdown("Column names are matched automatically (e.g. *Week Commencing*, *Invoice Date* → date; "
                    "*Material*, *Item Code* → sku; *Units Sold*, *Billed Qty* → qty; *Customer*, *Platform* → "
                    "channel; *Unrestricted Stock*, *SOH* → on_hand; *GIT*, *Open PO* → in_transit). Wide layouts "
                    "with one column per week/month are reshaped automatically. You can correct any mapping in the "
                    "**Data** tab.")
    st.stop()

frames, reports, errors = {}, {}, {}
for kind in KINDS:
    df, rep, err = standardized(kind)
    if err:
        errors[kind] = err
    elif df is not None:
        frames[kind], reports[kind] = df, rep

inputs_key = tuple((k, ss.std_cache[k][0]) for k in sorted(ss.std_cache) if k in ss.raw)
if inputs_key != ss.inputs_key:
    ss.inputs_key = inputs_key
    reset_results()

if "sell_out" not in frames:
    st.error(f"Sell-out file could not be mapped: {errors.get('sell_out')}. Fix the mapping in the Data tab below.")

if "sell_out" in frames and ss.session is None:
    inputs = PlanningInputs(sell_out=frames["sell_out"], sell_in=frames.get("sell_in"), wms=frames.get("wms"),
                            sku_master=frames.get("sku_master"), events=frames.get("events"), reports=reports)
    ss.session = PlanningSession(inputs, config, overrides=ss.overrides)
    run_clicked = True

session: PlanningSession | None = ss.session
if session is not None and run_clicked:
    session.config = config
    bar = st.progress(0.0, text="Starting")
    session.progress = lambda f, m: bar.progress(min(max(f, 0.0), 1.0), text=m)
    try:
        session.run()
    except Exception as e:
        st.exception(e)
    finally:
        session.progress = None
        bar.empty()

res = session.result if session is not None else None

# Planner (Claude when credentials exist, else offline), rebuilt when inputs or credentials change.
if session is not None:
    pkey = (id(session), hashlib.sha256(api_key.encode()).hexdigest() if api_key else env_key, model, effort)
    if ss.planner_key != pkey:
        ss.planner = make_planner(session, api_key or None, model=model, effort=effort)
        ss.planner.history = [m for m in ss.chat if m["role"] in ("user", "assistant")]
        ss.planner_key = pkey

if res is not None:
    with st.sidebar:
        st.header("4 · Export")
        ekey = (id(res), ss.brief)
        if ss.excel[0] != ekey:
            ss.excel = (ekey, to_excel(res, ss.brief))
        st.download_button("Download plan (Excel)", ss.excel[1], width="stretch",
                           file_name=f"laneige_plan_{res.plan_start:%Y%m%d}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

tabs = st.tabs(["AI planner", "Forecast", "Festival calendar", "Trend & seasonality", "Supply plan", "Data"])
T = theme()

# ------------------------------------------------------------------------------------------------
# AI planner
# ------------------------------------------------------------------------------------------------
with tabs[0]:
    if res is None:
        st.warning("Run the forecast first.")
    else:
        is_claude = isinstance(ss.planner, ClaudePlanner)
        st.markdown(f"**Planner:** {'Claude (`' + model + '`)' if is_claude else 'Offline rule-based mode'}"
                    + ("" if is_claude else " · add an Anthropic API key in the sidebar to enable the AI agent."))
        if st.button("Run AI planning cycle", type="primary"):
            with st.status("Planner working...", expanded=True) as status:
                try:
                    ss.brief = ss.planner.run_autonomous(
                        on_event=lambda kind, text: st.write(("🔧 " if kind == "tool" else "💬 ") + text))
                    ss.brief_by = "Claude" if is_claude else "engine"
                    status.update(label="Planning cycle complete", state="complete", expanded=False)
                except Exception as e:
                    status.update(label="Planner failed", state="error")
                    st.error(str(e))
            res = session.result
        if ss.brief:
            st.markdown(ss.brief)
        else:
            with st.expander("Engine brief (rule-based). Run the AI planning cycle for Claude's analysis",
                             expanded=True):
                st.markdown(ins.build_brief(res))

        st.divider()
        st.subheader("Ask the planner")
        for m in ss.chat:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
        q = st.chat_input("e.g. Why is October below last year? Which POs must I place this week? "
                          "BBD actually starts 4-Oct-2026")
        if q:
            ss.chat.append({"role": "user", "content": q})
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                with st.status("Thinking...", expanded=False) as status:
                    try:
                        answer = ss.planner.ask(q, on_event=lambda kind, text: st.write(
                            ("🔧 " if kind == "tool" else "💬 ") + text))
                        status.update(label="Done", state="complete")
                    except Exception as e:
                        answer = f"Sorry, the planner failed: {e}"
                        status.update(label="Failed", state="error")
                st.markdown(answer)
            ss.chat.append({"role": "assistant", "content": answer})
            if session.result is not res:
                st.rerun()

if res is None:
    st.stop()

# ------------------------------------------------------------------------------------------------
# Forecast
# ------------------------------------------------------------------------------------------------
sku_names = res.profiles.set_index("sku")["product_name"].fillna("").to_dict()
sku_options = ["All SKUs"] + sorted(sku_names, key=lambda s: -res.forecast[res.forecast["sku"] == s]["forecast"].sum())


def label(s):
    return s if s == "All SKUs" else f"{s} · {sku_names.get(s, '')}"


with tabs[1]:
    s = res.summary
    near = 13 if res.freq == "W" else 3
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Forecast, next {near} {res.bucket_label}s", f"{s.get(f'forecast_next_{near}', 0):,.0f} units")
    m2.metric("vs last year", fmt_pct(s.get("vs_ly_pct")))
    m3.metric("vs last year, festival-aligned", fmt_pct(s.get("vs_aligned_ly_pct")),
              help="Last year's actuals with festival & sale volume moved to this year's dates.")
    m4.metric("Backtest accuracy", f"{s['portfolio_accuracy']:.1%}" if s.get("portfolio_accuracy") else "n/a",
              help=f"Volume-weighted 1 - WAPE over rolling backtests; bias {s.get('portfolio_bias') or 0:+.1%}.")
    pick = st.selectbox("SKU", sku_options, format_func=label, key="fc_sku")
    sku = None if pick == "All SKUs" else pick
    hist = res.history if sku is None else res.history[res.history["sku"] == sku]
    fc = res.forecast if sku is None else res.forecast[res.forecast["sku"] == sku]
    br = res.shift_bridge if (sku is None or res.shift_bridge.empty) else res.shift_bridge[res.shift_bridge["sku"] == sku]
    st.plotly_chart(charts.forecast_chart(hist, fc, br, res.calendar, T,
                                          f"Sell-out: {label(pick)}", history_buckets=52 if res.freq == "W" else 24),
                    width="stretch")
    mb = ins.monthly_bridge(res, sku)
    st.plotly_chart(charts.monthly_bridge_chart(mb, T, "By month: forecast vs last year (actual and festival-aligned)"),
                    width="stretch")
    with st.expander("Table view: monthly"):
        st.dataframe(mb, width="stretch", hide_index=True)
    if sku:
        acc = res.accuracy[res.accuracy["sku"] == sku].iloc[0]
        st.markdown(f"**Model:** {acc['model']}"
                    + (f" ({acc['ensemble_of']})" if acc["ensemble_of"] else "")
                    + (f" · backtest accuracy {acc['accuracy']:.1%}, bias {acc['bias']:+.1%}, naive accuracy "
                       f"{1 - acc['naive_wape']:.1%}" if pd.notna(acc["accuracy"]) and pd.notna(acc["naive_wape"]) else "")
                    + (f" · {acc['notes']}" if acc["notes"] else ""))
    with st.expander("Table view: forecast by bucket"):
        st.dataframe(fc.drop(columns=["category"], errors="ignore"), width="stretch", hide_index=True)
    st.subheader("Accuracy by SKU")
    st.dataframe(res.accuracy[["sku", "product_name", "model", "accuracy", "bias", "naive_wape", "fva",
                               "demand_class", "status", "notes"]],
                 width="stretch", hide_index=True,
                 column_config={"accuracy": st.column_config.NumberColumn(format="percent"),
                                "bias": st.column_config.NumberColumn(format="percent"),
                                "naive_wape": st.column_config.NumberColumn("naive WAPE", format="percent"),
                                "fva": st.column_config.NumberColumn("value added vs naive", format="percent")})

# ------------------------------------------------------------------------------------------------
# Festival calendar
# ------------------------------------------------------------------------------------------------
with tabs[2]:
    y = res.plan_start.year
    horizon_end = res.forecast["bucket"].max()
    years = [y - 2, y - 1, y]
    st.plotly_chart(charts.festival_timeline_chart(res.calendar_shifts, years, T,
                                                   "Festival & sale windows by year (same Jan-Dec axis)"),
                    width="stretch")
    if res.calendar_log:
        st.markdown("**Automatic calendar adjustments**")
        for line in res.calendar_log:
            st.markdown(f"- {line}")
    st.subheader("Upcoming events in the horizon: edit dates and re-run")
    up = res.calendar[(res.calendar["end"] >= pd.Timestamp(res.plan_start)) &
                      (res.calendar["start"] <= horizon_end)].copy()
    edit = pd.DataFrame({"event": up["name"], "start": up["start"].dt.date, "end": up["end"].dt.date,
                         "source": up["source"], "confidence": up["confidence"], "note": up["note"]})
    edited = st.data_editor(edit, hide_index=True, width="stretch", disabled=["event", "source", "confidence", "note"],
                            column_config={"start": st.column_config.DateColumn(format="DD-MMM-YYYY"),
                                           "end": st.column_config.DateColumn(format="DD-MMM-YYYY")},
                            key="cal_editor")
    with st.form("add_event"):
        st.markdown("**Add a brand promotion or correct any date**")
        c1, c2, c3 = st.columns([2, 1, 1])
        ev_name = c1.text_input("Event name", placeholder="e.g. Laneige Brand Day on Nykaa, Big Billion Days")
        ev_start = c2.date_input("Start", value=None)
        ev_end = c3.date_input("End", value=None)
        add = st.form_submit_button("Add event")
    if st.button("Apply calendar changes & re-run", type="primary") or (add and ev_name and ev_start):
        for (_, a), (_, b) in zip(edit.iterrows(), edited.iterrows()):
            if a["start"] != b["start"] or a["end"] != b["end"]:
                session.set_event(a["event"], str(b["start"]), str(b["end"]))
        if add and ev_name and ev_start:
            session.set_event(ev_name, str(ev_start), str(ev_end or ev_start))
        ss.overrides = session.overrides
        with st.spinner("Re-running forecast with your calendar"):
            session.run()
        st.rerun()
    st.subheader("All events by year")
    yr = st.selectbox("Year", sorted(res.calendar_shifts["year"].unique(), reverse=True),
                      index=sorted(res.calendar_shifts["year"].unique(), reverse=True).index(y)
                      if y in set(res.calendar_shifts["year"]) else 0)
    st.dataframe(res.calendar_shifts[res.calendar_shifts["year"] == yr], width="stretch", hide_index=True)
    if not res.portfolio_events.empty:
        with st.expander("Learned uplift per event (portfolio / pool level)"):
            st.dataframe(res.portfolio_events, width="stretch", hide_index=True)

# ------------------------------------------------------------------------------------------------
# Trend & seasonality
# ------------------------------------------------------------------------------------------------
with tabs[3]:
    prof = res.profiles
    st.dataframe(prof[["sku", "product_name", "category", "trend_label", "yoy_growth", "momentum",
                       "seasonality_label", "seasonality_strength", "peak_months", "low_months", "demand_class",
                       "status", "top_events"]],
                 width="stretch", hide_index=True,
                 column_config={"yoy_growth": st.column_config.NumberColumn("YoY growth (festival-adj.)",
                                                                             format="percent"),
                                "momentum": st.column_config.NumberColumn("trend momentum", format="percent"),
                                "seasonality_strength": st.column_config.ProgressColumn(min_value=0, max_value=1)})
    pick2 = st.selectbox("SKU", sku_options[1:], format_func=label, key="ts_sku")
    card = ins.sku_card(res, pick2)
    c1, c2 = st.columns(2)
    with c1:
        if card["seasonal_index_by_month"]:
            st.plotly_chart(charts.seasonal_index_chart(card["seasonal_index_by_month"], T,
                                                        "Seasonal index by month (1.0 = average)"),
                            width="stretch")
    with c2:
        lifts = res.event_lifts[res.event_lifts["sku"] == pick2] if not res.event_lifts.empty else pd.DataFrame()
        if not lifts.empty:
            st.plotly_chart(charts.event_lift_chart(lifts, T, "Festival / sale uplift per day in window"),
                            width="stretch")
    with st.expander("Table view: event lifts & model tournament"):
        if not lifts.empty:
            st.dataframe(lifts.drop(columns=["product_name", "category"], errors="ignore"), width="stretch",
                         hide_index=True)
        st.dataframe(pd.DataFrame(card["model_tournament"]), width="stretch", hide_index=True)

# ------------------------------------------------------------------------------------------------
# Supply
# ------------------------------------------------------------------------------------------------
with tabs[4]:
    if res.alerts.empty and res.supply_plan.empty:
        st.info("Upload a WMS stock file to get the DC supply plan and alerts. The sell-in plan is below.")
    if not res.alerts.empty:
        st.subheader("Alerts")
        st.dataframe(res.alerts, width="stretch", hide_index=True,
                     column_config={"units": st.column_config.NumberColumn(format="localized"),
                                    "value": st.column_config.NumberColumn("value (₹)", format="localized")})
    if not res.orders.empty:
        st.subheader("Purchase orders")
        st.dataframe(res.orders, width="stretch", hide_index=True,
                     column_config={"value": st.column_config.NumberColumn("value (₹)", format="localized")})
    pick3 = st.selectbox("SKU", sku_options[1:], format_func=label, key="sup_sku")
    c1, c2 = st.columns(2)
    with c1:
        if not res.supply_plan.empty:
            sp = res.supply_plan[res.supply_plan["sku"] == pick3]
            st.plotly_chart(charts.supply_chart(sp, T, "DC stock projection"), width="stretch")
    with c2:
        si = res.sellin_plan[res.sellin_plan["sku"] == pick3]
        st.plotly_chart(charts.sellin_chart(si, T, "Sell-in plan vs sell-out forecast"), width="stretch")
    with st.expander("Table view: supply & sell-in plan"):
        if not res.supply_plan.empty:
            st.dataframe(res.supply_plan[res.supply_plan["sku"] == pick3], width="stretch", hide_index=True)
        st.dataframe(si, width="stretch", hide_index=True)

# ------------------------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Column mapping")
    for kind, raw in ss.raw.items():
        _, rep, err = standardized(kind)
        with st.expander(f"{KINDS[kind]}: {len(raw):,} rows" + (" · ⚠ not mapped" if err else ""),
                         expanded=bool(err)):
            if err:
                st.error(err)
            if rep is not None:
                st.caption(f"Granularity: {rep.granularity or 'n/a'} · {rep.date_min or ''} to {rep.date_max or ''} · "
                           f"{rep.n_skus} SKUs" + (" · reshaped from wide layout" if rep.reshaped_from_wide else ""))
                for w in rep.warnings:
                    st.caption(f"• {w}")
            current = rep.mapping if rep is not None else {}
            cols = ["(none)"] + list(map(str, raw.columns))
            fields = SCHEMAS[kind]["required"] + SCHEMAS[kind]["optional"]
            chosen = {}
            grid = st.columns(4)
            for i, f in enumerate(fields):
                default = current.get(f, "(none)")
                chosen[f] = grid[i % 4].selectbox(f + (" *" if f in SCHEMAS[kind]["required"] else ""), cols,
                                                  index=cols.index(default) if default in cols else 0,
                                                  key=f"map_{kind}_{f}")
            if st.button("Apply mapping", key=f"apply_{kind}"):
                ss.mapping[kind] = {f: c for f, c in chosen.items() if c != "(none)"}
                st.rerun()
            st.dataframe(raw.head(5), width="stretch", hide_index=True)
    st.subheader("Data quality")
    st.dataframe(res.data_quality, width="stretch", hide_index=True)
    if not res.anomalies.empty:
        st.subheader("Unexplained spikes / dips (possible missing promotions)")
        st.dataframe(res.anomalies, width="stretch", hide_index=True)
    with st.expander("Engine log"):
        for line in res.log + res.calendar_log:
            st.markdown(f"- {line}")
