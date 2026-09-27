"""End-to-end planning run: data -> calendar -> forecasts -> explainability -> sell-in -> supply."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import calendar as cal
from . import supply as sup
from .config import EngineConfig
from .forecaster import (MODEL_LABELS, Grid, SeriesResult, bucket_start, event_lifts, make_grid, profile,
                         run_portfolio, shift_bridge)
from .io import LoadReport, detect_granularity


@dataclass
class PlanningInputs:
    sell_out: pd.DataFrame
    sell_in: pd.DataFrame | None = None
    wms: pd.DataFrame | None = None
    sku_master: pd.DataFrame | None = None
    events: pd.DataFrame | None = None
    reports: dict[str, LoadReport] = field(default_factory=dict)


@dataclass
class PlanningResult:
    config: EngineConfig
    freq: str
    grid: Grid
    key_cols: list[str]
    plan_start: pd.Timestamp
    calendar: pd.DataFrame
    calendar_shifts: pd.DataFrame
    calendar_log: list[str]
    history: pd.DataFrame
    forecast: pd.DataFrame
    accuracy: pd.DataFrame
    model_scores: pd.DataFrame
    profiles: pd.DataFrame
    event_lifts: pd.DataFrame
    portfolio_events: pd.DataFrame
    shift_bridge: pd.DataFrame
    anomalies: pd.DataFrame
    sellin_plan: pd.DataFrame
    supply_plan: pd.DataFrame
    alerts: pd.DataFrame
    orders: pd.DataFrame
    data_quality: pd.DataFrame
    summary: dict
    log: list[str]
    series: list[SeriesResult]

    @property
    def bucket_label(self) -> str:
        return "week" if self.freq == "W" else "month"


def _key_tuple(row, key_cols):
    return tuple(row[c] for c in key_cols)


def _event_name(key: str, occ: pd.DataFrame) -> str:
    if key in cal.EVENTS:
        return cal.EVENTS[key].name
    m = occ.loc[occ["event"] == key, "name"]
    return str(m.iloc[0]) if len(m) else key


def run_planning(inputs: PlanningInputs, config: EngineConfig | None = None,
                 progress: Callable[[float, str], None] | None = None) -> PlanningResult:
    cfg = config or EngineConfig()
    say = progress or (lambda f, m: None)
    log: list[str] = []
    dq: list[dict] = []
    so = inputs.sell_out.copy()
    for name, rep in inputs.reports.items():
        for w in rep.warnings:
            dq.append(dict(severity="info", dataset=name, issue=w))

    # ---------------- granularity & buckets ----------------
    say(0.02, "Detecting granularity")
    native, native_days = detect_granularity(so["date"])
    freq = cfg.freq if cfg.freq in {"W", "M"} else ("M" if native == "M" else "W")
    if freq == "W" and native == "M":
        freq = "M"
        log.append("Data is monthly, so the plan is monthly.")
    week_start = int(so["date"].dt.weekday.mode().iloc[0]) if native == "W" else 0
    so["bucket"] = bucket_start(so["date"], freq, week_start)
    last_date = so["date"].max()
    last_bucket = so["bucket"].max()
    if native == "D":
        bucket_end = (last_bucket + pd.Timedelta(days=6)) if freq == "W" else (last_bucket + pd.offsets.MonthEnd(0))
        if last_date < bucket_end:
            so = so[so["bucket"] < last_bucket]
            log.append(f"Last {('week' if freq == 'W' else 'month')} ({last_bucket:%d-%b-%Y}) is incomplete "
                       f"(data to {last_date:%d-%b-%Y}) and was excluded from history.")
            last_bucket = so["bucket"].max()
    first_bucket = so["bucket"].min()
    horizon = cfg.resolved_horizon(freq)
    key_cols = ["sku", "channel"] if cfg.level == "sku_channel" else ["sku"]
    log.append(f"Native data: {native_days}-day buckets ({native}); planning in "
               f"{'weekly' if freq == 'W' else 'monthly'} buckets, horizon {horizon}.")

    # ---------------- calendar (with auto-alignment of sale dates) ----------------
    say(0.06, "Building festival & sale calendar")
    native_total = so.groupby(so["date"].dt.normalize() if native == "D" else so["bucket"])["qty"].sum().sort_index()
    if native == "D":
        native_total = native_total.asfreq("D", fill_value=0.0)
    horizon_end = last_bucket + pd.Timedelta(days=(horizon + 2) * (7 if freq == "W" else 31))
    years = range(first_bucket.year - 1, horizon_end.year + 2)
    plan_start_guess = last_bucket + (pd.Timedelta(days=7) if freq == "W" else pd.offsets.MonthBegin(1))
    occ, cal_log = cal.build_calendar(years, inputs.events, cfg.disabled_events,
                                      native=native_total if native in {"D", "W"} else None,
                                      native_days=1 if native == "D" else 7,
                                      auto_align=cfg.auto_align_sale_dates, as_of=plan_start_guess)
    grid = make_grid(freq, first_bucket, last_bucket, horizon, occ)
    plan_start = grid.starts[grid.n_hist]

    # ---------------- history panel ----------------
    say(0.1, "Aggregating history")
    hist = so.groupby(key_cols + ["bucket"], as_index=False)["qty"].sum()
    hist["qty"] = hist["qty"].clip(lower=0)
    keys = sorted({_key_tuple(r, key_cols) for _, r in hist[key_cols].drop_duplicates().iterrows()})
    kidx = {k: i for i, k in enumerate(keys)}
    bidx = {b: i for i, b in enumerate(grid.starts[: grid.n_hist])}
    Y = np.zeros((grid.n_hist, len(keys)))
    for r in hist.itertuples(index=False):
        k = tuple(getattr(r, c) for c in key_cols)
        if r.bucket in bidx:
            Y[bidx[r.bucket], kidx[k]] += r.qty
    C = np.zeros_like(Y, dtype=bool)

    # Stockout censoring from WMS history.
    if inputs.wms is not None and "date" in inputs.wms and inputs.wms["date"].nunique() > 3:
        w = inputs.wms.dropna(subset=["date"]).copy()
        w["bucket"] = bucket_start(w["date"], freq, week_start)
        daily = w.groupby(["sku", "date", "bucket"], as_index=False)["on_hand"].sum()
        oos = daily.groupby(["sku", "bucket"])["on_hand"].min() <= 0
        n_c = 0
        for (sku, b), flag in oos.items():
            if flag and b in bidx:
                for k, j in kidx.items():
                    if k[0] == sku:
                        C[bidx[b], j] = True
                        n_c += 1
        if n_c:
            log.append(f"{n_c} SKU-buckets had zero DC stock in WMS history; excluded from model fitting "
                       f"(lost sales are not treated as low demand).")
    # Stockout censoring from reported channel stock.
    if "channel_stock" in so and so["channel_stock"].notna().any():
        cs = so.groupby(key_cols + ["bucket"])["channel_stock"].sum(min_count=1)
        for k_b, v in cs.items():
            k_b = k_b if isinstance(k_b, tuple) else (k_b,)
            k, b = tuple(k_b[:-1]), k_b[-1]
            if pd.notna(v) and v <= 0 and k in kidx and b in bidx:
                C[bidx[b], kidx[k]] = True
    # Suspected stockouts: runs of zero sales in an otherwise healthy selling series.
    suspected = 0
    for j in range(len(keys)):
        y = Y[:, j]
        nz = np.nonzero(y > 0)[0]
        if len(nz) < 10:
            continue
        t = nz[0]
        while t < grid.n_hist:
            if y[t] == 0:
                e = t
                while e < grid.n_hist and y[e] == 0:
                    e += 1
                before = y[max(nz[0], t - 8):t]
                if e < grid.n_hist and e - t >= 2 and len(before) >= 4 and np.median(before) >= 5:
                    C[t:e, j] = True
                    suspected += 1
                t = e
            else:
                t += 1
    if suspected:
        dq.append(dict(severity="warning", dataset="sell_out",
                       issue=f"{suspected} runs of zero sales in actively selling items look like stockouts; "
                             f"they were excluded from model fitting. Confirm with WMS stock history."))

    # ---------------- attributes ----------------
    names, cats = {}, {}
    if inputs.sku_master is not None:
        m = inputs.sku_master
        if "product_name" in m:
            names.update(m.set_index("sku")["product_name"].dropna().to_dict())
        if "category" in m:
            cats.update(m.set_index("sku")["category"].dropna().to_dict())
    for col, target in (("product_name", names), ("category", cats)):
        if col in so:
            for sku, v in so.dropna(subset=[col]).groupby("sku")[col].agg(lambda s: s.mode().iloc[0]).items():
                target.setdefault(sku, v)

    # ---------------- pools for borrowing strength ----------------
    pools: dict[str, list[int]] = {}
    for j, k in enumerate(keys):
        g = (k[1] if cfg.level == "sku_channel" else cats.get(k[0], "ALL"))
        pools.setdefault(str(g), []).append(j)
    small = [p for p, idx in pools.items() if len(idx) < 3 and p != "ALL"]
    for p in small:
        pools.setdefault("ALL", []).extend(pools.pop(p))
    if len(pools) > 1:
        log.append("Festival lifts & seasonality are pooled within: " + ", ".join(sorted(pools)) + ".")

    # ---------------- forecasting ----------------
    say(0.15, "Forecasting")
    series, cache = run_portfolio(Y, C, keys, pools, grid, cfg.backtest_folds,
                                  progress=lambda f, m: say(0.15 + 0.6 * f, m))

    # ---------------- assemble tables ----------------
    say(0.78, "Explaining trend, seasonality and festival shifts")
    fut_starts = grid.starts[grid.n_hist:]
    fut_labels = grid.labels[grid.n_hist:]

    def key_dict(k):
        d = dict(zip(key_cols, k))
        d["product_name"] = names.get(k[0], "")
        d["category"] = cats.get(k[0], "")
        return d

    fc_rows, acc_rows, score_rows, prof_rows, lift_rows, bridge_rows, anom_rows, hist_rows = ([] for _ in range(8))
    member_pool = cache.member_pool
    for j, r in enumerate(series):
        kd = key_dict(r.key)
        for h in range(grid.horizon):
            fc_rows.append({**kd, "bucket": fut_starts[h], "forecast": r.forecast[h], "p10": r.p10[h],
                            "p90": r.p90[h], "baseline": r.baseline[h], "event_factor": r.event_factor[h],
                            "events": fut_labels[h], "model": MODEL_LABELS.get(r.model, r.model)})
        for t in range(grid.n_hist):
            if t >= r.i0 or Y[t, j] > 0:
                hist_rows.append({**{c: v for c, v in zip(key_cols, r.key)}, "bucket": grid.starts[t],
                                  "qty": Y[t, j], "censored": bool(C[t, j])})
        members = r.scores.get("ensemble", {}).get("members", []) if r.model == "ensemble" else []
        acc_rows.append({**kd, "model": MODEL_LABELS.get(r.model, r.model),
                         "ensemble_of": " + ".join(MODEL_LABELS[m] for m in members),
                         "wape": r.wape, "accuracy": (max(0.0, 1 - r.wape) if r.wape is not None else None),
                         "bias": r.bias, "naive_wape": r.naive_wape,
                         "fva": (r.naive_wape - r.wape) if (r.wape is not None and r.naive_wape is not None) else None,
                         "backtest_folds": r.n_folds, "status": r.status, "demand_class": r.demand_class,
                         "interval_sigma": r.sigma, "notes": " ".join(r.notes)})
        for m, s in r.scores.items():
            score_rows.append({**kd, "model": MODEL_LABELS.get(m, m), "wape": s["wape"], "bias": s["bias"],
                               "selected": m == r.model})
        p = profile(r, Y[:, j], grid)
        si = p.pop("seasonal_index", None)
        prow = {**kd, **p}
        if si:
            prow.update({f"idx_{m}": v for m, v in si.items()})
        pool_fit = cache.fit_for(member_pool[j], grid.n_hist)
        lifts = event_lifts(r, grid, pool_fit.gamma if pool_fit is not None else None)
        top = sorted(lifts, key=lambda d: -abs(d["incremental_last_12m"]))[:3]
        prow["top_events"] = "; ".join(f"{_event_name(d['event'], occ)} ({d['incremental_last_12m']:+,.0f} units)"
                                       for d in top if abs(d["incremental_last_12m"]) >= 1)
        prof_rows.append(prow)
        for d in lifts:
            lift_rows.append({**kd, "event_name": _event_name(d["event"], occ), **d})
        for b in shift_bridge(r, Y[:, j], grid):
            bridge_rows.append({**kd, **b})
        if r.fit is not None:
            fit = r.fit
            exp_units = fit.fitted_rate * grid.days[r.i0:grid.n_hist] * fit.event_factor
            for t in np.nonzero(fit.anomalies)[0]:
                gi = r.i0 + t
                anom_rows.append({**kd, "bucket": grid.starts[gi], "actual": Y[gi, j], "expected": exp_units[t],
                                  "deviation_pct": (Y[gi, j] / max(exp_units[t], 1e-9) - 1) * 100,
                                  "events_in_bucket": grid.labels[gi]})

    forecast = pd.DataFrame(fc_rows)
    history = pd.DataFrame(hist_rows)
    accuracy = pd.DataFrame(acc_rows)
    model_scores = pd.DataFrame(score_rows)
    profiles = pd.DataFrame(prof_rows)
    lifts_df = pd.DataFrame(lift_rows)
    bridge = pd.DataFrame(bridge_rows)
    anomalies = pd.DataFrame(anom_rows)
    if not anomalies.empty:
        anomalies = anomalies[anomalies["bucket"] >= grid.starts[max(0, grid.n_hist - 2 * grid.lag)]]
        anomalies = anomalies.sort_values("deviation_pct", key=np.abs, ascending=False)

    # Portfolio-level lifts (pool fits)
    port_rows = []
    for pool in pools:
        pf = cache.fit_for(pool, grid.n_hist)
        if pf is None:
            continue
        for ev in grid.event_keys:
            row = {"pool": pool, "event": _event_name(ev, occ)}
            for ph in ("pre", "main", "post"):
                nm = f"{ev}|{ph}"
                row[f"{ph}_lift_pct"] = float(pf.gamma[grid.x_names.index(nm)] * 100) if nm in grid.x_names else None
            port_rows.append(row)
    portfolio_events = pd.DataFrame(port_rows)

    # ---------------- sell-in & supply ----------------
    say(0.88, "Planning sell-in and DC supply")
    sellin_df, supply_df, alerts, orders = _plan_supply(inputs, cfg, grid, freq, week_start, key_cols, keys,
                                                        series, Y, names, dq, log)

    # ---------------- data quality ----------------
    _quality_checks(inputs, so, grid, keys, dq, last_date)

    # ---------------- headline summary ----------------
    summary = _summary(grid, freq, forecast, bridge, accuracy, alerts, occ, plan_start, keys, series)
    shifts = cal.shift_table(occ)
    say(1.0, "Done")
    return PlanningResult(config=cfg, freq=freq, grid=grid, key_cols=key_cols, plan_start=plan_start,
                          calendar=occ, calendar_shifts=shifts, calendar_log=cal_log, history=history,
                          forecast=forecast, accuracy=accuracy, model_scores=model_scores, profiles=profiles,
                          event_lifts=lifts_df, portfolio_events=portfolio_events, shift_bridge=bridge,
                          anomalies=anomalies, sellin_plan=sellin_df, supply_plan=supply_df, alerts=alerts,
                          orders=orders, data_quality=pd.DataFrame(dq, columns=["severity", "dataset", "issue"]),
                          summary=summary, log=log, series=series)


# ------------------------------------------------------------------------------------------------
def _plan_supply(inputs, cfg, grid, freq, week_start, key_cols, keys, series, Y, names, dq, log):
    H = grid.horizon
    fut = grid.starts[grid.n_hist:]
    bucket_days = float(np.median(grid.days))
    weeks_per_bucket = bucket_days / 7.0
    cover_buckets = cfg.channel_cover_weeks / weeks_per_bucket
    window = 13 if freq == "W" else 3
    plan_start = fut[0]

    # Sell-in history on the same buckets/keys.
    SI = None
    if inputs.sell_in is not None and not inputs.sell_in.empty:
        si = inputs.sell_in.copy()
        if "channel" not in si:
            si["channel"] = "ALL"
        si["bucket"] = bucket_start(si["date"], freq, week_start)
        agg = si.groupby([c for c in key_cols] + ["bucket"])["qty"].sum()
        SI = np.zeros_like(Y)
        bidx = {b: i for i, b in enumerate(grid.starts[: grid.n_hist])}
        kidx = {k: i for i, k in enumerate(keys)}
        for kb, q in agg.items():
            kb = kb if isinstance(kb, tuple) else (kb,)
            k, b = tuple(kb[:-1]), kb[-1]
            if k in kidx and b in bidx:
                SI[bidx[b], kidx[k]] += q
        lag = grid.lag
        tot_si, tot_so = SI[-lag:].sum(), Y[-lag:].sum()
        if tot_so > 0 and tot_si / tot_so > 1.2:
            dq.append(dict(severity="warning", dataset="sell_in",
                           issue=f"Sell-in exceeded sell-out by {tot_si / tot_so - 1:.0%} over the last year: either "
                                 f"channel stock is building up or sell-out data does not cover every channel."))

    # Reported channel stock (latest bucket), if any.
    cs_reported = {}
    so = inputs.sell_out
    if "channel_stock" in so and so["channel_stock"].notna().any():
        latest = so.dropna(subset=["channel_stock"])
        latest = latest[latest["date"] == latest["date"].max()]
        cs_reported = latest.groupby(key_cols)["channel_stock"].sum().to_dict()
        cs_reported = {(k if isinstance(k, tuple) else (k,)): v for k, v in cs_reported.items()}

    sellin_rows = []
    sku_demand: dict[str, np.ndarray] = {}
    sku_sigma: dict[str, list[tuple[float, float]]] = {}
    sku_channel_stock: dict[str, float] = {}
    for j, (k, r) in enumerate(zip(keys, series)):
        if k in cs_reported:
            cs0, how = float(cs_reported[k]), "reported"
        else:
            cs0, how = sup.estimate_channel_stock(SI[:, j] if SI is not None else None, Y[:, j], cover_buckets, window)
        sellin, stock, target = sup.sellin_plan(r.forecast, cs0, cover_buckets)
        sku = k[0]
        sku_demand[sku] = sku_demand.get(sku, np.zeros(H)) + sellin
        sku_sigma.setdefault(sku, []).append((r.sigma, float(r.forecast.sum())))
        sku_channel_stock[sku] = sku_channel_stock.get(sku, 0.0) + cs0
        for h in range(H):
            sellin_rows.append({**dict(zip(key_cols, k)), "product_name": names.get(sku, ""), "bucket": fut[h],
                                "sell_out_forecast": r.forecast[h], "sell_in_plan": sellin[h],
                                "channel_stock_end": stock[h], "channel_target": target[h],
                                "channel_stock_start": cs0 if h == 0 else None,
                                "channel_stock_basis": how if h == 0 else ""})
    sellin_df = pd.DataFrame(sellin_rows)

    if inputs.wms is None or inputs.wms.empty:
        log.append("No WMS file: sell-in plan produced, DC supply plan skipped.")
        return sellin_df, pd.DataFrame(), _empty_alerts(), pd.DataFrame()

    w = inputs.wms.copy()
    if "date" in w and w["date"].notna().any():
        latest_date = w["date"].max()
        snap = w[(w["date"] == latest_date) | w["date"].isna()]
        snapshot_note = f"WMS snapshot {latest_date:%d-%b-%Y}"
    else:
        snap = w
        snapshot_note = "WMS snapshot (undated)"
    log.append(f"{snapshot_note}; plan starts {plan_start:%d-%b-%Y}.")
    on_hand = snap.groupby("sku")["on_hand"].sum().to_dict()
    master = inputs.sku_master.set_index("sku") if inputs.sku_master is not None else pd.DataFrame()

    def attr(sku, col):
        if col in master.columns and sku in master.index and pd.notna(master.at[sku, col]):
            return float(master.at[sku, col])
        if col in snap.columns:
            v = snap.loc[snap["sku"] == sku, col].dropna()
            if len(v):
                return float(v.iloc[0])
        return None

    # Prices for value-at-risk (master price, else average selling price).
    asp = {}
    if "value" in so:
        recent = so[so["date"] >= so["date"].max() - pd.Timedelta(days=365)]
        g = recent.groupby("sku")[["value", "qty"]].sum()
        asp = (g["value"] / g["qty"].replace(0, np.nan)).dropna().to_dict()

    supply_rows, alert_rows, order_rows = [], [], []
    today = plan_start
    all_skus = sorted(set(sku_demand) | set(on_hand))
    for sku in all_skus:
        demand = sku_demand.get(sku, np.zeros(H))
        oh = float(on_hand.get(sku, 0.0))
        lt_days = attr(sku, "lead_time_days") or cfg.default_lead_time_days
        lt = int(math.ceil(lt_days / bucket_days))
        moq, case = attr(sku, "moq"), attr(sku, "case_pack")
        price = attr(sku, "price") or asp.get(sku)
        sig_pairs = sku_sigma.get(sku, [(0.4, 1.0)])
        tot = sum(p[1] for p in sig_pairs) or 1.0
        sigma = sum(p[0] * p[1] for p in sig_pairs) / tot
        cycle = max(1, int(round(cfg.order_cycle_weeks / weeks_per_bucket)))
        ss = sup.safety_stock(sigma, demand, lt, cfg.service_level, cycle)
        receipts = np.zeros(H)
        overdue = 0.0
        rows = snap[snap["sku"] == sku]
        for _, rr in rows.iterrows():
            q = float(rr.get("in_transit", 0) or 0)
            if q <= 0:
                continue
            eta = rr.get("eta")
            if eta is None or pd.isna(eta):
                receipts[0] += q
                continue
            i = int((pd.Timestamp(eta) - plan_start).days // bucket_days)
            if i < 0:
                overdue += q
                receipts[0] += q
            elif i < H:
                receipts[i] += q
        plan = sup.dc_plan(demand, oh, receipts, lt, ss, moq, case, cycle)
        weekly_demand = max(demand[: max(1, int(round(8 / weeks_per_bucket)))].mean() / weeks_per_bucket, 1e-9)
        pname = names.get(sku, "")
        for h in range(H):
            release = fut[h] - pd.Timedelta(days=lt_days)
            q = plan["planned_receipt"][h]
            supply_rows.append({"sku": sku, "product_name": pname, "bucket": fut[h],
                                "dc_demand_sell_in": demand[h], "scheduled_receipts": receipts[h],
                                "planned_receipts": q, "expedite": bool(plan["expedite"][h]),
                                "order_release_date": (today.date() if plan["expedite"][h] else release.date()) if q > 0 else None,
                                "projected_on_hand": plan["projected"][h],
                                "projected_if_no_new_orders": plan["projected_no_new_orders"][h],
                                "safety_stock": ss, "weeks_of_cover": plan["projected"][h] / weekly_demand})
            if q > 0 and not plan["expedite"][h]:
                status = "PLACE NOW" if release <= today + pd.Timedelta(days=bucket_days) else "PLANNED"
                order_rows.append({"sku": sku, "product_name": pname, "status": status, "order_qty": q,
                                   "release_date": release.date(), "needed_by": fut[h].date(),
                                   "lead_time_days": lt_days, "value": q * price if price else None})
        exp_idx = np.nonzero(plan["expedite"])[0]
        cover_now = oh / weekly_demand
        channel_cover = sku_channel_stock.get(sku, 0.0) / weekly_demand
        if len(exp_idx):
            first = int(exp_idx[0])
            short = float(plan["planned_receipt"][exp_idx].sum())
            order_rows.append({"sku": sku, "product_name": pname, "status": "EXPEDITE", "order_qty": short,
                               "release_date": today.date(), "needed_by": fut[first].date(),
                               "lead_time_days": lt_days, "value": short * price if price else None})
            severe = first < max(1, math.ceil(lt / 2)) or short > weekly_demand * weeks_per_bucket
            alert_rows.append(dict(sku=sku, product_name=pname, severity="critical" if severe else "high",
                                   alert="Stockout inside lead time",
                                   detail=f"DC stock runs out in the {freq_label(freq)} of {fut[first]:%d-%b-%Y} "
                                          f"({first + 1} {freq_label(freq)}(s) out); a normal PO takes {lt_days:.0f} days. "
                                          f"Expedite / air-freight / reallocate ~{short:,.0f} units.",
                                   units=short, value=short * price if price else None, when=fut[first].date()))
        now_orders = [o for o in order_rows if o["sku"] == sku and o["status"] == "PLACE NOW"]
        if now_orders:
            o = now_orders[0]
            alert_rows.append(dict(sku=sku, product_name=pname, severity="high", alert="Place PO now",
                                   detail=f"Order {o['order_qty']:,.0f} units now to land by {o['needed_by']:%d-%b-%Y} "
                                          f"and stay above safety stock.",
                                   units=o["order_qty"], value=o["value"], when=o["needed_by"]))
        low = np.nonzero((plan["projected"] < ss) & ~plan["expedite"])[0]
        low = low[low < lt]
        if len(low) and not len(exp_idx):
            first = int(low[0])
            alert_rows.append(dict(sku=sku, product_name=pname, severity="medium", alert="Below safety stock",
                                   detail=f"DC stock dips below safety stock ({ss:,.0f}) in the {freq_label(freq)} of "
                                          f"{fut[first]:%d-%b-%Y}; service risk if demand runs above forecast.",
                                   units=float(ss - plan["projected"][first]), value=None, when=fut[first].date()))
        total_cover = cover_now + channel_cover
        if total_cover > cfg.excess_cover_weeks and weekly_demand > 1e-6:
            excess_units = (total_cover - cfg.excess_cover_weeks) * weekly_demand
            alert_rows.append(dict(sku=sku, product_name=pname, severity="medium", alert="Excess stock",
                                   detail=f"{total_cover:,.0f} weeks of cover (DC {cover_now:,.0f} + channel "
                                          f"{channel_cover:,.0f}) vs {cfg.excess_cover_weeks:.0f}-week limit; "
                                          f"hold POs or plan activation.",
                                   units=excess_units, value=excess_units * price if price else None, when=None))
        if sku not in sku_demand and oh > 0:
            alert_rows.append(dict(sku=sku, product_name=names.get(sku, ""), severity="medium", alert="No sales",
                                   detail=f"{oh:,.0f} units in the DC but no sell-out history: check listing/obsolescence.",
                                   units=oh, value=oh * price if price else None, when=None))
        if overdue > 0:
            alert_rows.append(dict(sku=sku, product_name=names.get(sku, ""), severity="medium", alert="Overdue inbound",
                                   detail=f"{overdue:,.0f} units in transit are past their ETA; assumed to arrive now.",
                                   units=overdue, value=None, when=None))
    alerts = pd.DataFrame(alert_rows, columns=["sku", "product_name", "severity", "alert", "detail", "units",
                                               "value", "when"])
    if not alerts.empty:
        order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        alerts = alerts.sort_values(["severity", "units"], key=lambda s: s.map(order) if s.name == "severity" else -s)
    orders = pd.DataFrame(order_rows)
    if not orders.empty:
        orders = orders.sort_values(["release_date", "sku"]).reset_index(drop=True)
    return sellin_df, pd.DataFrame(supply_rows), alerts.reset_index(drop=True), orders


def freq_label(freq: str) -> str:
    return "week" if freq == "W" else "month"


def _empty_alerts() -> pd.DataFrame:
    return pd.DataFrame(columns=["sku", "product_name", "severity", "alert", "detail", "units", "value", "when"])


def _quality_checks(inputs, so, grid, keys, dq, last_date):
    skus = {k[0] for k in keys}
    if inputs.wms is not None and not inputs.wms.empty:
        wms_skus = set(inputs.wms["sku"])
        missing = sorted(skus - wms_skus)
        if missing:
            dq.append(dict(severity="warning", dataset="wms",
                           issue=f"{len(missing)} selling SKUs are not in the WMS file (treated as zero stock): "
                                 + ", ".join(missing[:8]) + ("..." if len(missing) > 8 else "")))
        if "date" in inputs.wms and inputs.wms["date"].notna().any():
            gap = (last_date - inputs.wms["date"].max()).days
            if abs(gap) > 10:
                dq.append(dict(severity="warning", dataset="wms",
                               issue=f"WMS snapshot is {abs(gap)} days {'older' if gap > 0 else 'newer'} than the last "
                                     f"sell-out date; stock position may be stale."))
    if inputs.sell_in is not None and not inputs.sell_in.empty:
        extra = sorted(set(inputs.sell_in["sku"]) - skus)
        if extra:
            dq.append(dict(severity="info", dataset="sell_in",
                           issue=f"{len(extra)} SKUs have sell-in but no sell-out (new listings or missing sell-out): "
                                 + ", ".join(extra[:8])))
    total = so.groupby("bucket")["qty"].sum()
    missing_buckets = [b for b in grid.starts[: grid.n_hist] if b not in total.index]
    if missing_buckets:
        dq.append(dict(severity="warning", dataset="sell_out",
                       issue=f"{len(missing_buckets)} {freq_label(grid.freq)}s have no sell-out rows at all "
                             f"(e.g. {missing_buckets[0]:%d-%b-%Y}); treated as zero sales."))
    dups = so.duplicated(subset=["date", "sku", "channel", "qty"]).sum()
    if dups:
        dq.append(dict(severity="info", dataset="sell_out",
                       issue=f"{dups} exact duplicate rows found and summed; remove them if they are double exports."))
    age = (pd.Timestamp.today().normalize() - last_date).days
    if age > 45:
        dq.append(dict(severity="warning", dataset="sell_out",
                       issue=f"Latest sell-out is {age} days old ({last_date:%d-%b-%Y}); refresh for a current plan."))


def _summary(grid, freq, forecast, bridge, accuracy, alerts, occ, plan_start, keys, series) -> dict:
    near = 13 if freq == "W" else 3
    near_end = grid.starts[min(grid.n_hist + near, len(grid.starts) - 1)]
    fc_near = forecast[forecast["bucket"] < near_end]["forecast"].sum()
    b_near = bridge[bridge["bucket"] < near_end] if not bridge.empty else bridge
    ly = b_near["ly_actual"].sum() if not b_near.empty else None
    aligned = b_near["ly_aligned"].sum() if not b_near.empty else None
    acc = accuracy.dropna(subset=["wape"])
    vol = {tuple(k): float(r.forecast.sum()) for k, r in zip(keys, series)}
    weights = np.array([vol.get(tuple(row[c] for c in (["sku", "channel"] if "channel" in acc else ["sku"])), 0.0)
                        for _, row in acc.iterrows()]) if not acc.empty else np.array([])
    wape = float(np.average(acc["wape"], weights=weights)) if weights.sum() > 0 else None
    bias = float(np.average(acc["bias"], weights=weights)) if weights.sum() > 0 else None
    horizon_end = grid.starts[-1]
    moves = cal.shift_table(occ)
    moves = moves[(pd.to_datetime(moves["start"]) >= plan_start) & (pd.to_datetime(moves["start"]) <= horizon_end)]
    big = moves[(moves["shift_vs_ly_days"].abs().fillna(0) >= 7) | (moves["month_change"] != "")]
    to_confirm = moves[moves["confidence"].isin(["estimated", "verify"])]
    return {
        "plan_start": plan_start.date(),
        "horizon_buckets": grid.horizon,
        "bucket": freq_label(freq),
        "series": len(keys),
        "forecast_total": float(forecast["forecast"].sum()),
        f"forecast_next_{near}": float(fc_near),
        f"ly_same_{near}": float(ly) if ly is not None else None,
        f"ly_aligned_next_{near}": float(aligned) if aligned is not None else None,
        "vs_ly_pct": (fc_near / ly - 1) * 100 if ly else None,
        "vs_aligned_ly_pct": (fc_near / aligned - 1) * 100 if aligned else None,
        "portfolio_wape": wape,
        "portfolio_accuracy": (1 - wape) if wape is not None else None,
        "portfolio_bias": bias,
        "critical_alerts": int((alerts["severity"] == "critical").sum()) if not alerts.empty else 0,
        "high_alerts": int((alerts["severity"] == "high").sum()) if not alerts.empty else 0,
        "festival_moves_in_horizon": [f"{r.event} {r.year}: {r.start:%d-%b} ({r.shift_vs_ly_days:+.0f} days vs LY"
                                      + (f", {r.month_change}" if r.month_change else "") + ")"
                                      for r in big.itertuples()],
        "dates_to_confirm": [f"{r.event} {r.year}: {r.start:%d-%b}-{r.end:%d-%b} ({r.source})"
                             for r in to_confirm.itertuples()],
        "models_used": accuracy["model"].value_counts().to_dict(),
    }
