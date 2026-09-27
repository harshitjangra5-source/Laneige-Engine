import numpy as np
import pandas as pd

from laneige_engine import calendar as cal
from laneige_engine.config import EngineConfig
from laneige_engine.forecaster import make_grid, run_portfolio
from laneige_engine.pipeline import PlanningInputs, run_planning


def _simulate_daily(first, last, occ, base=40.0, lift=2.5, seed=0, sku="S1"):
    rng = np.random.default_rng(seed)
    days = pd.date_range(first, last, freq="D")
    mult = np.ones(len(days))
    for _, r in occ[occ["event"] == "diwali"].iterrows():
        m = (days >= r["start"] - pd.Timedelta(days=21)) & (days < r["start"])
        mult[m] *= lift
    qty = rng.poisson(base * mult)
    return pd.DataFrame({"date": days, "sku": sku, "channel": "ALL", "qty": qty.astype(float)})


def test_weekly_forecast_moves_diwali_uplift_to_new_dates():
    occ = cal.base_occurrences(range(2022, 2028))
    df = pd.concat([_simulate_daily("2023-01-02", "2026-09-20", occ, seed=s, sku=f"S{s}") for s in range(3)])
    res = run_planning(PlanningInputs(sell_out=df), EngineConfig(horizon=12))
    fc = res.forecast.groupby("bucket")["forecast"].sum()
    in_2026_window = fc[(fc.index >= "2026-10-19") & (fc.index <= "2026-11-02")].mean()
    in_2025_window_dates = fc[(fc.index >= "2026-09-28") & (fc.index <= "2026-10-05")].mean()
    assert in_2026_window > 1.8 * in_2025_window_dates
    # Last year's Diwali weeks (Sep 29 - Oct 19, 2025) must NOT be copied onto the same 2026 weeks.
    b = res.shift_bridge.groupby("bucket")[["ly_actual", "ly_aligned"]].sum()
    oct5 = b.loc[pd.Timestamp("2026-10-05")]
    assert oct5["ly_aligned"] < 0.6 * oct5["ly_actual"]


def test_monthly_data_october_down_november_up():
    occ = cal.base_occurrences(range(2021, 2028))
    daily = _simulate_daily("2022-01-01", "2026-08-31", occ, base=100, lift=3.0)
    monthly = daily.assign(date=daily["date"].dt.to_period("M").dt.start_time).groupby(
        ["date", "sku", "channel"], as_index=False)["qty"].sum()
    res = run_planning(PlanningInputs(sell_out=monthly), EngineConfig())
    assert res.freq == "M"
    fc = res.forecast.set_index("bucket")["forecast"]
    hist = monthly.set_index("date")["qty"]
    # Diwali: 20-Oct-2025 (pre-window mostly Oct) vs 8-Nov-2026 (pre-window split Oct/Nov)
    assert fc[pd.Timestamp("2026-10-01")] < hist[pd.Timestamp("2025-10-01")]
    assert fc[pd.Timestamp("2026-11-01")] > 1.2 * hist[pd.Timestamp("2025-11-01")]


def test_inactive_and_intermittent_series():
    occ = cal.base_occurrences(range(2023, 2027))
    grid = make_grid("W", pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-29"), 8, occ)
    n = grid.n_hist
    rng = np.random.default_rng(3)
    Y = np.zeros((n, 3))
    Y[:40, 0] = 20                                           # stopped selling long ago
    Y[:, 1] = rng.poisson(0.4, n) * rng.integers(1, 6, n)    # intermittent
    Y[:, 2] = rng.poisson(50, n)                             # smooth
    res, _ = run_portfolio(Y, np.zeros_like(Y, dtype=bool), [("A",), ("B",), ("C",)],
                           {"ALL": [0, 1, 2]}, grid, folds=3)
    assert res[0].status == "inactive" and res[0].forecast.sum() == 0
    assert res[1].demand_class in {"intermittent", "lumpy"}
    assert "croston" in res[1].scores
    assert np.all(np.isfinite(res[2].forecast)) and res[2].forecast.mean() > 30


def test_stockout_zero_run_is_censored():
    occ = cal.base_occurrences(range(2023, 2027))
    df = _simulate_daily("2024-01-01", "2026-06-28", occ, base=30)
    gap = (df["date"] >= "2025-06-02") & (df["date"] <= "2025-06-22")
    df.loc[gap, "qty"] = 0
    res = run_planning(PlanningInputs(sell_out=df), EngineConfig(horizon=8))
    h = res.history.set_index("bucket")
    assert h.loc[pd.Timestamp("2025-06-09"), "censored"]
    assert any("look like stockouts" in i for i in res.data_quality["issue"])
