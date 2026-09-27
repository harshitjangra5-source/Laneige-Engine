"""Portfolio forecasting: per-series backtests, automatic model selection, prediction intervals,
trend / seasonality profiles, event lifts and the festival-shift bridge."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import calendar as cal
from .models import (PoolPrior, SeriesFit, croston_sba, demand_class, fit_series, holt_damped,
                     pool_prior_from, theta)

MODEL_LABELS = {
    "regression": "Seasonal regression + events",
    "ets": "Damped-trend ETS + events",
    "theta": "Theta + events",
    "ly_aligned": "Last year festival-aligned x growth",
    "moving_avg": "Seasonal moving average + events",
    "croston": "Croston SBA (intermittent)",
    "ensemble": "Ensemble (best two)",
    "launch_avg": "Launch average (very new item)",
    "inactive": "Inactive (no recent sales)",
}
Z80 = 1.2816


@dataclass
class Grid:
    freq: str
    starts: pd.DatetimeIndex
    days: np.ndarray
    n_hist: int
    ty: np.ndarray
    doy: np.ndarray
    X: np.ndarray
    x_names: list[str]
    labels: list[str]
    lag: int
    ppy: float

    @property
    def horizon(self) -> int:
        return len(self.starts) - self.n_hist

    @property
    def event_keys(self) -> list[str]:
        return sorted({n.split("|")[0] for n in self.x_names})


def bucket_start(dates: pd.Series, freq: str, week_start: int = 0) -> pd.Series:
    d = pd.to_datetime(dates).dt.normalize()
    if freq == "M":
        return d.dt.to_period("M").dt.start_time
    return d - pd.to_timedelta((d.dt.weekday - week_start) % 7, unit="D")


def make_grid(freq: str, first: pd.Timestamp, last_hist: pd.Timestamp, horizon: int,
              occ: pd.DataFrame) -> Grid:
    if freq == "M":
        hist = pd.date_range(first, last_hist, freq="MS")
        starts = pd.date_range(first, periods=len(hist) + horizon, freq="MS")
        days = starts.days_in_month.to_numpy().astype(float)
        lag = 12
    else:
        hist = pd.date_range(first, last_hist, freq="7D")
        starts = pd.date_range(first, periods=len(hist) + horizon, freq="7D")
        days = np.full(len(starts), 7.0)
        lag = 52
    ty = ((starts - starts[0]).days.to_numpy() / 365.25).astype(float)
    mid = starts + pd.to_timedelta(days / 2, unit="D")
    doy = mid.dayofyear.to_numpy().astype(float)
    X, names = cal.event_features(starts, days.astype(np.int64), occ)
    labels = cal.events_by_period(starts, days.astype(np.int64), occ)
    return Grid(freq=freq, starts=starts, days=days, n_hist=len(hist), ty=ty, doy=doy, X=X, x_names=names,
                labels=labels, lag=lag, ppy=365.25 / float(np.median(days)))


@dataclass
class SeriesResult:
    key: tuple
    i0: int
    status: str
    model: str
    forecast: np.ndarray
    p10: np.ndarray
    p90: np.ndarray
    baseline: np.ndarray
    event_factor: np.ndarray
    sigma: float
    scores: dict[str, dict] = field(default_factory=dict)
    wape: float | None = None
    bias: float | None = None
    naive_wape: float | None = None
    n_folds: int = 0
    demand_class: str = ""
    adi: float = 0.0
    cv2: float = 0.0
    fit: SeriesFit | None = None
    notes: list[str] = field(default_factory=list)


# ------------------------------------------------------------------------------------------------
class PoolCache:
    """Pool (category / channel / portfolio) fits used as priors for member series, per cutoff."""

    def __init__(self, Y: np.ndarray, C: np.ndarray, pools: dict[str, list[int]], grid: Grid):
        self.Y, self.C, self.pools, self.grid = Y, C, pools, grid
        self.cache: dict[tuple[str, int], tuple[PoolPrior | None, SeriesFit | None]] = {}
        self.member_pool = {i: p for p, idx in pools.items() for i in idx}

    def get(self, series_idx: int, cut: int) -> PoolPrior | None:
        return self._fit(self.member_pool[series_idx], cut)[0]

    def fit_for(self, pool: str, cut: int) -> SeriesFit | None:
        return self._fit(pool, cut)[1]

    def _fit(self, pool: str, cut: int):
        key = (pool, cut)
        if key not in self.cache:
            g = self.grid
            idx = self.pools[pool]
            y = self.Y[:cut, idx].sum(axis=1)
            cens = self.C[:cut, idx].all(axis=1)
            nz = np.nonzero(y > 0)[0]
            if len(nz) == 0 or cut - nz[0] < 8:
                self.cache[key] = (None, None)
            else:
                i0 = nz[0]
                fit = fit_series(y[i0:], g.days[i0:cut], g.ty[i0:cut], g.doy[i0:cut], g.X[i0:cut],
                                 cens[i0:], prior=None, is_pool=True)
                self.cache[key] = (pool_prior_from(fit), fit)
        return self.cache[key]


def _model_rates(fit: SeriesFit, y_tr: np.ndarray, grid: Grid, i0: int, cut: int, fut: np.ndarray,
                 cls: str) -> dict[str, np.ndarray]:
    """Baseline daily-rate forecasts from each candidate model for buckets `fut`."""
    ty_f, doy_f = grid.ty[fut], grid.doy[fut]
    h = len(fut)
    out: dict[str, np.ndarray] = {"regression": fit.baseline_rate(ty_f, doy_f)}
    clean, c = fit.clean_rate, fit.c
    n = len(clean)
    seas_tr, seas_f = fit.seasonal_log(grid.doy[i0:cut]), fit.seasonal_log(doy_f)
    a = np.log(clean + c) - seas_tr
    if n >= 8:
        out["ets"] = np.maximum(np.exp(holt_damped(a, fit.phi, h) + seas_f) - c, 0.0)
        out["theta"] = np.maximum(np.exp(theta(a, h) + seas_f) - c, 0.0)
    lag = grid.lag
    m = 13 if lag == 52 else 3
    if n >= lag + m:
        g = clean[n - m:].sum() / max(clean[n - m - lag: n - lag].sum(), 1e-9)
        g = float(np.clip(g, 0.5, 2.0))
        steps = fut - cut
        k = steps // lag + 1
        src = n + steps - k * lag
        out["ly_aligned"] = clean[src] * g ** k
    mm = min(8 if lag == 52 else 3, n)
    if mm >= 2:
        adj = np.exp(seas_f - seas_tr[-mm:].mean())
        out["moving_avg"] = clean[-mm:].mean() * adj
    if cls in {"intermittent", "lumpy"}:
        mean_e = float(fit.event_factor.mean()) or 1.0
        out["croston"] = croston_sba(y_tr, h) / np.median(grid.days) / mean_e
    return out


def _event_factor(grid: Grid, idx: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    if grid.X.shape[1] == 0:
        return np.ones(len(idx))
    return np.clip(1.0 + grid.X[idx] @ gamma, 0.2, None)


def _score(A: np.ndarray, F: np.ndarray) -> tuple[float, float]:
    s = A.sum()
    if s <= 0:
        return (float(np.abs(F).sum() > 0), 0.0)
    return float(np.abs(A - F).sum() / s), float((F - A).sum() / s)


def forecast_series(j: int, key: tuple, y: np.ndarray, cens: np.ndarray, grid: Grid, pools: PoolCache,
                    folds: int) -> SeriesResult:
    n_hist, H, lag = grid.n_hist, grid.horizon, grid.lag
    fut = np.arange(n_hist, n_hist + H)
    days_f = grid.days[fut]
    nz = np.nonzero(y > 0)[0]
    zeros = np.zeros(H)
    if len(nz) == 0:
        return SeriesResult(key, n_hist, "inactive", "inactive", zeros, zeros, zeros, zeros, np.ones(H), 0.0,
                            notes=["No sales in history."])
    i0 = int(nz[0])
    inactive_window = 26 if lag == 52 else 6
    if y[max(i0, n_hist - inactive_window):].sum() <= 0 and n_hist - i0 > inactive_window:
        return SeriesResult(key, i0, "inactive", "inactive", zeros, zeros, zeros, zeros, np.ones(H), 0.0,
                            notes=[f"No sales in the last {inactive_window} buckets; forecast set to zero."])
    y_tr = y[i0:]
    cls, adi, cv2 = demand_class(y_tr[~cens[i0:]])
    n_train = n_hist - i0
    prior_full = pools.get(j, n_hist)

    if n_train < 6:
        E_f = _event_factor(grid, fut, prior_full.gamma if prior_full else np.zeros(grid.X.shape[1]))
        rate = y_tr.sum() / grid.days[i0:n_hist].sum()
        fc = rate * days_f * E_f
        sigma = 0.6
        return SeriesResult(key, i0, "new", "launch_avg", fc, fc * np.exp(-Z80 * sigma), fc * np.exp(Z80 * sigma),
                            rate * days_f, E_f, sigma, demand_class=cls, adi=adi, cv2=cv2,
                            notes=[f"Only {n_train} buckets of history: launch average x portfolio festival lifts."])

    # ---------------- rolling-origin backtest ----------------
    h_bt = min(13 if lag == 52 else 3, H)
    min_train = max(lag // 2, 8)
    errs: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    naive: list[tuple[np.ndarray, np.ndarray]] = []
    n_folds = 0
    for f in range(1, folds + 1):
        cut = n_hist - f * h_bt
        if cut - i0 < min_train:
            break
        idx = np.arange(cut, n_hist - (f - 1) * h_bt)
        prior = pools.get(j, cut)
        fit = fit_series(y[i0:cut], grid.days[i0:cut], grid.ty[i0:cut], grid.doy[i0:cut], grid.X[i0:cut],
                         cens[i0:cut], prior=prior)
        rates = _model_rates(fit, y[i0:cut], grid, i0, cut, idx, cls)
        E = _event_factor(grid, idx, fit.gamma)
        keep = ~cens[idx]
        A = y[idx][keep]
        for name, r in rates.items():
            errs[name].append((A, (r * grid.days[idx] * E)[keep]))
        ly = idx - lag
        nv = np.where(ly >= i0, y[np.maximum(ly, 0)], y[cut - 1])
        naive.append((A, nv[keep]))
        n_folds += 1

    scores: dict[str, dict] = {}
    for name, pairs in errs.items():
        A = np.concatenate([p[0] for p in pairs])
        F = np.concatenate([p[1] for p in pairs])
        w, b = _score(A, F)
        scores[name] = {"wape": w, "bias": b}
    if len(scores) >= 2:
        top2 = sorted(scores, key=lambda k: scores[k]["wape"])[:2]
        pairs = [(errs[top2[0]][i][0], (errs[top2[0]][i][1] + errs[top2[1]][i][1]) / 2)
                 for i in range(len(errs[top2[0]]))]
        errs["ensemble"] = pairs
        A = np.concatenate([p[0] for p in pairs])
        F = np.concatenate([p[1] for p in pairs])
        w, b = _score(A, F)
        scores["ensemble"] = {"wape": w, "bias": b, "members": top2}
    naive_wape = None
    if naive:
        naive_wape = _score(np.concatenate([p[0] for p in naive]), np.concatenate([p[1] for p in naive]))[0]

    if scores:
        best = min(scores, key=lambda k: (round(scores[k]["wape"], 4), k != "ensemble"))
    else:
        best = "regression"

    # ---------------- final fit on full history ----------------
    fit = fit_series(y_tr, grid.days[i0:n_hist], grid.ty[i0:n_hist], grid.doy[i0:n_hist], grid.X[i0:n_hist],
                     cens[i0:], prior=prior_full)
    rates = _model_rates(fit, y_tr, grid, i0, n_hist, fut, cls)
    if best == "ensemble":
        members = scores["ensemble"]["members"]
        base_rate = np.mean([rates[m] for m in members if m in rates], axis=0)
    else:
        base_rate = rates.get(best, rates["regression"])
    E_f = _event_factor(grid, fut, fit.gamma)
    baseline = base_rate * days_f
    fc = baseline * E_f

    # ---------------- intervals from backtest errors ----------------
    if best in errs:
        A = np.concatenate([p[0] for p in errs[best]])
        F = np.concatenate([p[1] for p in errs[best]])
        e = np.clip(np.log((A + 1.0) / (F + 1.0)), -2, 2)
        sigma = float(np.clip(np.sqrt(np.mean(e ** 2)), 0.05, 1.5)) if len(e) else 0.4
    else:
        sigma = 0.4
    steps = np.arange(1, H + 1)
    widen = np.sqrt(np.maximum(1.0, steps / max(h_bt, 1)))
    p10, p90 = fc * np.exp(-Z80 * sigma * widen), fc * np.exp(Z80 * sigma * widen)

    status = "new" if n_train < lag else "ok"
    notes = []
    if status == "new":
        notes.append("Less than one year of history: seasonality borrowed from similar products.")
    if n_folds == 0:
        notes.append("Not enough history to backtest; default model used.")
    wape = scores.get(best, {}).get("wape")
    bias = scores.get(best, {}).get("bias")
    return SeriesResult(key, i0, status, best, fc, p10, p90, baseline, E_f, sigma, scores, wape, bias, naive_wape,
                        n_folds, cls, adi, cv2, fit, notes)


def run_portfolio(Y: np.ndarray, C: np.ndarray, keys: list[tuple], pools: dict[str, list[int]], grid: Grid,
                  folds: int, progress: Callable[[float, str], None] | None = None
                  ) -> tuple[list[SeriesResult], PoolCache]:
    cache = PoolCache(Y, C, pools, grid)
    results = []
    S = len(keys)
    for j, key in enumerate(keys):
        if progress and (j % max(1, S // 50) == 0):
            progress(j / max(S, 1), f"Forecasting {' / '.join(map(str, key))} ({j + 1}/{S})")
        results.append(forecast_series(j, key, Y[:, j], C[:, j], grid, cache, folds))
    return results, cache


# ------------------------------------------------------------------------------------------------
# Explainability: profiles, event lifts, festival-shift bridge
# ------------------------------------------------------------------------------------------------
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_MID_MONTH_DOY = np.array([15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349], dtype=float)


def seasonal_index(fit: SeriesFit) -> np.ndarray:
    s = fit.seasonal_log(_MID_MONTH_DOY)
    return np.exp(s - s.mean())


def profile(res: SeriesResult, y: np.ndarray, grid: Grid) -> dict:
    fit = res.fit
    lag, n_hist = grid.lag, grid.n_hist
    out = {"status": res.status, "demand_class": res.demand_class, "history_buckets": n_hist - res.i0,
           "first_sale": grid.starts[res.i0].date() if res.i0 < n_hist else None}
    recent = 13 if lag == 52 else 3
    out["avg_recent"] = float(y[max(res.i0, n_hist - recent):n_hist].mean()) if n_hist > res.i0 else 0.0
    if fit is None:
        out.update(trend_label="n/a", yoy_growth=None, momentum=None, seasonality_strength=None,
                   seasonality_label="n/a", peak_months="", low_months="")
        return out
    clean_units = fit.clean_rate * grid.days[res.i0:n_hist]
    n = len(clean_units)
    yoy = None
    if n >= 2 * lag:
        yoy = float(clean_units[-lag:].sum() / max(clean_units[-2 * lag:-lag].sum(), 1e-9) - 1)
    momentum = float(np.exp(fit.slope_end) - 1)
    g = yoy if yoy is not None else momentum
    out["yoy_growth"] = yoy
    out["momentum"] = momentum
    out["trend_label"] = ("Growing" if g > 0.08 else "Declining" if g < -0.08 else "Stable")
    # Seasonality strength (Hyndman): 1 - Var(R) / Var(S + R) on log scale.
    ty, doy = grid.ty[res.i0:n_hist], grid.doy[res.i0:n_hist]
    S = fit.seasonal_log(doy)
    R = np.log(fit.clean_rate + fit.c) - np.log(fit.baseline_rate(ty, doy) + fit.c)
    strength = float(max(0.0, 1 - np.var(R) / max(np.var(S + R), 1e-9)))
    out["seasonality_strength"] = strength
    out["seasonality_label"] = "Strong" if strength >= 0.6 else "Moderate" if strength >= 0.3 else "Weak"
    if fit.pool_used_for_season:
        out["seasonality_label"] += " (borrowed)"
    idx = seasonal_index(fit)
    order = np.argsort(-idx)
    out["seasonal_index"] = {m: round(float(v), 3) for m, v in zip(MONTHS, idx)}
    out["peak_months"] = ", ".join(MONTHS[i] for i in order[:3] if idx[i] > 1.05)
    out["low_months"] = ", ".join(MONTHS[i] for i in order[::-1][:3] if idx[i] < 0.95)
    return out


def event_lifts(res: SeriesResult, grid: Grid, pool_gamma: np.ndarray | None) -> list[dict]:
    fit = res.fit
    if fit is None or grid.X.shape[1] == 0:
        return []
    n_hist, lag = grid.n_hist, grid.lag
    base_units_hist = fit.fitted_rate * grid.days[res.i0:n_hist]
    Xh = grid.X[res.i0:n_hist]
    last_year = slice(max(0, len(base_units_hist) - lag), len(base_units_hist))
    fut = np.arange(n_hist, len(grid.starts))
    rows = []
    for ev in grid.event_keys:
        cols = [i for i, nme in enumerate(grid.x_names) if nme.startswith(ev + "|")]
        phases = {grid.x_names[i].split("|")[1]: i for i in cols}
        inc_hist = float(sum((base_units_hist[last_year] * Xh[last_year, i] * fit.gamma[i]).sum() for i in cols))
        inc_fut = float(sum((res.baseline * grid.X[fut, i] * fit.gamma[i]).sum() for i in cols))
        row = {"event": ev,
               "incremental_last_12m": inc_hist, "incremental_horizon": inc_fut}
        for ph in ("pre", "main", "post"):
            row[f"{ph}_lift_pct"] = float(fit.gamma[phases[ph]] * 100) if ph in phases else None
            row[f"{ph}_pool_pct"] = (float(pool_gamma[phases[ph]] * 100)
                                     if (ph in phases and pool_gamma is not None) else None)
        rows.append(row)
    return rows


def shift_bridge(res: SeriesResult, y: np.ndarray, grid: Grid) -> list[dict]:
    """Per future bucket: LY actual, LY re-aligned to this year's festival dates, and forecast."""
    fit = res.fit
    rows = []
    gamma = fit.gamma if fit is not None else np.zeros(grid.X.shape[1])
    for h in range(grid.horizon):
        t = grid.n_hist + h
        ly = t - grid.lag
        if ly < res.i0 or ly < 0:
            continue
        e_ly = float(_event_factor(grid, np.array([ly]), gamma)[0])
        e_ty = float(_event_factor(grid, np.array([t]), gamma)[0])
        a_ly = float(y[ly])
        aligned = a_ly / e_ly * e_ty
        rows.append({"bucket": grid.starts[t], "ly_bucket": grid.starts[ly], "ly_actual": a_ly,
                     "ly_aligned": aligned, "festival_shift_impact": aligned - a_ly,
                     "forecast": float(res.forecast[h]), "events_ty": grid.labels[t], "events_ly": grid.labels[ly]})
    return rows
