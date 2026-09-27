"""Statistical building blocks: event-aware baseline decomposition and baseline forecasters.

Demand model per series (units in bucket t, with d_t days):

    y_t = d_t * baseline_rate_t * (1 + x_t . gamma)

* baseline_rate_t = exp(level + piecewise-linear trend + annual Fourier seasonality) - c
* x_t  = share of bucket t inside each festival / sale phase (from calendar.event_features)
* gamma = daily uplift of each phase (e.g. 0.8 -> +80% on the days inside the window)

Trend/seasonality and uplifts are estimated by backfitting. Uplifts are shrunk toward the
portfolio ("pool") estimate, and seasonality toward the pool's seasonality when history is
short, so new SKUs borrow patterns from similar products.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TWO_PI = 2 * np.pi


def ridge_to_prior(D: np.ndarray, t: np.ndarray, w: np.ndarray, lam: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """argmin sum w (t - D b)^2 + sum lam_j (b_j - prior_j)^2."""
    Dw = D * w[:, None]
    A = D.T @ Dw + np.diag(lam)
    rhs = Dw.T @ t + lam * prior
    try:
        return np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, rhs, rcond=None)[0]


@dataclass
class PoolPrior:
    gamma: np.ndarray
    fourier: np.ndarray


@dataclass
class Design:
    ppy: float                 # buckets per year
    n_harm: int
    knots: np.ndarray          # in years since series start
    t0: float                  # tyears of the first training bucket

    @property
    def n_trend(self) -> int:
        return 2 + len(self.knots)

    @property
    def fourier_slice(self) -> slice:
        return slice(self.n_trend, self.n_trend + 2 * self.n_harm)

    def trend_part(self, ty: np.ndarray) -> np.ndarray:
        t = ty - self.t0
        cols = [np.ones_like(t), t] + [np.maximum(0.0, t - k) for k in self.knots]
        return np.column_stack(cols)

    def fourier_part(self, doy: np.ndarray) -> np.ndarray:
        cols = []
        for h in range(1, self.n_harm + 1):
            ang = TWO_PI * h * doy / 365.25
            cols += [np.sin(ang), np.cos(ang)]
        return np.column_stack(cols) if cols else np.zeros((len(doy), 0))

    def matrix(self, ty: np.ndarray, doy: np.ndarray) -> np.ndarray:
        return np.hstack([self.trend_part(ty), self.fourier_part(doy)])


def fourier_basis(doy: np.ndarray, n_harm: int) -> np.ndarray:
    return Design(1, n_harm, np.array([]), 0.0).fourier_part(doy)


@dataclass
class SeriesFit:
    """Result of decomposing one series' history into baseline x events."""
    design: Design
    beta: np.ndarray
    gamma: np.ndarray
    c: float
    level_k: float
    slope_end: float
    t_end: float
    phi: float
    resid_scale: float
    fitted_rate: np.ndarray        # baseline rate on training buckets (after level calibration)
    clean_rate: np.ndarray         # de-evented, stockout-imputed, outlier-clipped rate
    event_factor: np.ndarray       # 1 + x.gamma on training buckets
    anomalies: np.ndarray          # bool: unexplained spikes/dips on training buckets
    pool_used_for_season: bool = False
    notes: list[str] = field(default_factory=list)

    def seasonal_log(self, doy: np.ndarray) -> np.ndarray:
        return self.design.fourier_part(doy) @ self.beta[self.design.fourier_slice]

    def trend_log(self, ty: np.ndarray) -> np.ndarray:
        """Trend on log scale; beyond the training end the last slope is damped."""
        d = self.design
        inside = d.trend_part(np.minimum(ty, self.t_end)) @ self.beta[: d.n_trend]
        extra = np.clip(ty - self.t_end, 0, None)
        steps = np.round(extra * d.ppy)
        damped_years = np.where(steps > 0, self.phi * (1 - self.phi ** steps) / (1 - self.phi) / d.ppy, 0.0)
        return inside + self.slope_end * damped_years

    def baseline_rate(self, ty: np.ndarray, doy: np.ndarray) -> np.ndarray:
        return np.maximum(np.exp(self.trend_log(ty) + self.seasonal_log(doy)) - self.c, 0.0) * self.level_k


def _harmonics(ppy: float) -> int:
    return 6 if ppy > 20 else 4


def fit_series(y: np.ndarray, days: np.ndarray, ty: np.ndarray, doy: np.ndarray, X: np.ndarray,
               censored: np.ndarray, prior: PoolPrior | None, is_pool: bool = False,
               n_iter: int = 5) -> SeriesFit:
    """Backfit baseline (trend + seasonality) and event uplifts on one series' history."""
    n = len(y)
    ppy = 365.25 / float(np.median(days))
    years = n / ppy
    knot_step = 0.5
    knots = np.arange(knot_step, max(years - 0.75, 0.0), knot_step)
    design = Design(ppy=ppy, n_harm=_harmonics(ppy), knots=knots, t0=float(ty[0]))
    D = design.matrix(ty, doy)
    P = D.shape[1]
    K = X.shape[1]
    fs = design.fourier_slice

    w_base = (~censored).astype(float)
    if w_base.sum() < 3:
        w_base = np.ones(n)
    rate = y / days
    pos = rate[(rate > 0) & (w_base > 0)]
    mean_rate = float(pos.mean()) if pos.size else 1.0
    c = max(0.5 / float(np.median(days)), 0.05 * mean_rate)

    # Penalties. Weights are normalised to mean 1 so penalties scale with n.
    lam = np.full(P, 1e-6)
    lam[2: design.n_trend] = 0.5 * n                                     # slope changes
    harm = np.repeat(np.arange(1, design.n_harm + 1), 2).astype(float)
    if is_pool or prior is None:
        f_prior = np.zeros(2 * design.n_harm)
        lam[fs] = 0.004 * n * harm
    else:
        f_prior = prior.fourier
        shortness = min(1e4, (2 * ppy / max(n, 1.0)) ** 4)
        lam[fs] = 0.02 * n * harm * max(shortness, 0.05)
    beta_prior = np.zeros(P)
    beta_prior[fs] = f_prior
    pool_season = (not is_pool) and prior is not None and n < 1.25 * ppy

    g_prior = np.zeros(K) if (is_pool or prior is None) else prior.gamma
    gamma = g_prior.copy()
    rw = np.ones(n)
    beta = np.zeros(P)
    fitted_rate = np.full(n, mean_rate)
    for _ in range(n_iter):
        E = np.clip(1.0 + X @ gamma, 0.2, None) if K else np.ones(n)
        z = np.log(np.maximum(rate / E, 0.0) + c)
        w = w_base * rw
        w = w / max(w.mean(), 1e-9)
        beta = ridge_to_prior(D, z, w, lam, beta_prior)
        res = z - D @ beta
        s = 1.4826 * float(np.median(np.abs(res[w_base > 0]))) + 1e-6
        rw = np.clip(2.5 * s / np.maximum(np.abs(res), 1e-9), 0.0, 1.0)
        fitted_rate = np.maximum(np.exp(D @ beta) - c, 1e-9)
        if K:
            B = fitted_rate * days
            r = y / B - 1.0
            we = w_base * np.clip(B / max(B[w_base > 0].mean() if (w_base > 0).any() else 1, 1e-9), 0.2, 5.0)
            gram = (X * we[:, None]).T @ X
            diag = np.diag(gram)
            base_l = float(np.median(diag[diag > 0])) if (diag > 0).any() else 1.0
            lam_g = np.full(K, (0.05 if (is_pool or prior is None) else 1.0) * base_l)
            gamma = ridge_to_prior(X, np.clip(r, -1.0, 20.0), we, lam_g, g_prior)
            gamma = np.clip(gamma, -0.8, 20.0)

    E = np.clip(1.0 + X @ gamma, 0.2, None) if K else np.ones(n)
    # Level calibration on the last year removes log-retransform bias.
    recent = np.zeros(n, dtype=bool)
    recent[max(0, n - int(round(ppy))):] = True
    m = recent & (w_base > 0)
    fitted_units = fitted_rate * days * E
    level_k = float(np.clip(y[m].sum() / max(fitted_units[m].sum(), 1e-9), 0.7, 1.4)) if m.any() and y[m].sum() > 0 else 1.0
    fitted_rate = fitted_rate * level_k

    slope_end = float(beta[1] + beta[2: design.n_trend].sum())
    slope_end = float(np.clip(slope_end, -0.7, 0.47))  # at most -50% / +60% a year
    phi = 0.985 if ppy > 20 else 0.93  # per-bucket trend damping (~same per-year damping)

    res = np.log(np.maximum(rate / E, 0.0) + c) - np.log(fitted_rate / level_k + c)
    s = 1.4826 * float(np.median(np.abs(res[w_base > 0]))) + 1e-6
    anomalies = (np.abs(res) > max(3.5 * s, 0.35)) & (w_base > 0)
    clean = rate / E
    lo, hi = fitted_rate * np.exp(-3 * s), fitted_rate * np.exp(3 * s)
    clean = np.where(anomalies, np.clip(clean, lo, hi), clean)
    clean = np.where(censored, fitted_rate, clean)

    return SeriesFit(design=design, beta=beta, gamma=gamma, c=c, level_k=level_k, slope_end=slope_end,
                     t_end=float(ty[-1]), phi=phi, resid_scale=s, fitted_rate=fitted_rate, clean_rate=clean,
                     event_factor=E, anomalies=anomalies, pool_used_for_season=pool_season)


def pool_prior_from(fit: SeriesFit) -> PoolPrior:
    return PoolPrior(gamma=fit.gamma.copy(), fourier=fit.beta[fit.design.fourier_slice].copy())


# ------------------------------------------------------------------------------------------------
# Baseline forecasters (operate on the de-evented "clean" rate)
# ------------------------------------------------------------------------------------------------
def holt_damped(a: np.ndarray, phi: float, h: int) -> np.ndarray:
    """Damped-trend Holt on a (log, seasonally adjusted) series with grid-searched smoothing."""
    n = len(a)
    if n < 4:
        return np.full(h, a[-1] if n else 0.0)
    best = None
    for alpha in (0.05, 0.1, 0.2, 0.35, 0.5):
        for beta in (0.0, 0.02, 0.06, 0.12):
            level, trend = a[0], float(np.mean(np.diff(a[: min(n, 8)])))
            sse = 0.0
            for x in a[1:]:
                pred = level + phi * trend
                err = x - pred
                sse += err * err
                new_level = pred + alpha * err
                trend = phi * trend + beta * (new_level - level - phi * trend) if beta else phi * trend
                level = new_level
            if best is None or sse < best[0]:
                best = (sse, level, trend)
    _, level, trend = best
    steps = np.arange(1, h + 1)
    return level + trend * phi * (1 - phi ** steps) / (1 - phi)


def theta(a: np.ndarray, h: int) -> np.ndarray:
    """Classic Theta method (SES with half the linear drift)."""
    n = len(a)
    if n < 4:
        return np.full(h, a[-1] if n else 0.0)
    t = np.arange(n)
    b0 = np.polyfit(t, a, 1)[0]
    best = None
    for alpha in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7):
        level, sse = a[0], 0.0
        for x in a[1:]:
            sse += (x - level) ** 2
            level = level + alpha * (x - level)
        if best is None or sse < best[0]:
            best = (sse, alpha, level)
    _, alpha, level = best
    steps = np.arange(1, h + 1)
    drift = 0.5 * b0 * (steps - 1 + 1 / alpha - (1 - alpha) ** n / alpha)
    return level + drift


def croston_sba(y: np.ndarray, h: int, alpha: float = 0.1) -> np.ndarray:
    """Syntetos-Boylan approximation for intermittent demand (units per bucket)."""
    nz = np.nonzero(y > 0)[0]
    if len(nz) == 0:
        return np.zeros(h)
    z, p, last = float(y[nz[0]]), float(nz[0] + 1), nz[0]
    for i in nz[1:]:
        z += alpha * (y[i] - z)
        p += alpha * ((i - last) - p)
        last = i
    return np.full(h, (1 - alpha / 2) * z / max(p, 1.0))


def demand_class(y: np.ndarray) -> tuple[str, float, float]:
    """Syntetos-Boylan-Croston classification -> (class, ADI, CV^2)."""
    nz = y[y > 0]
    if len(nz) == 0:
        return "no demand", float("inf"), 0.0
    adi = len(y) / len(nz)
    cv2 = float((nz.std() / nz.mean()) ** 2) if len(nz) > 1 and nz.mean() > 0 else 0.0
    if adi < 1.32:
        cls = "smooth" if cv2 < 0.49 else "erratic"
    else:
        cls = "intermittent" if cv2 < 0.49 else "lumpy"
    return cls, float(adi), cv2
