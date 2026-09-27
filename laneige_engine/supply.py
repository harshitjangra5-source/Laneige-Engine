"""Sell-in requirement (channel replenishment) and DC supply plan from WMS stock.

Flow per SKU:
    sell-out forecast --(channel cover target)--> sell-in plan = demand on the DC
    DC on-hand (WMS) + inbound --(MRP netting, safety stock, lead time, MOQ)--> orders & alerts

The channel target is *forward-looking* (next N weeks of sell-out), so partners are
automatically pre-loaded ahead of Diwali / sale events wherever those fall this year.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np


def forward_cover(fc: np.ndarray, cover_buckets: float) -> np.ndarray:
    """Target stock at the end of each bucket = sell-out of the next `cover_buckets` buckets."""
    H = len(fc)
    tail = fc[-min(4, H):].mean() if H else 0.0
    ext = np.concatenate([fc, np.full(int(math.ceil(cover_buckets)) + 1, tail)])
    full = int(math.floor(cover_buckets))
    frac = cover_buckets - full
    out = np.zeros(H)
    for t in range(H):
        out[t] = ext[t + 1: t + 1 + full].sum() + frac * ext[t + 1 + full]
    return out


def estimate_channel_stock(sell_in: np.ndarray | None, sell_out: np.ndarray, cover_buckets: float,
                           window: int) -> tuple[float, str]:
    """Current partner stock when it is not reported.

    Assumes partners held the target cover `window` buckets ago and adds net sell-in since.
    """
    so = sell_out[-window:]
    avg = float(so.mean()) if len(so) else 0.0
    if sell_in is None or len(sell_in) < window:
        return cover_buckets * avg, "assumed at target cover (no sell-in history)"
    net = float(sell_in[-window:].sum() - so.sum())
    return max(0.0, cover_buckets * avg + net), f"estimated from sell-in minus sell-out over the last {window} buckets"


def sellin_plan(fc: np.ndarray, cs0: float, cover_buckets: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = forward_cover(fc, cover_buckets)
    H = len(fc)
    sellin, stock = np.zeros(H), np.zeros(H)
    cs = cs0
    for t in range(H):
        need = fc[t] + target[t] - cs
        sellin[t] = max(0.0, need)
        cs = cs + sellin[t] - fc[t]
        stock[t] = cs
    return sellin, stock, target


def round_qty(q: float, moq: float | None, case_pack: float | None) -> float:
    if q <= 0:
        return 0.0
    if moq and q < moq:
        q = moq
    if case_pack and case_pack > 1:
        q = math.ceil(q / case_pack) * case_pack
    return float(math.ceil(q))


def dc_plan(demand: np.ndarray, on_hand: float, receipts: np.ndarray, lt_buckets: int, safety_stock: float,
            moq: float | None, case_pack: float | None, cycle_buckets: int = 4) -> dict[str, np.ndarray]:
    """MRP netting for one SKU at the DC.

    * Inside the lead time no normal PO can land: any deficit becomes an expedite quantity
      (enough to avoid a stockout, rounded to case pack).
    * From the lead time on, normal POs keep stock above safety stock and cover `cycle_buckets`.
    """
    H = len(demand)
    planned = np.zeros(H)
    expedite = np.zeros(H, dtype=bool)
    projected = np.zeros(H)
    no_action = np.zeros(H)
    prev = prev_na = on_hand
    for t in range(H):
        avail = prev + receipts[t]
        end = avail - demand[t]
        if t < lt_buckets:
            if end < 0:
                q = round_qty(-end, None, case_pack)
                planned[t], expedite[t] = q, True
                end += q
        elif end < safety_stock:
            need = demand[t: min(H, t + cycle_buckets)].sum() + safety_stock - avail
            q = round_qty(need, moq, case_pack)
            planned[t] = q
            end += q
        prev = end
        projected[t] = end
        prev_na = prev_na + receipts[t] - demand[t]
        no_action[t] = prev_na
    return {"planned_receipt": planned, "projected": projected, "projected_no_new_orders": no_action,
            "expedite": expedite}


def safety_stock(sigma_rel: float, demand: np.ndarray, lt_buckets: int, service_level: float, review: int = 1) -> float:
    """z * sigma * sqrt(lead time + review), with sigma from the backtested forecast error."""
    z = NormalDist().inv_cdf(min(max(service_level, 0.5), 0.999))
    window = max(1, min(len(demand), lt_buckets + review))
    mean_d = float(demand[:window].mean()) if len(demand) else 0.0
    return float(z * min(sigma_rel, 1.0) * mean_d * math.sqrt(lt_buckets + review))
