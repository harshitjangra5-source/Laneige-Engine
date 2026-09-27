"""Synthetic, Laneige-style demo data (India) for trying the engine end to end.

Everything here is simulated. Real Laneige SKU names are used only so the demo reads
naturally; volumes, prices and patterns are invented.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import calendar as cal

START, END = pd.Timestamp("2023-01-01"), pd.Timestamp("2026-09-20")
SNAPSHOT = pd.Timestamp("2026-09-20")

# sku, name, category, daily units (all channels, Jan-2023), growth/yr, seasonal amp, peak day-of-year, MRP
SKUS = [
    ("LNG-LSM-BER-20", "Lip Sleeping Mask Berry 20g", "Lip Care", 60, 0.20, 0.35, 5, 1450),
    ("LNG-LSM-GUM-20", "Lip Sleeping Mask Gummy Bear 20g", "Lip Care", 25, 0.30, 0.35, 5, 1450),
    ("LNG-LGB-BER-10", "Lip Glowy Balm Berry 10g", "Lip Care", 30, 0.10, 0.25, 10, 1150),
    ("LNG-WSM-70", "Water Sleeping Mask 70ml", "Skincare", 20, 0.00, 0.15, 350, 2350),
    ("LNG-WBC-50", "Water Bank Blue Hyaluronic Cream 50ml", "Skincare", 18, 0.15, 0.20, 150, 2650),
    ("LNG-CST-170", "Cream Skin Toner & Moisturizer 170ml", "Skincare", 22, 0.05, 0.25, 15, 2450),
    ("LNG-WBS-50", "Water Bank Blue Hyaluronic Serum 50ml", "Skincare", 10, -0.15, 0.10, 120, 2950),
    ("LNG-RCS-50", "Radian-C Sun Cream SPF50 50ml", "Sun Care", 14, 0.25, 0.50, 135, 1950),
    ("LNG-NCM-15", "Neo Cushion Matte 15g", "Makeup", 12, 0.05, 0.10, 320, 3250),
    ("LNG-HLD-SET", "Lip Sleeping Mask Holiday Gift Set", "Gift Sets", 1.2, 0.10, 0.00, 1, 2200),
    ("LNG-BFM-60", "Bouncy & Firm Sleeping Mask 60ml (launch Mar-26)", "Skincare", 15, 0.20, 0.15, 350, 2750),
]
LAUNCH = {"LNG-BFM-60": pd.Timestamp("2026-03-02")}
CHANNELS = {"Nykaa": 0.38, "Amazon": 0.22, "Flipkart": 0.10, "Myntra": 0.06, "Tira": 0.08, "Offline MT": 0.16}

# True daily multipliers used in the simulation: event -> phase -> channel -> lift (default key "*")
LIFTS = {
    "festive_mega_sale": {"main": {"Amazon": 3.2, "Flipkart": 3.5, "Myntra": 2.0, "Nykaa": 1.4, "Tira": 1.4, "*": 1.0},
                          "pre": {"Amazon": 0.8, "Flipkart": 0.8, "*": 1.0}},
    "pink_friday": {"main": {"Nykaa": 3.0, "Tira": 1.8, "*": 1.3}},
    "prime_day": {"main": {"Amazon": 2.6, "*": 1.0}},
    "mid_year_sale": {"main": {"Myntra": 2.5, "Nykaa": 1.8, "*": 1.0}},
    "republic_day_sale": {"main": {"Amazon": 1.6, "Flipkart": 1.6, "*": 1.0}},
    "independence_day_sale": {"main": {"Amazon": 1.6, "Flipkart": 1.6, "*": 1.0}},
    "diwali": {"pre": {"*": 1.5}, "main": {"*": 1.3}, "post": {"*": 0.8}},
    "navratri": {"main": {"*": 1.1}},
    "christmas_new_year": {"main": {"*": 1.3}},
    "valentines": {"pre": {"*": 1.15}},
    "raksha_bandhan": {"pre": {"*": 1.2}},
    "holi": {"pre": {"*": 1.05}},
    "brand_day": {"main": {"Nykaa": 2.2, "*": 1.0}},
}
CATEGORY_EXTRA = {  # extra multipliers on top of LIFTS for some categories
    ("Gift Sets", "diwali", "pre"): 8.0, ("Gift Sets", "christmas_new_year", "main"): 5.0,
    ("Makeup", "diwali", "pre"): 1.25, ("Lip Care", "valentines", "pre"): 1.3,
    ("Lip Care", "christmas_new_year", "main"): 1.2,
}
# The "real" sale dates differ from the engine's rule estimate so the auto-alignment has work to do.
TRUE_SHIFTS = {("pink_friday", 2024): -3, ("pink_friday", 2025): -3, ("mid_year_sale", 2024): 4,
               ("mid_year_sale", 2025): 4, ("mid_year_sale", 2023): 4}
BRAND_DAYS = [("2024-04-18", "2024-04-20"), ("2025-04-17", "2025-04-19"), ("2026-04-16", "2026-04-18"),
              ("2026-10-01", "2026-10-03")]
STOCKOUT = ("LNG-WBC-50", pd.Timestamp("2025-06-09"), pd.Timestamp("2025-06-29"))


def _true_calendar() -> pd.DataFrame:
    occ = cal.base_occurrences(range(2022, 2028))
    occ = occ.copy()
    for (ev, yr), s in TRUE_SHIFTS.items():
        m = (occ["event"] == ev) & (occ["year"] == yr)
        occ.loc[m, "start"] += pd.Timedelta(days=s)
        occ.loc[m, "end"] += pd.Timedelta(days=s)
    extra = [dict(event="brand_day", name="Laneige Brand Day on Nykaa", category="promo", year=pd.Timestamp(a).year,
                  start=pd.Timestamp(a), end=pd.Timestamp(b), pre_days=0, post_days=0, source="", confidence="",
                  note="") for a, b in BRAND_DAYS]
    return pd.concat([occ, pd.DataFrame(extra)], ignore_index=True)


def generate(seed: int = 7) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    days = pd.date_range(START, END, freq="D")
    doy = days.dayofyear.to_numpy()
    ty = ((days - START).days.to_numpy() / 365.25)
    weekend = np.where(days.weekday >= 5, 1.12, 0.95)
    occ = _true_calendar()

    def phase_mask(row, phase):
        if phase == "pre":
            a, b = row.start - pd.Timedelta(days=int(row.pre_days) or 5), row.start - pd.Timedelta(days=1)
        elif phase == "post":
            a, b = row.end + pd.Timedelta(days=1), row.end + pd.Timedelta(days=int(row.post_days) or 5)
        else:
            a, b = row.start, row.end
        return (days >= a) & (days <= b)

    masks = {}
    for row in occ.itertuples():
        for phase in ("pre", "main", "post"):
            masks.setdefault((row.event, phase), np.zeros(len(days), dtype=bool))
            masks[(row.event, phase)] |= phase_mask(row, phase)

    so_rows = []
    for sku, name, catg, base, growth, amp, peak, mrp in SKUS:
        level = base * np.exp(growth * ty) * (1 + amp * np.cos(2 * np.pi * (doy - peak) / 365.25)) * weekend
        if sku in LAUNCH:
            ramp = np.clip((days - LAUNCH[sku]).days.to_numpy() / 60.0, 0, 1)
            level = level * ramp
        for ch, share in CHANNELS.items():
            lam = level * share
            mult = np.ones(len(days))
            for (ev, phase), m in masks.items():
                spec = LIFTS.get(ev, {}).get(phase)
                if not spec:
                    continue
                lift = spec.get(ch, spec.get("*", 1.0)) * CATEGORY_EXTRA.get((catg, ev, phase), 1.0)
                mult = np.where(m, mult * lift, mult)
            lam = lam * mult
            if sku == STOCKOUT[0]:
                lam = np.where((days >= STOCKOUT[1]) & (days <= STOCKOUT[2]), 0.0, lam)
            qty = rng.poisson(lam * rng.gamma(20, 1 / 20, len(days)))
            disc = np.where(mult > 1.3, 0.72, 0.88)
            keep = qty > 0
            so_rows.append(pd.DataFrame({"Date": days[keep], "SKU Code": sku, "Product Name": name,
                                         "Category": catg, "Channel": ch, "Units Sold": qty[keep],
                                         "Net Sales (INR)": np.round(qty[keep] * mrp * disc[keep], 0)}))
    sell_out = pd.concat(so_rows, ignore_index=True).sort_values(["Date", "SKU Code", "Channel"])

    # Sell-in: partners order fortnightly to cover the next ~5 weeks, pre-loading ahead of events.
    wk = sell_out.assign(Week=sell_out["Date"] - pd.to_timedelta(sell_out["Date"].dt.weekday, unit="D"))
    weekly = wk.groupby(["SKU Code", "Channel", "Week"])["Units Sold"].sum().unstack(fill_value=0)
    si_rows = []
    for (sku, ch), row in weekly.iterrows():
        v = row.to_numpy().astype(float)
        weeks = row.index
        for i in range(0, len(v), 2):
            fwd = v[i + 2: i + 4].sum() if i + 4 <= len(v) else v[max(0, i - 2):i].sum()
            q = max(0, rng.normal(fwd, 0.1 * fwd + 1))
            if q > 0:
                si_rows.append({"Invoice Date": (weeks[i] + pd.Timedelta(days=int(rng.integers(0, 5)))).strftime("%d-%m-%Y"),
                                "Material": sku, "Customer": ch, "Billed Qty": int(round(q))})
    sell_in = pd.DataFrame(si_rows)

    # WMS snapshot: two DCs plus open import POs (weeks from snapshot, weeks of demand).
    recent = weekly.loc[:, weekly.columns >= SNAPSHOT - pd.Timedelta(days=56)].sum(axis=1).groupby(level=0).sum() / 8
    position = {"LNG-LSM-BER-20": (4.0, [(6, 5)]), "LNG-LSM-GUM-20": (8.0, [(5, 5), (10, 5)]),
                "LNG-LGB-BER-10": (9.0, [(6, 6), (11, 6)]), "LNG-WSM-70": (12.0, [(8, 6)]),
                "LNG-WBC-50": (8.0, [(5, 6), (11, 6)]), "LNG-CST-170": (8.0, [(5, 6)]),
                "LNG-WBS-50": (34.0, []), "LNG-RCS-50": (16.0, []), "LNG-NCM-15": (11.0, [(7, 6)]),
                "LNG-HLD-SET": (4.0, []), "LNG-BFM-60": (8.0, [(4, 8)])}
    wms_rows = []
    for sku, *_ in SKUS:
        wk_demand = recent.get(sku, 0)
        cover, pos = position[sku]
        oh = wk_demand * cover
        for wh, share in (("Bhiwandi DC", 0.7), ("Gurgaon DC", 0.3)):
            wms_rows.append({"Snapshot Date": SNAPSHOT.date(), "Item Code": sku, "Warehouse": wh,
                             "Unrestricted Stock": int(oh * share), "Goods In Transit": 0, "ETA": None,
                             "Stock Status": "Good"})
        for weeks_out, weeks_qty in pos:
            qty = int(np.ceil(wk_demand * weeks_qty / 48) * 48)
            wms_rows.append({"Snapshot Date": SNAPSHOT.date(), "Item Code": sku, "Warehouse": "Bhiwandi DC",
                             "Unrestricted Stock": 0, "Goods In Transit": qty,
                             "ETA": (SNAPSHOT + pd.Timedelta(days=7 * weeks_out)).date(), "Stock Status": "Good"})
    wms_rows.append({"Snapshot Date": SNAPSHOT.date(), "Item Code": "LNG-WSM-70", "Warehouse": "Gurgaon DC",
                     "Unrestricted Stock": 180, "Goods In Transit": 0, "ETA": None, "Stock Status": "Blocked - QC"})
    wms = pd.DataFrame(wms_rows)

    master = pd.DataFrame([{"SKU": s, "Description": n, "Category": c, "Lead Time (days)": 84 if c != "Gift Sets" else 120,
                            "MOQ": 1200 if c != "Gift Sets" else 500, "Case Pack": 48, "MRP": p}
                           for s, n, c, *_, p in SKUS])
    events = pd.DataFrame([{"Event": "Laneige Brand Day on Nykaa", "Start Date": a, "End Date": b}
                           for a, b in BRAND_DAYS])
    return {"sell_out": sell_out, "sell_in": sell_in, "wms": wms, "sku_master": master, "events": events}


def write(folder: str | Path = "sample_data", seed: int = 7) -> dict[str, Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, df in generate(seed).items():
        p = folder / f"{name}.csv"
        df.to_csv(p, index=False)
        paths[name] = p
    return paths


if __name__ == "__main__":
    for k, p in write().items():
        print(f"{k:12s} -> {p}")


def demo_inputs(seed: int = 7):
    """Demo data passed through the same auto column-mapping as real uploads."""
    from .io import standardize
    from .pipeline import PlanningInputs

    raw = generate(seed)
    frames, reports = {}, {}
    for kind in ("sell_out", "sell_in", "wms", "sku_master", "events"):
        frames[kind], reports[kind] = standardize(raw[kind], kind)
    return PlanningInputs(sell_out=frames["sell_out"], sell_in=frames["sell_in"], wms=frames["wms"],
                          sku_master=frames["sku_master"], events=frames["events"], reports=reports)
