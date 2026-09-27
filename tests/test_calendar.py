import numpy as np
import pandas as pd

from laneige_engine import calendar as cal


def test_diwali_shift_2025_to_2026():
    occ = cal.base_occurrences(range(2024, 2028))
    t = cal.shift_table(occ)
    d26 = t[(t["key"] == "diwali") & (t["year"] == 2026)].iloc[0]
    assert str(d26["start"]) == "2026-11-08"
    assert d26["shift_vs_ly_days"] == 19
    assert d26["month_change"] == "Oct -> Nov"


def test_event_features_split_across_weeks():
    occ = cal.base_occurrences(range(2026, 2027))
    starts = pd.date_range("2026-10-05", periods=6, freq="7D")  # Mondays
    X, names = cal.event_features(starts, np.full(6, 7), occ)
    pre = X[:, names.index("diwali|pre")]
    # Diwali 8-Nov-2026; 21-day pre-window = 18-Oct .. 7-Nov
    np.testing.assert_allclose(pre, [0, 1 / 7, 1, 1, 6 / 7, 0], atol=1e-9)


def test_overrides_keep_separate_runs_and_merge_overlaps():
    occ = cal.base_occurrences(range(2026, 2027))
    ov = pd.DataFrame({"event": ["Brand Day", "Brand Day", "Amazon Great Indian Festival", "Flipkart BBD"],
                       "start": ["2026-04-16", "2026-10-01", "2026-10-05", "2026-10-06"],
                       "end": ["2026-04-18", "2026-10-03", "2026-10-20", "2026-10-12"]})
    out, log = cal.apply_overrides(occ, ov)
    brand = out[out["event"] == "brand_day"]
    assert len(brand) == 2
    fest = out[out["event"] == "festive_mega_sale"]
    assert len(fest) == 1
    assert fest.iloc[0]["start"] == pd.Timestamp("2026-10-05")
    assert fest.iloc[0]["end"] == pd.Timestamp("2026-10-20")
    assert fest.iloc[0]["source"] == "your calendar"
    assert any("moves it by -3 days" in line for line in log)


def test_parse_date_iso_and_day_first():
    assert cal.parse_date("2026-10-04") == pd.Timestamp("2026-10-04")
    assert cal.parse_date("04-10-2026") == pd.Timestamp("2026-10-04")
    assert pd.isna(cal.parse_date(None))


def test_calibration_snaps_estimated_sale_to_spike():
    occ = cal.base_occurrences(range(2024, 2026))
    rule = occ[(occ["event"] == "pink_friday") & (occ["year"] == 2024)].iloc[0]
    days = pd.date_range("2024-09-01", "2025-01-31", freq="D")
    y = pd.Series(100.0, index=days)
    true_start = rule["start"] - pd.Timedelta(days=3)
    y[(days >= true_start) & (days <= true_start + pd.Timedelta(days=9))] = 300.0
    out, log = cal.calibrate_sale_dates(occ, y, 1)
    moved = out[(out["event"] == "pink_friday") & (out["year"] == 2024)].iloc[0]
    assert moved["start"] == true_start
    assert moved["source"] == "auto-aligned"
    assert log


def test_match_event_aliases():
    assert cal.match_event_key("Flipkart Big Billion Days") == "festive_mega_sale"
    assert cal.match_event_key("Deepavali") == "diwali"
    assert cal.match_event_key("Nykaa Pink Friday Sale") == "pink_friday"
    assert cal.match_event_key("Laneige Brand Day") is None
