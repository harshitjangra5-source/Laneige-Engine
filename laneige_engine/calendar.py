"""India festival & sale-event calendar with automatic year-on-year date shifting.

How festival shifts are handled
-------------------------------
Every event is stored as dated *occurrences* (one per year). Demand models never see
"week 43" or "October"; they see, for each planning bucket, what fraction of the bucket
falls inside each event window (pre-window / main days / post-window). Because those
fractions are recomputed from each year's real dates, an uplift learned from Diwali on
20-Oct-2025 is automatically placed on 8-Nov-2026 when forecasting.

Sources of dates
----------------
* Lunar festivals (Diwali, Holi, Raksha Bandhan, Navratri/Dussehra, Eid) come from
  lookup tables. Please verify future years against the official holiday list.
* Fixed-date events (Valentine's, Christmas ...) and nth-weekday events (Mother's Day).
* E-commerce sales: reported dates where known, otherwise a rule anchored to the
  festival calendar (e.g. festive mega sale ~3 days before Navratri). The rule offset
  is re-learned from history, and historic sale windows can be auto-aligned to the
  spikes visible in the sell-out data.
* Your own calendar (upload) always wins.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

FIRST_YEAR, LAST_YEAR = 2017, 2030


@dataclass(frozen=True)
class EventDef:
    key: str
    name: str
    category: str          # festival | sale | gifting | holiday | promo
    pre_days: int = 0
    post_days: int = 0
    duration_days: int = 1  # length of rule-based occurrences
    description: str = ""


EVENTS: dict[str, EventDef] = {e.key: e for e in [
    EventDef("diwali", "Diwali", "festival", pre_days=21, post_days=7,
             description="Lakshmi Puja day. The 21-day pre-window covers Karwa Chauth, Dhanteras and "
                         "gifting; the post-window captures the usual post-festival dip."),
    EventDef("navratri", "Navratri & Dussehra", "festival", pre_days=3, post_days=0,
             description="Sharad Navratri start through Dussehra."),
    EventDef("holi", "Holi", "festival", pre_days=7, post_days=3),
    EventDef("raksha_bandhan", "Raksha Bandhan", "festival", pre_days=10, post_days=0,
             description="Gifting window before Rakhi."),
    EventDef("eid_ul_fitr", "Eid ul-Fitr", "festival", pre_days=10, post_days=2),
    EventDef("christmas_new_year", "Christmas & New Year", "holiday", pre_days=10, post_days=7),
    EventDef("valentines", "Valentine's Day", "gifting", pre_days=7),
    EventDef("womens_day", "Women's Day", "gifting", pre_days=3),
    EventDef("mothers_day", "Mother's Day", "gifting", pre_days=7),
    EventDef("festive_mega_sale", "Festive mega sale (Amazon GIF / Flipkart BBD)", "sale",
             pre_days=5, post_days=5, duration_days=8,
             description="Marketplace festive sales; the pre-window captures 'wait for the sale' dips."),
    EventDef("pink_friday", "Nykaa Pink Friday / Black Friday sale", "sale",
             pre_days=5, post_days=3, duration_days=10),
    EventDef("prime_day", "Amazon Prime Day", "sale", post_days=2, duration_days=2),
    EventDef("republic_day_sale", "Republic Day sale", "sale", post_days=2, duration_days=6),
    EventDef("independence_day_sale", "Independence Day sale", "sale", post_days=2, duration_days=6),
    EventDef("mid_year_sale", "Mid-year sale (Myntra EORS / Nykaa Hot Pink)", "sale",
             post_days=2, duration_days=6),
]}

SHORT_NAMES = {
    "festive_mega_sale": "Festive sale", "navratri": "Navratri", "pink_friday": "Pink Friday",
    "christmas_new_year": "Christmas/NY", "republic_day_sale": "R-Day sale", "independence_day_sale": "I-Day sale",
    "mid_year_sale": "Mid-year sale", "raksha_bandhan": "Rakhi", "eid_ul_fitr": "Eid", "prime_day": "Prime Day",
}


def short_name(key: str, name: str) -> str:
    return SHORT_NAMES.get(key, name.split(" (")[0])


ALIASES = {
    "diwali": ["diwali", "deepavali", "deepawali", "lakshmi puja"],
    "navratri": ["navratri", "navaratri", "dussehra", "dasara", "durga puja", "vijayadashami"],
    "holi": ["holi"],
    "raksha_bandhan": ["raksha bandhan", "rakhi", "rakshabandhan"],
    "eid_ul_fitr": ["eid", "eid ul fitr", "eid al fitr", "ramzan eid"],
    "christmas_new_year": ["christmas", "xmas", "new year", "christmas new year"],
    "valentines": ["valentine", "valentines", "valentine s day"],
    "womens_day": ["women s day", "womens day", "women day"],
    "mothers_day": ["mother s day", "mothers day"],
    "festive_mega_sale": ["big billion days", "bbd", "great indian festival", "gif", "festive sale",
                          "festive mega sale", "big fashion festival"],
    "pink_friday": ["pink friday", "black friday", "cyber monday"],
    "prime_day": ["prime day"],
    "republic_day_sale": ["republic day"],
    "independence_day_sale": ["independence day", "freedom festival", "freedom sale"],
    "mid_year_sale": ["eors", "end of reason", "hot pink", "mid year sale", "end of season"],
}

# --- Lunar festival tables (India, commonly observed dates) ---------------------------------
_LUNAR = {
    "diwali": {2017: "10-19", 2018: "11-07", 2019: "10-27", 2020: "11-14", 2021: "11-04", 2022: "10-24",
               2023: "11-12", 2024: "10-31", 2025: "10-20", 2026: "11-08", 2027: "10-29", 2028: "10-17",
               2029: "11-05", 2030: "10-26"},
    "holi": {2017: "03-13", 2018: "03-02", 2019: "03-21", 2020: "03-10", 2021: "03-29", 2022: "03-18",
             2023: "03-08", 2024: "03-25", 2025: "03-14", 2026: "03-04", 2027: "03-22", 2028: "03-11",
             2029: "03-01", 2030: "03-20"},
    "raksha_bandhan": {2017: "08-07", 2018: "08-26", 2019: "08-15", 2020: "08-03", 2021: "08-22",
                       2022: "08-11", 2023: "08-30", 2024: "08-19", 2025: "08-09", 2026: "08-28",
                       2027: "08-17", 2028: "08-05", 2029: "08-23", 2030: "08-13"},
    "eid_ul_fitr": {2017: "06-26", 2018: "06-16", 2019: "06-05", 2020: "05-25", 2021: "05-14",
                    2022: "05-03", 2023: "04-22", 2024: "04-11", 2025: "03-31", 2026: "03-21",
                    2027: "03-10", 2028: "02-27", 2029: "02-15", 2030: "02-05"},
}
_NAVRATRI_START = {2017: "09-21", 2018: "10-10", 2019: "09-29", 2020: "10-17", 2021: "10-07", 2022: "09-26",
                   2023: "10-15", 2024: "10-03", 2025: "09-22", 2026: "10-11", 2027: "09-30", 2028: "09-19",
                   2029: "10-07", 2030: "09-27"}
_DUSSEHRA = {2017: "09-30", 2018: "10-19", 2019: "10-08", 2020: "10-25", 2021: "10-15", 2022: "10-05",
             2023: "10-24", 2024: "10-12", 2025: "10-02", 2026: "10-20", 2027: "10-09", 2028: "09-27",
             2029: "10-16", 2030: "10-06"}

# --- Publicly reported marketplace sale windows (approximate; verify / override) -------------
_REPORTED = {
    "festive_mega_sale": {2019: ("09-29", "10-04"), 2020: ("10-16", "10-21"), 2021: ("10-03", "10-10"),
                          2022: ("09-23", "09-30"), 2023: ("10-08", "10-15"), 2024: ("09-27", "10-06"),
                          2025: ("09-23", "10-02")},
    "prime_day": {2019: ("07-15", "07-16"), 2020: ("08-06", "08-07"), 2021: ("07-26", "07-27"),
                  2022: ("07-23", "07-24"), 2023: ("07-15", "07-16"), 2024: ("07-20", "07-21"),
                  2025: ("07-12", "07-14")},
}

HISTORY_SOURCES = {"lunar table", "fixed date", "reported", "your calendar", "auto-aligned"}
ESTIMATED_SOURCES = {"rule estimate", "reported"}


_ISO_RE = re.compile(r"^\s*\d{4}-\d{1,2}-\d{1,2}")


def parse_date(value) -> pd.Timestamp:
    """Parse one date: ISO (YYYY-MM-DD) as-is, otherwise Indian day-first (dd-mm-yyyy). NaT if unreadable."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NaT
    if isinstance(value, (pd.Timestamp, np.datetime64)) or hasattr(value, "year"):
        return pd.Timestamp(value).normalize()
    text = str(value).strip()
    if _ISO_RE.match(text):
        return pd.to_datetime(text[:10], format="%Y-%m-%d", errors="coerce")
    return pd.to_datetime(text, dayfirst=True, errors="coerce")


def _d(year: int, mmdd: str) -> pd.Timestamp:
    return pd.Timestamp(f"{year}-{mmdd}")


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=month, day=1)
    offset = (weekday - first.weekday()) % 7
    return first + pd.Timedelta(days=offset + 7 * (n - 1))


def rule_window(key: str, year: int) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Rule-based (estimated) window for sale events; also used to learn rule offsets."""
    ev = EVENTS[key]
    dur = pd.Timedelta(days=ev.duration_days - 1)
    if key == "festive_mega_sale":
        if year not in _NAVRATRI_START:
            return None
        start = _d(year, _NAVRATRI_START[year]) - pd.Timedelta(days=3)
    elif key == "pink_friday":
        black_friday = _nth_weekday(year, 11, 3, 4) + pd.Timedelta(days=1)
        start = black_friday - pd.Timedelta(days=5)
    elif key == "prime_day":
        start = _nth_weekday(year, 7, 5, 3)          # third Saturday of July
    elif key == "republic_day_sale":
        start = pd.Timestamp(year=year, month=1, day=13)
    elif key == "independence_day_sale":
        start = pd.Timestamp(year=year, month=8, day=6)
    elif key == "mid_year_sale":
        start = pd.Timestamp(year=year, month=6, day=1)
    else:
        return None
    return start, start + dur


def _fixed_window(key: str, year: int) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    if key == "valentines":
        d = pd.Timestamp(year=year, month=2, day=14)
    elif key == "womens_day":
        d = pd.Timestamp(year=year, month=3, day=8)
    elif key == "mothers_day":
        d = _nth_weekday(year, 5, 6, 2)
    elif key == "christmas_new_year":
        return pd.Timestamp(year=year, month=12, day=24), pd.Timestamp(year=year, month=12, day=31)
    else:
        return None
    return d, d


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def match_event_key(name: str) -> str | None:
    norm = " " + re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip() + " "
    if _slug(name) in EVENTS:
        return _slug(name)
    for key, words in ALIASES.items():
        for w in words:
            if f" {w} " in norm:
                return key
    return None


def base_occurrences(years: range, disabled: set[str] | None = None) -> pd.DataFrame:
    """Calendar before user overrides and data calibration."""
    disabled = disabled or set()
    rows = []
    for key, ev in EVENTS.items():
        if key in disabled:
            continue
        for y in years:
            src, conf, win = None, None, None
            if key in _LUNAR and y in _LUNAR[key]:
                d = _d(y, _LUNAR[key][y])
                win, src = (d, d), "lunar table"
            elif key == "navratri" and y in _NAVRATRI_START:
                win, src = (_d(y, _NAVRATRI_START[y]), _d(y, _DUSSEHRA[y])), "lunar table"
            elif key in _REPORTED and y in _REPORTED[key]:
                s, e = _REPORTED[key][y]
                win, src = (_d(y, s), _d(y, e)), "reported"
            elif (fw := _fixed_window(key, y)) is not None:
                win, src = fw, "fixed date"
            elif (rw := rule_window(key, y)) is not None:
                win, src = rw, "rule estimate"
            if win is None:
                continue
            conf = {"lunar table": "high" if y <= 2026 else "verify",
                    "fixed date": "high", "reported": "approx", "rule estimate": "estimated"}[src]
            rows.append(dict(event=key, name=ev.name, category=ev.category, year=y,
                             start=win[0], end=win[1], pre_days=ev.pre_days, post_days=ev.post_days,
                             source=src, confidence=conf, note=""))
    return pd.DataFrame(rows)


def apply_overrides(occ: pd.DataFrame, overrides: pd.DataFrame | None) -> tuple[pd.DataFrame, list[str]]:
    """Merge a user calendar (columns: event, start, [end, pre_days, post_days, category]).

    Rows naming the same event are merged only when their windows overlap (e.g. "Amazon GIF" and
    "Flipkart BBD" on the same days); separate runs of a promo in one year stay separate. Any user
    row for an event in a year replaces the engine's own date for that event and year.
    """
    log: list[str] = []
    if overrides is None or overrides.empty:
        return occ, log
    occ = occ.copy()
    ov = overrides.copy()
    ov["start"] = ov["start"].map(parse_date)
    ov["end"] = ov["end"].map(parse_date) if "end" in ov else ov["start"]
    ov["end"] = ov["end"].fillna(ov["start"])
    ov = ov.dropna(subset=["start"])
    ov["key"] = [match_event_key(e) or _slug(e) for e in ov["event"]]
    clusters: list[dict] = []
    for key, g in ov.sort_values("start").groupby("key", sort=False):
        current = None
        for _, r in g.iterrows():
            if current is not None and r["start"] <= current["end"] + pd.Timedelta(days=1):
                current["end"] = max(current["end"], r["end"])
                continue
            current = dict(key=key, name=str(r["event"]), start=r["start"], end=r["end"],
                           pre=r.get("pre_days"), post=r.get("post_days"), cat=r.get("category"))
            clusters.append(current)
    new_rows = []
    for c in clusters:
        key, year = c["key"], int(c["start"].year)
        ev = EVENTS.get(key)
        pre = int(c["pre"]) if pd.notna(c["pre"]) else (ev.pre_days if ev else 0)
        post = int(c["post"]) if pd.notna(c["post"]) else (ev.post_days if ev else 0)
        cat = c["cat"] if isinstance(c["cat"], str) and c["cat"] else (ev.category if ev else "promo")
        mask = (occ["event"] == key) & (occ["year"] == year)
        if mask.any():
            old = occ.loc[mask].iloc[0]
            moved = (c["start"] - old["start"]).days
            if moved:
                log.append(f"{old['name']} {year}: your calendar moves it by {moved:+d} days "
                           f"({old['start']:%d-%b} -> {c['start']:%d-%b}).")
            occ = occ.loc[~mask]
        new_rows.append(dict(event=key, name=ev.name if ev else c["name"], category=cat, year=year,
                             start=c["start"], end=c["end"], pre_days=pre, post_days=post,
                             source="your calendar", confidence="user", note=""))
    occ = pd.concat([occ, pd.DataFrame(new_rows)], ignore_index=True)
    return occ.sort_values(["start", "event"]).reset_index(drop=True), log


def _window_overlap(a: pd.Timestamp, b: pd.Timestamp, starts: np.ndarray, days: np.ndarray) -> np.ndarray:
    """Days of [a, b] (inclusive) that fall into each bucket [start, start+days-1]."""
    s = starts.astype("datetime64[D]").astype(np.int64)
    e = s + days - 1
    lo = np.maximum(s, day_number(a))
    hi = np.minimum(e, day_number(b))
    return np.clip(hi - lo + 1, 0, None).astype(float)


def day_number(ts: pd.Timestamp) -> np.int64:
    return np.datetime64(pd.Timestamp(ts).to_datetime64(), "D").astype(np.int64)


def phase_windows(row) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    out = []
    if row["pre_days"] > 0:
        out.append(("pre", row["start"] - pd.Timedelta(days=int(row["pre_days"])), row["start"] - pd.Timedelta(days=1)))
    out.append(("main", row["start"], row["end"]))
    if row["post_days"] > 0:
        out.append(("post", row["end"] + pd.Timedelta(days=1), row["end"] + pd.Timedelta(days=int(row["post_days"]))))
    return out


def event_features(period_starts: pd.DatetimeIndex, period_days: np.ndarray,
                   occ: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Matrix [n_periods x n_event_phases]: share of each bucket covered by each event phase."""
    starts = np.asarray(period_starts.values)
    cols: dict[str, np.ndarray] = {}
    for _, row in occ.iterrows():
        for phase, a, b in phase_windows(row):
            ov = _window_overlap(a, b, starts, period_days) / period_days
            name = f"{row['event']}|{phase}"
            cols[name] = cols.get(name, 0.0) + ov
    names = sorted(cols)
    if not names:
        return np.zeros((len(starts), 0)), []
    X = np.column_stack([cols[n] for n in names])
    return np.clip(X, 0.0, 1.0), names


def events_by_period(period_starts: pd.DatetimeIndex, period_days: np.ndarray, occ: pd.DataFrame) -> list[str]:
    """Human-readable list of event main-days falling in each bucket (for charts & tables)."""
    starts = np.asarray(period_starts.values)
    labels = [[] for _ in range(len(starts))]
    for _, row in occ.iterrows():
        ov = _window_overlap(row["start"], row["end"], starts, period_days)
        for i in np.nonzero(ov)[0]:
            short = row["name"].split(" (")[0]
            if short not in labels[i]:
                labels[i].append(short)
    return [", ".join(l) for l in labels]


def calibrate_sale_dates(occ: pd.DataFrame, native: pd.Series, native_days: int,
                         max_shift: int = 7) -> tuple[pd.DataFrame, list[str]]:
    """Snap estimated historic sale windows to the spikes visible in total sell-out.

    native: total quantity indexed by bucket start at the data's own granularity (daily/weekly).
    Only occurrences whose source is estimated/reported are moved; festivals are never moved.
    """
    log: list[str] = []
    if native_days > 7 or len(native) < 20:
        return occ, log
    occ = occ.copy()
    y = native.astype(float)
    window = 29 if native_days == 1 else 9
    base = y.rolling(window, center=True, min_periods=max(3, window // 3)).median()
    ratio = (y / base.replace(0, np.nan) - 1).fillna(0).clip(-1, 5)
    starts = np.asarray(ratio.index.values)
    days = np.full(len(starts), native_days)
    mad = float(np.median(np.abs(ratio - ratio.median()))) * 1.4826 + 1e-6
    first, last = ratio.index.min(), ratio.index.max() + pd.Timedelta(days=native_days - 1)

    cand = occ[(occ["category"] == "sale") & occ["source"].isin(ESTIMATED_SOURCES)
               & (occ["start"] - pd.Timedelta(days=max_shift + 10) >= first)
               & (occ["end"] + pd.Timedelta(days=max_shift + 10) <= last)]
    for idx, row in cand.iterrows():
        length = (row["end"] - row["start"]).days + 1
        scores = {}
        for s in range(-max_shift, max_shift + 1):
            a = row["start"] + pd.Timedelta(days=s)
            b = row["end"] + pd.Timedelta(days=s)
            w = _window_overlap(a, b, starts, days) / length
            scores[s] = float(np.dot(w, ratio.values))
        best = max(scores, key=lambda s: (round(scores[s], 3), -abs(s)))
        gain = scores[best] - scores[0]
        noise = 3 * mad / np.sqrt(max(1.0, length / native_days))
        if best != 0 and gain > max(0.08, 0.25 * abs(scores[best])) and scores[best] > noise:
            occ.loc[idx, "start"] = row["start"] + pd.Timedelta(days=best)
            occ.loc[idx, "end"] = row["end"] + pd.Timedelta(days=best)
            occ.loc[idx, "source"] = "auto-aligned"
            occ.loc[idx, "note"] = f"moved {best:+d} days to match the sell-out spike"
            log.append(f"{row['name']} {row['year']}: auto-aligned {best:+d} days "
                       f"({row['start']:%d-%b} -> {occ.loc[idx, 'start']:%d-%b}) to match the spike in sell-out.")
    return occ, log


def learn_rule_offsets(occ: pd.DataFrame, as_of: pd.Timestamp) -> tuple[pd.DataFrame, list[str]]:
    """Shift future rule-estimated sale dates by the median error the rule had in past years."""
    log: list[str] = []
    occ = occ.copy()
    for key in occ.loc[occ["category"] == "sale", "event"].unique():
        if key not in EVENTS:
            continue
        hist = occ[(occ["event"] == key) & (occ["end"] < as_of)
                   & occ["source"].isin({"reported", "your calendar", "auto-aligned"})]
        errs = []
        for _, r in hist.iterrows():
            rw = rule_window(key, int(r["year"]))
            if rw is not None:
                errs.append((r["start"] - rw[0]).days)
        if len(errs) < 2:
            continue
        recent = np.array(errs[-4:])
        off = int(round(float(np.median(recent))))
        consistent = max((recent > 0).mean(), (recent < 0).mean()) >= 0.75
        if not consistent:
            continue
        fut = (occ["event"] == key) & (occ["source"] == "rule estimate") & (occ["start"] >= as_of)
        if abs(off) >= 2 and fut.any():
            occ.loc[fut, "start"] += pd.Timedelta(days=off)
            occ.loc[fut, "end"] += pd.Timedelta(days=off)
            occ.loc[fut, "note"] = f"rule shifted {off:+d} days (learned from past years)"
            log.append(f"{EVENTS[key].name}: future dates shifted {off:+d} days based on how the "
                       f"sale actually ran in past years.")
    return occ, log


def build_calendar(years: range, overrides: pd.DataFrame | None = None, disabled: list[str] | None = None,
                   native: pd.Series | None = None, native_days: int = 7, auto_align: bool = True,
                   as_of: pd.Timestamp | None = None) -> tuple[pd.DataFrame, list[str]]:
    occ = base_occurrences(years, set(disabled or []))
    occ, log = apply_overrides(occ, overrides)
    if auto_align and native is not None:
        occ, log2 = calibrate_sale_dates(occ, native, native_days)
        log += log2
    if as_of is not None:
        occ, log3 = learn_rule_offsets(occ, as_of)
        log += log3
    return occ.sort_values(["start", "event"]).reset_index(drop=True), log


def shift_table(occ: pd.DataFrame) -> pd.DataFrame:
    """Year-on-year movement of each event (days and month changes) for the planner.

    Events that run several times a year are matched to last year's run with the same order.
    """
    rows = []
    for key, g in occ.sort_values("start").groupby("event", sort=False):
        g = g.assign(nth=g.groupby("year").cumcount())
        prev_by = {(int(r["year"]), int(r["nth"])): r for _, r in g.iterrows()}
        for _, r in g.iterrows():
            y = int(r["year"])
            prev = prev_by.get((y - 1, int(r["nth"])))
            shift = None
            if prev is not None:
                shift = int((r["start"] - (prev["start"] + pd.DateOffset(years=1))).days)
            rows.append(dict(event=r["name"], key=key, category=r["category"], year=y,
                             start=r["start"].date(), end=r["end"].date(),
                             weekday=r["start"].strftime("%a"),
                             shift_vs_ly_days=shift,
                             month_change=(f"{prev['start']:%b} -> {r['start']:%b}"
                                           if prev is not None and prev["start"].month != r["start"].month else ""),
                             source=r["source"], confidence=r["confidence"], note=r.get("note", "")))
    out = pd.DataFrame(rows).sort_values(["year", "start"]).reset_index(drop=True)
    out["shift_vs_ly_days"] = out["shift_vs_ly_days"].astype("Float64")
    return out
