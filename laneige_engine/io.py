"""Load sell-out / sell-in / WMS / SKU-master / event files and map their columns automatically.

Handles CSV and Excel, title rows above the header, wide layouts (one column per week or
month), Excel serial dates, dd-mm-yyyy vs mm-dd-yyyy, month labels such as "Jan-24" and
ISO weeks such as "2024-W23".
"""
from __future__ import annotations

import difflib
import io as _io
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SCHEMAS: dict[str, dict[str, list[str]]] = {
    "sell_out": {"required": ["date", "sku", "qty"],
                 "optional": ["channel", "value", "channel_stock", "product_name", "category"]},
    "sell_in": {"required": ["date", "sku", "qty"],
                "optional": ["channel", "value", "product_name", "category"]},
    "wms": {"required": ["sku", "on_hand"],
            "optional": ["date", "location", "in_transit", "eta", "status", "lead_time_days", "moq",
                         "case_pack", "product_name"]},
    "sku_master": {"required": ["sku"],
                   "optional": ["product_name", "category", "lead_time_days", "moq", "case_pack", "price",
                                "status"]},
    "events": {"required": ["event", "start"],
               "optional": ["end", "pre_days", "post_days", "category"]},
}

SYNONYMS: dict[str, list[str]] = {
    "date": ["date", "day", "week", "week commencing", "week start", "week starting", "week beginning",
             "week ending", "wk", "month", "period", "invoice date", "order date", "transaction date",
             "posting date", "billing date", "bill date", "sale date", "doc date", "snapshot date", "as of",
             "as of date", "report date", "stock date", "fiscal week", "calendar date"],
    "sku": ["sku", "sku code", "sku id", "item", "item code", "item id", "material", "material code",
            "material no", "product code", "product id", "article", "article code", "ean", "barcode", "asin",
            "fsn", "style id", "part number", "code"],
    "qty": ["qty", "quantity", "units", "unit sales", "sales qty", "sales quantity", "sold qty", "net qty",
            "net units", "volume", "pcs", "pieces", "sell out qty", "sellout qty", "sell out units",
            "sell in qty", "sellin qty", "billed qty", "invoice qty", "invoiced qty", "shipped qty",
            "dispatch qty", "despatch qty", "orders", "total orders", "order qty", "units sold", "sell out",
            "sellout", "sell in", "sellin", "demand"],
    "value": ["value", "sales value", "revenue", "gmv", "net sales", "amount", "nsv", "gsv", "net revenue",
              "sales amount", "invoice value", "billed value", "inr", "sales inr"],
    "channel": ["channel", "customer", "customer name", "account", "platform", "marketplace", "retailer",
                "distributor", "partner", "banner", "store", "outlet", "sold to", "ship to"],
    "channel_stock": ["channel stock", "partner stock", "partner inventory", "retailer stock",
                      "closing stock", "store stock", "channel inventory", "customer stock", "soh"],
    "product_name": ["product name", "description", "item description", "item name", "product",
                     "product description", "sku name", "sku description", "material description", "title",
                     "article description"],
    "category": ["category", "sub category", "subcategory", "brand line", "range", "franchise", "segment",
                 "line", "product line", "division"],
    "on_hand": ["on hand", "on hand qty", "soh", "stock on hand", "stock", "available", "available qty",
                "available stock", "closing stock", "inventory", "qty on hand", "free stock", "unrestricted",
                "unrestricted stock", "physical stock", "saleable", "saleable stock", "good stock", "atp",
                "current stock", "stock qty", "inventory qty", "wh stock", "dc stock"],
    "in_transit": ["in transit", "intransit", "git", "goods in transit", "open po", "open po qty", "po qty",
                   "inbound", "inbound qty", "incoming", "on order", "pending po", "transit qty"],
    "eta": ["eta", "expected date", "arrival date", "po eta", "delivery date", "expected arrival",
            "expected delivery", "due date", "grn date"],
    "location": ["location", "warehouse", "wh", "dc", "site", "plant", "depot", "warehouse name",
                 "location name", "fc", "fulfilment centre", "fulfillment center"],
    "status": ["status", "stock status", "inventory status", "stock type", "quality status", "condition"],
    "lead_time_days": ["lead time", "lead time days", "lt", "lt days", "leadtime", "replenishment lead time"],
    "moq": ["moq", "min order qty", "minimum order qty", "minimum order quantity", "min order"],
    "case_pack": ["case pack", "case size", "pack size", "carton qty", "carton size", "inner", "case qty",
                  "units per case"],
    "price": ["price", "mrp", "unit price", "asp", "selling price", "rate", "list price"],
    "event": ["event", "event name", "promo", "promotion", "promo name", "sale", "sale name", "festival",
              "campaign", "occasion"],
    "start": ["start", "start date", "from", "from date", "begin", "begins", "event date", "promo start",
              "sale start", "date"],
    "end": ["end", "end date", "to", "to date", "till", "until", "promo end", "sale end"],
    "pre_days": ["pre days", "pre window", "days before", "lead in"],
    "post_days": ["post days", "post window", "days after", "tail"],
}

NUMERIC_FIELDS = {"qty", "value", "channel_stock", "on_hand", "in_transit", "lead_time_days", "moq", "case_pack",
                  "price", "pre_days", "post_days"}
DATE_FIELDS = {"date", "eta", "start", "end"}
# Headers containing these words are plans, not actuals: never map them to quantities.
_EXCLUDE_FOR_QTY = re.compile(r"forecast|fcst|target|budget|plan|variance|accuracy|%|percent|growth")


def _norm(text) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text).lower())).strip()


@dataclass
class LoadReport:
    kind: str
    rows_in: int = 0
    rows_out: int = 0
    mapping: dict[str, str] = field(default_factory=dict)
    unmapped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    granularity: str | None = None
    date_min: str | None = None
    date_max: str | None = None
    n_skus: int = 0
    reshaped_from_wide: bool = False

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ----------------------------------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------------------------------
def read_table(src, sheet=None) -> pd.DataFrame:
    """Read CSV/Excel from a path, bytes or a Streamlit UploadedFile; auto-detect the header row."""
    name = getattr(src, "name", src if isinstance(src, str) else "")
    data = src.getvalue() if hasattr(src, "getvalue") else src
    is_excel = str(name).lower().endswith((".xlsx", ".xlsm", ".xls"))
    if isinstance(data, (bytes, bytearray)):
        buf = _io.BytesIO(data)
    else:
        buf = data
    if is_excel:
        raw = pd.read_excel(buf, sheet_name=sheet or 0, header=None)
    else:
        if isinstance(buf, _io.BytesIO):
            text = buf.getvalue().decode("utf-8-sig", errors="replace")
            buf = _io.StringIO(text)
        raw = pd.read_csv(buf, header=None, sep=None, engine="python", dtype=str, skip_blank_lines=True)
    return _promote_header(raw)


def _promote_header(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.dropna(how="all").dropna(axis=1, how="all")
    if raw.empty:
        return raw
    ncols = raw.shape[1]
    header_idx = raw.index[0]
    for idx in raw.index[:15]:
        row = raw.loc[idx]
        filled = row.notna().sum()
        texty = sum(isinstance(v, str) and not _looks_numeric(v) for v in row) + \
            sum(isinstance(v, (pd.Timestamp, np.datetime64)) for v in row)
        if filled >= max(2, 0.6 * ncols) and texty >= max(2, 0.5 * filled):
            header_idx = idx
            break
    header = [(_header_text(v) if pd.notna(v) else f"col_{i}") for i, v in enumerate(raw.loc[header_idx])]
    header = _dedupe(header)
    df = raw.loc[raw.index > header_idx].copy()
    df.columns = header
    df = df.dropna(how="all").reset_index(drop=True)
    return df


def _header_text(v) -> str:
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(v).strftime("%Y-%m-%d")
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()


def _dedupe(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}_{seen[n]}")
        else:
            seen[n] = 0
            out.append(n)
    return out


def _looks_numeric(v: str) -> bool:
    return bool(re.fullmatch(r"\s*[-+]?[\d,]*\.?\d+\s*", str(v)))


def to_number(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    txt = s.astype(str).str.replace(r"[,\s₹$]", "", regex=True).str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    txt = txt.replace({"": np.nan, "nan": np.nan, "None": np.nan, "-": np.nan, "NaN": np.nan})
    return pd.to_numeric(txt, errors="coerce")


# ----------------------------------------------------------------------------------------------
# Dates
# ----------------------------------------------------------------------------------------------
_MONTH_RE = re.compile(r"^([A-Za-z]{3,9})[\s\-'/_.]*(\d{2}|\d{4})$")
_YM_RE = re.compile(r"^(\d{4})[\-/.](\d{1,2})$")
_MY_RE = re.compile(r"^(\d{1,2})[\-/.](\d{4})$")
_ISOWEEK_RE = re.compile(r"^(\d{4})[\s\-]?W(?:K)?\s?(\d{1,2})$", re.I)


def parse_dates(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        out = pd.to_datetime(s)
        if getattr(out.dt, "tz", None) is not None:
            out = out.dt.tz_localize(None)
        return out.dt.normalize()
    num = to_number(s) if not pd.api.types.is_numeric_dtype(s) else s.astype(float)
    if num.notna().mean() > 0.9:
        vals = num.dropna()
        if vals.between(20000, 80000).all():
            return pd.to_datetime(num, unit="D", origin="1899-12-30", errors="coerce").dt.normalize()
        if vals.between(19000101, 21001231).all():
            return pd.to_datetime(num.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    txt = s.astype(str).str.strip()
    txt = txt.where(~txt.isin(["", "nan", "NaT", "None"]))
    sample = txt.dropna()
    if sample.empty:
        return pd.to_datetime(txt, errors="coerce")
    if sample.str.match(_ISOWEEK_RE).mean() > 0.8:
        parts = txt.str.extract(_ISOWEEK_RE)
        return pd.to_datetime(parts[0] + "-W" + parts[1].str.zfill(2) + "-1", format="%G-W%V-%u", errors="coerce")
    if sample.str.match(_MONTH_RE).mean() > 0.8:
        parts = txt.str.extract(_MONTH_RE)
        year = parts[1].where(parts[1].str.len() == 4, "20" + parts[1])
        return pd.to_datetime("01 " + parts[0].str[:3] + " " + year, format="%d %b %Y", errors="coerce")
    if sample.str.match(_YM_RE).mean() > 0.8:
        parts = txt.str.extract(_YM_RE)
        return pd.to_datetime(parts[0] + "-" + parts[1].str.zfill(2) + "-01", errors="coerce")
    if sample.str.match(_MY_RE).mean() > 0.8:
        parts = txt.str.extract(_MY_RE)
        return pd.to_datetime(parts[1] + "-" + parts[0].str.zfill(2) + "-01", errors="coerce")
    iso = pd.to_datetime(txt, format="ISO8601", errors="coerce")
    if iso.notna().sum() >= 0.9 * sample.size:
        return iso.dt.normalize()
    day_first = pd.to_datetime(txt, dayfirst=True, errors="coerce", format="mixed")
    month_first = pd.to_datetime(txt, dayfirst=False, errors="coerce", format="mixed")
    if month_first.notna().sum() > day_first.notna().sum():
        return month_first.dt.normalize()
    return day_first.dt.normalize()  # India convention (dd-mm-yyyy) wins ties


def detect_granularity(dates: pd.Series) -> tuple[str, int]:
    """Return ("D"|"W"|"M", typical days between observations)."""
    u = pd.Series(pd.to_datetime(dates.dropna().unique())).sort_values()
    if len(u) < 3:
        return "W", 7
    gap = float(u.diff().dt.days.dropna().median())
    if gap <= 2:
        return "D", 1
    if gap <= 10:
        return "W", 7
    return "M", 30


# ----------------------------------------------------------------------------------------------
# Column mapping
# ----------------------------------------------------------------------------------------------
def _score(field_name: str, header: str) -> float:
    h = _norm(header)
    if not h:
        return 0.0
    best = 0.0
    for syn in SYNONYMS.get(field_name, []) + [field_name.replace("_", " ")]:
        if h == syn:
            return 1.0
        if re.search(rf"(^| ){re.escape(syn)}( |$)", h):
            best = max(best, 0.80 + min(len(syn), 15) / 100)
        ratio = difflib.SequenceMatcher(None, h, syn).ratio()
        if ratio >= 0.82:
            best = max(best, ratio * 0.9)
    if field_name in {"qty", "value", "on_hand"} and _EXCLUDE_FOR_QTY.search(h):
        best *= 0.3
    return best


def auto_map(df: pd.DataFrame, kind: str) -> tuple[dict[str, str], list[str]]:
    """Map canonical field -> source column. Returns (mapping, warnings)."""
    schema = SCHEMAS[kind]
    fields = schema["required"] + schema["optional"]
    warnings: list[str] = []
    scored = []
    for f in fields:
        for col in df.columns:
            sc = _score(f, col)
            if sc >= 0.6:
                scored.append((sc, f, col))
    scored.sort(key=lambda t: -t[0])
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for sc, f, col in scored:
        if f in mapping or col in used:
            continue
        if f in DATE_FIELDS and _fill_rate(df[col], parse_dates) < 0.6:
            continue
        if f in NUMERIC_FIELDS and _fill_rate(df[col], to_number) < 0.6:
            continue
        mapping[f] = col
        used.add(col)

    # Content-based fallbacks for required fields.
    if "date" in schema["required"] and "date" not in mapping:
        for col in df.columns:
            if col not in used and parse_dates(df[col].head(200)).notna().mean() > 0.8:
                mapping["date"] = col
                used.add(col)
                warnings.append(f"Date column guessed from content: '{col}'.")
                break
    qty_field = "on_hand" if kind == "wms" else "qty"
    if qty_field in schema["required"] and qty_field not in mapping:
        generic = [c for c in df.columns if c not in used and _score("qty", c) >= 0.6]
        numeric = [c for c in df.columns if c not in used and not _EXCLUDE_FOR_QTY.search(_norm(c))
                   and to_number(df[c].head(200)).notna().mean() > 0.8]
        pick = (generic or numeric or [None])[0]
        if pick is not None:
            mapping[qty_field] = pick
            used.add(pick)
            warnings.append(f"Quantity column guessed: '{pick}'. Please confirm.")
    if "sku" in schema["required"] and "sku" not in mapping and "product_name" in mapping:
        mapping["sku"] = mapping["product_name"]
        warnings.append("No SKU code column found; using the product name as the SKU identifier.")
    return mapping, warnings


def _fill_rate(col: pd.Series, parser) -> float:
    """Share of non-blank cells (first 200) that the parser understands."""
    head = col.head(200)
    blank = head.isna() | head.astype(str).str.strip().isin(["", "nan", "None", "NaT"])
    if (~blank).sum() == 0:
        return 0.0
    return float(parser(head[~blank]).notna().mean())


def _melt_wide(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """If period columns run across (e.g. one column per week), turn them into rows."""
    date_cols = []
    for c in df.columns:
        if _norm(c) in {"total", "grand total"}:
            continue
        parsed = parse_dates(pd.Series([c]))
        if parsed.notna().all() and re.search(r"\d", str(c)):
            date_cols.append(c)
    if len(date_cols) >= 3 and len(date_cols) >= 0.5 * (df.shape[1] - 1):
        ids = [c for c in df.columns if c not in date_cols and _norm(c) not in {"total", "grand total"}]
        long = df.melt(id_vars=ids, value_vars=date_cols, var_name="Date", value_name="Qty")
        return long, True
    return df, False


def standardize(df: pd.DataFrame, kind: str, mapping: dict[str, str] | None = None
                ) -> tuple[pd.DataFrame, LoadReport]:
    rep = LoadReport(kind=kind, rows_in=len(df))
    if kind in {"sell_out", "sell_in"}:
        df, rep.reshaped_from_wide = _melt_wide(df)
        if rep.reshaped_from_wide:
            rep.warnings.append("Wide layout detected (one column per period) and reshaped to rows.")
    auto, warns = auto_map(df, kind)
    if mapping:
        auto.update({k: v for k, v in mapping.items() if v in df.columns})
    rep.mapping, rep.warnings = auto, rep.warnings + warns
    missing = [f for f in SCHEMAS[kind]["required"] if f not in auto]
    if missing:
        raise ValueError(f"{kind}: could not find column(s) for {missing}. Columns seen: {list(df.columns)}. "
                         f"Rename them or pass an explicit mapping.")
    rep.unmapped = [c for c in df.columns if c not in auto.values()]

    out = pd.DataFrame(index=df.index)
    for f, col in auto.items():
        if f in DATE_FIELDS:
            out[f] = parse_dates(df[col])
        elif f in NUMERIC_FIELDS:
            out[f] = to_number(df[col])
        else:
            out[f] = _clean_text(df[col])
    if kind in {"sell_out", "sell_in"}:
        out = _finish_sales(out, rep)
    elif kind == "wms":
        out = _finish_wms(out, rep)
    elif kind == "sku_master":
        out = out.dropna(subset=["sku"]).drop_duplicates("sku", keep="last")
    elif kind == "events":
        out = out.dropna(subset=["event", "start"])
    rep.rows_out = len(out)
    if "sku" in out:
        rep.n_skus = int(out["sku"].nunique())
    return out.reset_index(drop=True), rep


def _clean_text(s: pd.Series) -> pd.Series:
    def fix(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        t = str(v).strip()
        return re.sub(r"\.0$", "", t) if re.fullmatch(r"\d+\.0", t) else t
    return s.map(fix)


def _finish_sales(out: pd.DataFrame, rep: LoadReport) -> pd.DataFrame:
    n0 = len(out)
    out = out.dropna(subset=["date", "sku"])
    if len(out) < n0:
        rep.warnings.append(f"{n0 - len(out)} rows dropped: missing date or SKU.")
    nan_qty = int(out["qty"].isna().sum())
    if nan_qty:
        rep.warnings.append(f"{nan_qty} rows with blank quantity treated as 0.")
    out["qty"] = out["qty"].fillna(0.0)
    neg = int((out["qty"] < 0).sum())
    if neg:
        rep.warnings.append(f"{neg} rows with negative quantity (returns) kept as netting entries.")
    if "channel" not in out:
        out["channel"] = "ALL"
    out["channel"] = out["channel"].fillna("UNKNOWN")
    g, days = detect_granularity(out["date"])
    rep.granularity = {"D": "daily", "W": "weekly", "M": "monthly"}[g]
    rep.date_min = str(out["date"].min().date())
    rep.date_max = str(out["date"].max().date())
    return out


def _finish_wms(out: pd.DataFrame, rep: LoadReport) -> pd.DataFrame:
    out = out.dropna(subset=["sku"])
    out["on_hand"] = out["on_hand"].fillna(0.0)
    if "location" not in out:
        out["location"] = "ALL"
    if "in_transit" not in out:
        out["in_transit"] = 0.0
    out["in_transit"] = out["in_transit"].fillna(0.0)
    if "status" in out:
        bad = out["status"].astype(str).str.lower().str.contains(
            r"block|damag|expir|qc|quarantin|hold|reject|return|scrap|non.?sale", regex=True)
        if bad.any():
            rep.warnings.append(f"{int(bad.sum())} WMS rows with non-saleable status (blocked/damaged/expired/QC) "
                                f"excluded from available stock.")
            out.loc[bad, "on_hand"] = 0.0
    if "date" in out and out["date"].notna().any():
        rep.date_min = str(out["date"].min().date())
        rep.date_max = str(out["date"].max().date())
    return out


def load(src, kind: str, mapping: dict[str, str] | None = None, sheet=None) -> tuple[pd.DataFrame, LoadReport]:
    return standardize(read_table(src, sheet=sheet), kind, mapping)
