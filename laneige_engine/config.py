"""Engine settings. Every value has a sensible default so the engine runs hands-free."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class EngineConfig:
    # Planning bucket: "auto" (weekly for daily/weekly data, monthly for monthly data), "W" or "M".
    freq: str = "auto"
    # Forecast horizon in planning buckets. None -> 26 weeks or 12 months.
    horizon: int | None = None
    # Forecast level: "sku" (all channels summed) or "sku_channel".
    level: str = "sku"

    # Backtesting (rolling-origin) used to pick the best model per series.
    backtest_folds: int = 4

    # Festival / sale calendar.
    auto_align_sale_dates: bool = True     # snap estimated sale dates to spikes seen in the data
    disabled_events: list[str] = field(default_factory=list)

    # Supply planning.
    service_level: float = 0.95
    default_lead_time_days: int = 60       # used when WMS / SKU master has no lead time
    channel_cover_weeks: float = 4.0       # stock partners should hold, in weeks of forward sell-out
    excess_cover_weeks: float = 26.0       # DC + channel cover above this is flagged as excess
    order_cycle_weeks: float = 4.0         # each normal PO covers this many weeks of demand

    def resolved_horizon(self, freq: str) -> int:
        if self.horizon:
            return int(self.horizon)
        return 26 if freq == "W" else 12

    def to_dict(self) -> dict:
        return asdict(self)
