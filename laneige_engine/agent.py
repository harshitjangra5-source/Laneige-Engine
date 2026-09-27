"""Agentic layer: Claude drives the planning engine through tools.

* `PlanningSession` holds the data, settings, calendar overrides and the latest result.
* `ClaudePlanner` gives Claude tools over that session (run the forecast, inspect SKUs,
  check / correct festival dates, review supply risks) and either runs the whole planning
  cycle autonomously or answers planner questions in chat.
* `OfflinePlanner` offers the same interface without an API key (rule-based brief).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import pandas as pd

from . import calendar as cal
from . import insights as ins
from .config import EngineConfig
from .pipeline import PlanningInputs, PlanningResult, run_planning

DEFAULT_MODEL = "claude-opus-5"
EventCallback = Callable[[str, str], None]

SYSTEM_PROMPT = """You are the demand & supply planning agent for Laneige in India (K-beauty: lip care, skincare, \
sun care, makeup, gift sets), working for the brand's S&OP planner.

You operate a statistical forecasting engine through tools. The engine:
- takes sell-out (consumer sales by SKU/channel), sell-in (shipments to partners) and WMS (DC stock + inbound);
- decomposes each SKU's history into trend x seasonality x festival/sale uplift, with uplifts tied to each year's \
actual event dates (Diwali, Navratri/Dussehra, Holi, Raksha Bandhan, Eid, Christmas, Valentine's, Mother's Day, \
Amazon GIF / Flipkart BBD festive sale, Nykaa Pink Friday, Prime Day, Republic/Independence Day sales, mid-year \
sales, plus the planner's own promos). When a festival moves (e.g. Diwali 20-Oct-2025 -> 8-Nov-2026) the uplift \
moves with it;
- backtests several models per SKU (seasonal regression, damped ETS, Theta, last-year festival-aligned, moving \
average, Croston, ensemble) and keeps the most accurate one;
- converts sell-out forecast into a sell-in plan (partners hold N weeks of forward cover, so they pre-load ahead \
of festivals) and a DC supply plan (safety stock, lead time, MOQ, expedites, POs to place).

How to work:
- Use tools for every number you quote. Never invent or estimate figures the tools did not return.
- Compare against last year *aligned to this year's festival dates* as well as raw last year; explain gaps that \
come from festival timing rather than real growth.
- Treat calendar dates marked "rule estimate", "approx" or "verify" as assumptions to confirm with the planner. \
If the planner gives you actual dates, apply them with set_event_dates and re-run the forecast.
- If accuracy is poor for a SKU, look for causes (stockouts, unexplained spikes that suggest missing promos, \
new-item history, intermittent demand) before blaming the model.
- Be concise and action-oriented. Use Indian conventions for money (₹ lakh / crore) and dd-Mon-yyyy dates. \
Format answers in Markdown with short sections and bullet points; use small tables only when they help."""

AUTONOMOUS_TASK = """Run this planning cycle end to end and write the S&OP brief.

1. Check the data overview and data quality. Note anything that could distort the forecast.
2. Review the festival & sale calendar for the forecast horizon: which events moved vs last year, and which \
dates are estimates the planner must confirm.
3. Run the forecast with the current settings (change them only if the data clearly calls for it, and say why).
4. Review the portfolio summary, the biggest and fastest-moving SKUs, low-accuracy SKUs and the supply actions. \
Drill into the 2-4 SKUs that matter most.
5. Write the brief with these sections:
   - Headline (3 bullets: volume outlook vs LY and vs festival-aligned LY, accuracy, biggest risk)
   - Festival & sale timing effects (month by month where it matters)
   - Trend & seasonality (growers, decliners, seasonal peaks ahead)
   - Supply actions (expedites, POs to place now, excess), with units and value
   - Forecast exceptions and why
   - Dates & data to confirm (questions for the planner)"""


@dataclass
class PlanningSession:
    inputs: PlanningInputs
    config: EngineConfig = field(default_factory=EngineConfig)
    overrides: list[dict] = field(default_factory=list)
    result: PlanningResult | None = None
    progress: Callable[[float, str], None] | None = None

    def events_frame(self) -> pd.DataFrame | None:
        frames = []
        if self.inputs.events is not None and not self.inputs.events.empty:
            frames.append(self.inputs.events)
        if self.overrides:
            frames.append(pd.DataFrame(self.overrides))
        return pd.concat(frames, ignore_index=True) if frames else None

    def run(self) -> PlanningResult:
        inputs = replace(self.inputs, events=self.events_frame())
        self.result = run_planning(inputs, self.config, self.progress)
        return self.result

    def require(self) -> PlanningResult:
        return self.result if self.result is not None else self.run()

    def set_event(self, event: str, start: str, end: str | None = None) -> str:
        s = cal.parse_date(start)
        e = cal.parse_date(end) if end else s
        if pd.isna(s) or pd.isna(e):
            return f"Could not read dates '{start}' / '{end}'. Use YYYY-MM-DD."
        key = cal.match_event_key(event)
        self.overrides = [o for o in self.overrides
                          if not ((cal.match_event_key(o["event"]) or o["event"]) == (key or event)
                                  and pd.Timestamp(o["start"]).year == s.year)]
        self.overrides.append({"event": event, "start": s, "end": e})
        self.result = None
        label = cal.EVENTS[key].name if key else f"custom event '{event}'"
        return f"Calendar updated: {label} {s:%d-%b-%Y} to {e:%d-%b-%Y}. Re-run the forecast to apply it."


def _dump(obj) -> str:
    return json.dumps(ins._clean(obj), ensure_ascii=False, default=str)


def build_tools(session: PlanningSession) -> list:
    from anthropic import beta_tool

    @beta_tool
    def get_data_overview() -> str:
        """Summarise the loaded files: column mapping used, granularity, date range, SKUs, channels and warnings."""
        inp = session.inputs
        out = {"datasets": {k: {kk: v for kk, v in r.to_dict().items() if kk != "unmapped"}
                            for k, r in inp.reports.items()},
               "sell_out_channels": sorted(map(str, inp.sell_out["channel"].unique()))[:20],
               "sell_out_skus": int(inp.sell_out["sku"].nunique()),
               "settings": session.config.to_dict(),
               "calendar_overrides": session.overrides,
               "forecast_ready": session.result is not None}
        return _dump(out)

    @beta_tool
    def run_forecast(horizon: int = 0, level: str = "", service_level: float = 0.0,
                     channel_cover_weeks: float = 0.0, default_lead_time_days: int = 0) -> str:
        """Run (or re-run) the full forecast and supply plan and return the portfolio summary.

        Args:
            horizon: Forecast horizon in planning buckets (weeks or months). 0 keeps the current setting.
            level: "sku" (all channels summed) or "sku_channel". Empty keeps the current setting.
            service_level: Target service level for safety stock, e.g. 0.95. 0 keeps the current setting.
            channel_cover_weeks: Weeks of forward sell-out partners should hold. 0 keeps the current setting.
            default_lead_time_days: Lead time when the SKU master has none. 0 keeps the current setting.
        """
        c = session.config
        if horizon:
            c.horizon = int(horizon)
        if level in {"sku", "sku_channel"}:
            c.level = level
        if service_level:
            c.service_level = float(service_level)
        if channel_cover_weeks:
            c.channel_cover_weeks = float(channel_cover_weeks)
        if default_lead_time_days:
            c.default_lead_time_days = int(default_lead_time_days)
        res = session.run()
        return _dump(ins.portfolio_overview(res))

    @beta_tool
    def get_portfolio_summary() -> str:
        """Headline KPIs, month-by-month forecast vs last year and vs festival-aligned last year, and category mix."""
        return _dump(ins.portfolio_overview(session.require()))

    @beta_tool
    def get_festival_calendar(year: int = 0, upcoming_only: bool = True) -> str:
        """Festival & sale dates with year-on-year shift, date source and confidence.

        Args:
            year: Only this calendar year (0 = all years).
            upcoming_only: Only events on/after the plan start date.
        """
        res = session.require()
        return _dump({"events": ins.calendar_view(res, year or None, upcoming_only),
                      "adjustments_made": res.calendar_log})

    @beta_tool
    def set_event_dates(event: str, start_date: str, end_date: str = "") -> str:
        """Set the actual dates of a festival, marketplace sale or brand promotion (past or future), then re-run.

        Args:
            event: Event name, e.g. "Big Billion Days", "Diwali", "Nykaa Pink Friday" or a new promo name.
            start_date: First day, YYYY-MM-DD.
            end_date: Last day, YYYY-MM-DD (defaults to start_date).
        """
        return session.set_event(event, start_date, end_date or None)

    @beta_tool
    def list_skus(sort_by: str = "volume", limit: int = 10) -> str:
        """Rank SKUs with their trend, seasonality, model, accuracy and worst alert.

        Args:
            sort_by: One of "volume", "growth", "decline", "accuracy", "worst_accuracy".
            limit: Number of SKUs to return.
        """
        return _dump(ins.top_skus(session.require(), sort_by, int(limit)))

    @beta_tool
    def get_sku_insights(sku: str) -> str:
        """Everything about one SKU: trend, seasonal index, festival lifts, model tournament, forecast, supply.

        Args:
            sku: SKU code or part of the product name.
        """
        return _dump(ins.sku_card(session.require(), sku))

    @beta_tool
    def get_supply_actions(limit: int = 20) -> str:
        """Supply alerts (stockouts inside lead time, POs to place now, excess) and the order list.

        Args:
            limit: Maximum alerts / planned orders to return.
        """
        return _dump(ins.alerts_view(session.require(), int(limit)))

    @beta_tool
    def get_data_quality() -> str:
        """Data issues, engine log, calendar auto-adjustments and unexplained spikes/dips (possible missing promos)."""
        return _dump(ins.data_quality_view(session.require()))

    return [get_data_overview, run_forecast, get_portfolio_summary, get_festival_calendar, set_event_dates,
            list_skus, get_sku_insights, get_supply_actions, get_data_quality]


class ClaudePlanner:
    def __init__(self, session: PlanningSession, api_key: str | None = None, model: str = DEFAULT_MODEL,
                 effort: str = "high"):
        import anthropic

        self.anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.session = session
        self.model = model
        self.effort = effort
        self.tools = build_tools(session)
        self.history: list[dict] = []

    def ask(self, message: str, on_event: EventCallback | None = None) -> str:
        emit = on_event or (lambda kind, text: None)
        messages = self.history + [{"role": "user", "content": message}]
        runner = self.client.beta.messages.tool_runner(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            tools=self.tools,
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            max_iterations=30,
        )
        final = None
        try:
            for msg in runner:
                final = msg
                for block in msg.content:
                    if block.type == "tool_use":
                        emit("tool", f"{block.name}({_short_args(block.input)})")
                    elif block.type == "text" and msg.stop_reason == "tool_use" and block.text.strip():
                        emit("note", block.text.strip())
        except self.anthropic.AuthenticationError:
            raise RuntimeError("The Anthropic API key was rejected. Check the key in the sidebar or "
                               "ANTHROPIC_API_KEY.") from None
        except self.anthropic.RateLimitError:
            raise RuntimeError("Anthropic rate limit reached; try again in a minute.") from None
        except self.anthropic.APIConnectionError:
            raise RuntimeError("Could not reach the Anthropic API (network).") from None
        if final is None:
            raise RuntimeError("No response from Claude.")
        if final.stop_reason == "refusal":
            text = "Claude declined this request. Try rephrasing the question."
        else:
            text = "\n".join(b.text for b in final.content if b.type == "text").strip()
            if final.stop_reason == "max_tokens":
                text += "\n\n_(Response truncated: ask me to continue.)_"
        self.history += [{"role": "user", "content": message}, {"role": "assistant", "content": text or "(no text)"}]
        return text

    def run_autonomous(self, on_event: EventCallback | None = None) -> str:
        return self.ask(AUTONOMOUS_TASK, on_event)


class OfflinePlanner:
    """Same interface as ClaudePlanner, without an LLM: deterministic brief + keyword routing."""

    def __init__(self, session: PlanningSession):
        self.session = session
        self.history: list[dict] = []

    def run_autonomous(self, on_event: EventCallback | None = None) -> str:
        emit = on_event or (lambda kind, text: None)
        emit("tool", "run_forecast()")
        res = self.session.require()
        return ins.build_brief(res)

    def ask(self, message: str, on_event: EventCallback | None = None) -> str:
        res = self.session.require()
        m = message.lower()
        note = ("_Claude is not connected (no API key), so this is a rule-based answer. "
                "Add an Anthropic API key in the sidebar for full conversational analysis._\n\n")
        skus = [s for s in res.profiles["sku"] if s.lower() in m]
        if not skus:
            for s, n in zip(res.profiles["sku"], res.profiles["product_name"].fillna("")):
                if n and n.lower() in m:
                    skus.append(s)
        if skus:
            c = ins.sku_card(res, skus[0])
            p, mo = c["profile"], c["model"]
            lines = [f"### {c['product_name'] or c['sku']} ({c['sku']})",
                     f"- Trend: {p['trend_label']} (YoY {p['yoy_growth'] if p['yoy_growth'] is not None else 'n/a'})",
                     f"- Seasonality: {p['seasonality_label']}; peaks {p['peak_months'] or 'n/a'}",
                     f"- Model: {mo['model']}, backtest accuracy "
                     f"{mo['accuracy']:.0%}" if mo.get("accuracy") is not None else f"- Model: {mo['model']}",
                     f"- Top events: {p['top_events'] or 'n/a'}"]
            for a in c["alerts"]:
                lines.append(f"- **{a['severity'].upper()}** {a['detail']}")
            return note + "\n".join(lines)
        if re.search(r"alert|order|stock|expedite|po\b|supply|excess", m):
            v = ins.alerts_view(res, 15)
            lines = [f"- **{a['severity'].upper()}** {a['product_name'] or a['sku']}: {a['detail']}" for a in v["alerts"]]
            return note + ("\n".join(lines) if lines else "No supply alerts.")
        if re.search(r"festival|diwali|calendar|sale|shift|holi|rakhi|bbd|pink", m):
            ev = ins.calendar_view(res, upcoming_only=True)
            return note + "\n".join(f"- {e['event']} {e['year']}: {e['start']} to {e['end']} "
                                    f"({e['shift_vs_ly_days'] if e['shift_vs_ly_days'] is not None else 'n/a'} days vs LY, "
                                    f"{e['source']})" for e in ev[:20])
        return note + ins.build_brief(res)


def _short_args(d: dict) -> str:
    s = ", ".join(f"{k}={v!r}" for k, v in (d or {}).items())
    return s if len(s) < 80 else s[:77] + "..."


def make_planner(session: PlanningSession, api_key: str | None = None, model: str = DEFAULT_MODEL,
                 effort: str = "high"):
    """ClaudePlanner when credentials are available, otherwise OfflinePlanner."""
    has_creds = bool(api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                     or os.environ.get("ANTHROPIC_PROFILE")
                     or Path("~/.config/anthropic").expanduser().exists())
    if not has_creds:
        return OfflinePlanner(session)
    try:
        return ClaudePlanner(session, api_key=api_key, model=model, effort=effort)
    except Exception:
        return OfflinePlanner(session)
