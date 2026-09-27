# Laneige Demand & Supply Engine

An agentic demand-forecasting and supply-planning engine for India. You give it **sell-out**, **sell-in** and
**WMS** data. It then does the rest:

- **Trend & seasonality:** calculates trend and seasonality for every SKU on its own.
- **Festival & sale effects:** learns how much each festival and marketplace sale lifts sales, including the
  dips before and after.
- **Date shifting:** moves those effects to this year's dates. Diwali on 20-Oct-2025 becomes 8-Nov-2026, and
  estimated sale dates such as BBD/GIF, Pink Friday and EORS are corrected from your own data.
- **Model choice:** backtests several models per SKU and keeps the most accurate one.
- **Sell-in & supply plan:** turns the forecast into a sell-in plan and a DC supply plan with safety stock,
  expedites and POs to place.
- **AI planner:** a Claude agent runs the cycle, writes the S&OP brief and answers questions.

Everything is in a Streamlit web app with a one-click Excel export.

---

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Click **Load demo data** in the sidebar to see a full run on simulated Laneige-style data. It covers
11 SKUs, 6 channels and daily sell-out from Jan-2023 to Sep-2026. Then upload your own files.

To enable the AI planner, set an Anthropic API key. You can export it before starting the app or paste it
in the sidebar:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Without a key everything still works. The "AI planner" tab then uses a rule-based brief and simple answers.

---

## What to upload

CSV or Excel. Column names are matched automatically, and you can correct any mapping in the **Data** tab.
See `templates/` for examples.

| File | Required | Optional | Notes |
|---|---|---|---|
| **Sell-out** | date, sku, qty | channel, value, channel_stock, product_name, category | Consumer sales (Nykaa, Amazon, Flipkart, Myntra, Tira, offline…). Daily, weekly or monthly. |
| **Sell-in** | date, sku, qty | channel, value | Your shipments/invoices to partners. Used to estimate partner stock and the sell-in plan. |
| **WMS** | sku, on_hand | date, location, in_transit, eta, status, lead_time_days, moq, case_pack | DC stock. Blocked/QC/expired statuses are excluded. With several snapshot dates, zero-stock periods are treated as stockouts. |
| **SKU master** | sku | product_name, category, lead_time_days, moq, case_pack, price | Lead time, MOQ and case pack drive the PO plan; price gives ₹ values. |
| **Promo / event calendar** | event, start | end, pre_days, post_days, category | Your brand days and confirmed sale dates, past and planned. These always override the built-in calendar. |

The loader handles these layouts automatically:

- **Headers and titles:** title rows above the header and fuzzy header names (*Week Commencing*, *Invoice
  Date*, *Material*, *Billed Qty*, *Customer*, *Unrestricted Stock*, *GIT*…).
- **Date formats:** dd-mm-yyyy (Indian default), ISO dates, Excel serial dates, *Jan-24*, *2024-W23*.
- **Wide layouts:** one column per week or month is reshaped to rows.
- **Numbers:** thousands separators and ₹ signs.

Planning bucket: weekly for daily or weekly data, monthly for monthly data (you can override this). Weeks
follow your data's week start, e.g. Monday "week commencing".

---

## How festival & sale shifts are handled

Every event is stored as dated occurrences, one or more per year. For each planning bucket the engine
computes **what share of the bucket falls inside each event phase**. Each event has three phases:

- **pre:** gifting and anticipation, e.g. the 21 days before Diwali, which covers Karwa Chauth and Dhanteras.
- **main:** the event or sale days themselves.
- **post:** the usual dip after the event.

Uplifts are learned per phase, per SKU. They are then applied to next year's buckets using next year's dates.
The uplift follows the date, not the calendar week or month, so:

- **Weekly:** the Diwali build-up moves from late-Sep/Oct 2025 weeks to mid-Oct/early-Nov 2026 weeks.
- **Monthly:** October 2026 is forecast below October 2025 and November above, because the festival moved.
- **Festival-aligned LY:** the **"last year, festival-aligned"** line shows last year's actuals with festival
  volume moved to this year's dates. Planners can then separate real growth from calendar timing.

**Where dates come from**

| Source | Events |
|---|---|
| Lunar tables (2017–2030) | Diwali, Holi, Raksha Bandhan, Navratri → Dussehra, Eid ul-Fitr |
| Fixed / nth-weekday | Valentine's, Women's Day, Mother's Day, Christmas & New Year |
| Reported (approx.) | Amazon GIF / Flipkart BBD festive sale 2019–2025, Prime Day 2019–2025 |
| Rule estimates | Future festive sale (~3 days before Navratri), Pink Friday (week of Black Friday), Prime Day, Republic/Independence Day sales, mid-year sale (EORS / Hot Pink) |
| **Auto-aligned** | Estimated historic sale windows are snapped (±7 days) to the actual spike in your sell-out. Future estimates are shifted by the offset learned from past years. |
| **Your calendar** | Uploaded file, the calendar editor in the app, or telling the AI planner ("BBD starts 4-Oct-2026"). Always wins. |

Please verify future lunar dates against the official holiday list, and confirm marketplace sale dates when
they are announced. The app and the AI brief list every date that is still an estimate.

---

## Forecasting method

For each SKU (or SKU × channel):

1. **Decompose.** Sales = baseline × (1 + festival/sale uplift). The baseline is log-level + piecewise-linear
   trend + annual Fourier seasonality. The fit alternates between baseline and uplift, with robust weights so
   one-off spikes don't bend the baseline.
2. **Borrow strength.** Uplifts are shrunk toward the category's (or channel's) pooled estimate. SKUs with
   under ~15 months of history borrow the category's seasonality, which handles new launches.
3. **Clean history.** Stockouts are excluded rather than treated as low demand. They come from WMS zero stock,
   zero partner stock, or runs of zero sales in an actively selling SKU. Unexplained spikes or dips are clipped
   and listed as *possible missing promotions*.
4. **Model tournament.** Six candidate models are scored (see below). A rolling-origin backtest runs four
   folds of 13 weeks (or 3 months), covering a full year including the festive season. The best model or
   ensemble is chosen by WAPE.
5. **Intervals & explanations.** 80% intervals come from the backtest errors. Outputs also include YoY growth
   (festival-adjusted), trend momentum, seasonal index by month, the demand class (smooth / erratic /
   intermittent / lumpy), and the incremental units per event.

The six candidate models:

- Seasonal regression + events
- Damped-trend ETS
- Theta
- Last-year festival-aligned × growth
- Seasonal moving average
- Croston SBA (intermittent SKUs only)

On the demo data the engine recovers the simulated growth rates (e.g. +31% vs +30% true), seasonal peaks and
hidden sale-date shifts. Backtest accuracy is about 92% at weekly SKU level, against about 79% for a naive
last-year forecast.

## Sell-in & supply logic

- **Sell-in plan.** Partners should hold `channel_cover_weeks` (default 4) of *forward* sell-out. Because the
  target looks forward, they are pre-loaded ahead of Diwali and sale events. Partner stock comes from the
  `channel_stock` column; if that is missing, it is estimated from sell-in minus sell-out.
- **DC plan.** Projected on-hand = WMS on-hand + inbound (by ETA) − sell-in plan. Safety stock = z(service
  level) × forecast error × √(lead time + order cycle).
- **Orders.** Inside the lead time, shortfalls become one **EXPEDITE** line. After the lead time, POs keep
  stock above safety stock, cover `order_cycle_weeks` and are rounded to MOQ and case pack. They are marked
  **PLACE NOW** or **PLANNED** with a release date.
- **Alerts.** Stockout inside lead time, place PO now, below safety stock, excess cover (DC + channel), stock
  with no sales, and overdue inbound.

---

## The AI planner (Claude)

`laneige_engine/agent.py` gives Claude these tools over the engine:

| Tool | What it does |
|---|---|
| `get_data_overview` | Column mapping, granularity, date range, warnings |
| `run_forecast` | Run or re-run with a different horizon, level, service level, cover or lead time |
| `get_portfolio_summary` | KPIs, month-by-month vs LY and vs festival-aligned LY |
| `get_festival_calendar` / `set_event_dates` | Review dates and shifts; apply confirmed dates |
| `list_skus` / `get_sku_insights` | Rankings; per-SKU trend, seasonality, lifts, model tournament, forecast, supply |
| `get_supply_actions` | Expedites, POs to place now, excess |
| `get_data_quality` | Data issues, calendar auto-adjustments, unexplained spikes |

**Run AI planning cycle** makes Claude do the whole review and write the S&OP brief. The review covers data
checks, calendar shifts, the forecast run, drill-downs into key SKUs and supply actions. The chat box answers
questions such as "why is October below last year?" or "BBD starts 4-Oct-2026, update the plan". The agent
quotes only numbers returned by the tools.

API usage:

- **Model:** `claude-opus-5` by default (changeable in the sidebar), with adaptive thinking.
- **Refusal fallback:** server-side refusal fallback (`fallbacks="default"`) is enabled.
- **Tool loop:** the Anthropic SDK tool runner drives the tool calls.

## Excel export

The Excel export contains these sheets:

- Summary (KPIs + AI brief)
- Monthly vs LY
- Forecast (long) and Forecast pivot
- Sell-in plan
- DC supply plan
- Orders
- Alerts
- Trend & seasonality
- Event lifts
- Portfolio lifts
- Festival calendar
- Festival shift bridge
- Model accuracy and Model tournament
- Anomalies
- Data quality
- Settings

---

## Using it from Python

```python
from laneige_engine.io import load
from laneige_engine.pipeline import PlanningInputs, run_planning
from laneige_engine.config import EngineConfig
from laneige_engine.excel import to_excel

so, r1 = load("sell_out.xlsx", "sell_out")
si, r2 = load("sell_in.csv", "sell_in")
wms, r3 = load("wms.csv", "wms")
res = run_planning(PlanningInputs(sell_out=so, sell_in=si, wms=wms,
                                  reports={"sell_out": r1, "sell_in": r2, "wms": r3}),
                   EngineConfig(horizon=26, service_level=0.95))
open("plan.xlsx", "wb").write(to_excel(res))
```

`python -m laneige_engine.sample_data` writes the demo files to `sample_data/`.

## Project layout

```
app.py                      Streamlit web app
laneige_engine/
  io.py                     file reading, auto column mapping, date parsing
  calendar.py               India festival & sale calendar, date shifting, auto-alignment
  models.py                 event-aware decomposition + baseline models (ETS, Theta, Croston)
  forecaster.py             backtests, model selection, intervals, profiles, shift bridge
  supply.py                 sell-in plan, safety stock, MRP netting
  pipeline.py               end-to-end run, alerts, data quality, summary
  insights.py               shared views + offline brief
  agent.py                  Claude agent (tools, autonomous cycle, chat) + offline fallback
  charts.py / excel.py      visuals and workbook
  sample_data.py            simulated demo data
templates/                  example input files
tests/                      pytest suite
```

Run the tests with `pytest`.

## Settings

| Setting | Default | Meaning |
|---|---|---|
| Planning bucket | auto | Weekly (daily/weekly data) or monthly (monthly data) |
| Horizon | 26 weeks / 12 months | Forecast length |
| Level | SKU | or SKU × channel |
| Service level | 95% | Safety-stock target |
| Default lead time | 60 days | Used when the SKU master / WMS has none |
| Partner cover | 4 weeks | Forward sell-out partners should hold |
| PO cycle | 4 weeks | Demand covered by each normal PO |
| Excess threshold | 26 weeks | DC + channel cover above this is flagged |
| Auto-align sale dates | on | Snap estimated sale windows to spikes in your data |

## Known limitations / next steps

- **Promotions:** price and discount depth are not modelled yet; promotions enter only through the calendar.
  A `discount_pct` driver is the natural next step.
- **Reconciliation:** SKU × channel forecasts are not reconciled to SKU totals; the DC plan sums channel sell-in.
- **Future dates:** lunar dates beyond 2026 and all future marketplace sale dates must be confirmed; the app
  flags them.
- **History length:** at least ~1 year of history per category is needed to learn festival lifts. New SKUs
  borrow from their category.
