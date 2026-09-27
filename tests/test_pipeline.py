import io
import json

import numpy as np
import openpyxl
import pytest

from laneige_engine import insights as ins
from laneige_engine.agent import OfflinePlanner, PlanningSession, build_tools, make_planner
from laneige_engine.config import EngineConfig
from laneige_engine.excel import to_excel
from laneige_engine.pipeline import run_planning
from laneige_engine.sample_data import demo_inputs


@pytest.fixture(scope="module")
def inputs():
    return demo_inputs()


@pytest.fixture(scope="module")
def result(inputs):
    return run_planning(inputs)


def test_demo_end_to_end(result):
    assert result.freq == "W" and result.grid.horizon == 26
    assert len(result.forecast) == 11 * 26
    assert np.isfinite(result.forecast[["forecast", "p10", "p90"]].to_numpy()).all()
    assert (result.forecast["p10"] <= result.forecast["forecast"]).all()
    assert (result.forecast["forecast"] <= result.forecast["p90"]).all()
    assert result.summary["portfolio_accuracy"] > 0.8
    assert not result.alerts.empty and not result.orders.empty
    assert "Diwali 2026: 08-Nov" in " ".join(result.summary["festival_moves_in_horizon"])


def test_demo_calibration_recovers_true_sale_dates(result):
    log = " ".join(result.calendar_log)
    assert "Nykaa Pink Friday / Black Friday sale 2024: auto-aligned -3 days" in log
    assert "Mid-year sale (Myntra EORS / Nykaa Hot Pink) 2025: auto-aligned +4 days" in log


def test_demo_trend_and_seasonality_recovered(result):
    p = result.profiles.set_index("sku")
    assert p.loc["LNG-LSM-GUM-20", "trend_label"] == "Growing"
    assert p.loc["LNG-WBS-50", "trend_label"] == "Declining"
    assert "Dec" in p.loc["LNG-LSM-BER-20", "peak_months"] or "Jan" in p.loc["LNG-LSM-BER-20", "peak_months"]
    assert "May" in p.loc["LNG-RCS-50", "peak_months"] or "Jun" in p.loc["LNG-RCS-50", "peak_months"]


def test_brief_and_excel(result):
    brief = ins.build_brief(result)
    for section in ("Festival & sale shifts", "Trend & seasonality", "Supply actions"):
        assert section in brief
    wb = openpyxl.load_workbook(io.BytesIO(to_excel(result, brief)))
    for sheet in ("Summary", "Forecast", "DC supply plan", "Festival calendar", "Model accuracy"):
        assert sheet in wb.sheetnames


def test_sku_channel_level(inputs):
    res = run_planning(inputs, EngineConfig(level="sku_channel", horizon=8))
    assert {"sku", "channel"} <= set(res.forecast.columns)
    assert res.forecast["channel"].nunique() == 6
    assert not res.supply_plan.empty


def test_agent_tools_offline(inputs):
    session = PlanningSession(inputs, EngineConfig(horizon=10))
    tools = {t.name: t for t in build_tools(session)}
    summary = json.loads(tools["run_forecast"].call({}))
    assert summary["horizon_buckets"] == 10
    for name, args in [("get_portfolio_summary", {}), ("get_festival_calendar", {"year": 2026}),
                       ("list_skus", {"sort_by": "growth", "limit": 3}),
                       ("get_sku_insights", {"sku": "LNG-LSM-BER-20"}), ("get_supply_actions", {"limit": 5}),
                       ("get_data_quality", {}), ("get_data_overview", {})]:
        json.loads(tools[name].call(args))
    msg = tools["set_event_dates"].call({"event": "Flipkart Big Billion Days", "start_date": "2026-10-04",
                                         "end_date": "2026-10-11"})
    assert "04-Oct-2026" in msg
    tools["run_forecast"].call({})
    cal = json.loads(tools["get_festival_calendar"].call({"year": 2026}))["events"]
    fest = [e for e in cal if e["event"].startswith("Festive mega sale")][0]
    assert fest["start"] == "2026-10-04" and fest["source"] == "your calendar"


def test_offline_planner(inputs, monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HOME", "/nonexistent")
    planner = make_planner(PlanningSession(inputs, EngineConfig(horizon=8)))
    assert isinstance(planner, OfflinePlanner)
    assert "S&OP brief" in planner.run_autonomous()
    assert "Lip Sleeping Mask Berry 20g" in planner.ask("How is LNG-LSM-BER-20 doing?")
