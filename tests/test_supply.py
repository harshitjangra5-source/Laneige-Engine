import numpy as np

from laneige_engine import supply as sup


def test_forward_cover_fractional():
    fc = np.array([10, 20, 30, 40.0])
    np.testing.assert_allclose(sup.forward_cover(fc, 1.5)[:2], [20 + 15, 30 + 20])


def test_sellin_preloads_before_peak():
    fc = np.array([10, 10, 10, 50, 50, 10, 10, 10.0])
    sellin, stock, target = sup.sellin_plan(fc, cs0=20, cover_buckets=2)
    np.testing.assert_allclose(stock, target)         # partners end each bucket at target cover
    assert sellin[1] > sellin[0]                        # pre-load ahead of the peak
    assert abs(sellin.sum() - (fc.sum() + stock[-1] - 20)) < 1e-9


def test_dc_plan_expedite_inside_lead_time_and_po_after():
    demand = np.full(12, 100.0)
    plan = sup.dc_plan(demand, on_hand=250, receipts=np.zeros(12), lt_buckets=4, safety_stock=50,
                       moq=300, case_pack=24, cycle_buckets=4)
    exp = plan["expedite"]
    assert exp[:4].any() and not exp[4:].any()
    assert (plan["projected"][:4] >= 0).all()
    normal = plan["planned_receipt"][4:]
    assert (normal[normal > 0] >= 300).all() and (normal[normal > 0] % 24 == 0).all()
    assert (plan["projected"][4:] >= 50 - 1e-9).all()


def test_no_orders_when_stock_is_ample():
    plan = sup.dc_plan(np.full(10, 10.0), on_hand=1000, receipts=np.zeros(10), lt_buckets=3,
                       safety_stock=20, moq=None, case_pack=None)
    assert plan["planned_receipt"].sum() == 0


def test_safety_stock_scales_with_error_and_service():
    d = np.full(10, 100.0)
    lo = sup.safety_stock(0.1, d, 4, 0.9)
    hi = sup.safety_stock(0.3, d, 4, 0.98)
    assert 0 < lo < hi


def test_estimate_channel_stock_uses_net_sell_in():
    so = np.full(20, 100.0)
    si = so.copy()
    si[-13:] += 10
    cs, how = sup.estimate_channel_stock(si, so, 4, 13)
    assert cs == 4 * 100 + 130 and "estimated" in how
