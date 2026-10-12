import numpy as np
import pandas as pd
import pytest

from swingbot.execution.simulator import CostModel, simulate
from swingbot.reporting.tax import _net_tax, after_tax
from tests.conftest import make_panel

FREE = CostModel(fee_bps=0, slippage_bps={"default": 0})


def targets_for(panel, rows: dict[int, dict]):
    t = pd.DataFrame(np.nan, index=panel.index, columns=panel.assets)
    for i, w in rows.items():
        t.iloc[i] = 0.0
        for a, x in w.items():
            t.iloc[i, t.columns.get_loc(a)] = x
    return t


def test_signal_executes_next_open_not_same_bar():
    # price jumps at the close of day 5; a target decided at day 5's close must NOT capture the jump
    c = [100.0] * 5 + [110.0] * 5
    p = make_panel({"X": c})
    res = simulate(targets_for(p, {5: {"X": 1.0}}), p, FREE)
    assert res.equity.iloc[-1] == pytest.approx(1.0)
    # deciding one bar earlier (day 4) does capture it
    res2 = simulate(targets_for(p, {4: {"X": 1.0}}), p, FREE)
    assert res2.equity.iloc[-1] == pytest.approx(1.1)


def test_delay_shifts_execution_by_one_more_bar():
    c = [100.0] * 5 + [110.0] * 5
    p = make_panel({"X": c})
    res = simulate(targets_for(p, {4: {"X": 1.0}}), p, FREE, delay=1)
    assert res.equity.iloc[-1] == pytest.approx(1.0)


def test_decision_before_start_carries_into_first_day():
    c = [100.0] * 5 + [200.0] * 5
    p = make_panel({"X": c})
    res = simulate(targets_for(p, {0: {"X": 1.0}}), p, FREE, start=p.index[3])
    assert res.equity.iloc[-1] == pytest.approx(2.0)


def test_round_trip_costs():
    p = make_panel({"X": [100.0] * 10})
    cm = CostModel(fee_bps=25, slippage_bps={"default": 5})
    res = simulate(targets_for(p, {1: {"X": 1.0}, 4: {"X": 0.0}}), p, cm)
    r = 30 / 1e4
    assert res.equity.iloc[-1] == pytest.approx((1 / (1 + r)) * (1 - r))
    assert res.weights.iloc[-1].sum() == 0


def test_no_leverage_and_weights_normalized():
    p = make_panel({"X": [100.0] * 6, "Y": [50.0] * 6})
    res = simulate(targets_for(p, {0: {"X": 0.9, "Y": 0.9}}), p, FREE)
    assert res.weights.sum(axis=1).max() <= 1 + 1e-12
    assert res.weights.iloc[-1]["X"] == pytest.approx(0.5)


def test_no_trade_band_skips_small_rebalances():
    p = make_panel({"X": [100.0] * 10})
    t = targets_for(p, {0: {"X": 0.50}, 3: {"X": 0.51}})
    res = simulate(t, p, FREE, band=0.02)
    assert res.weights.iloc[-1]["X"] == pytest.approx(0.50)
    res2 = simulate(t, p, FREE, band=0.0)
    assert res2.weights.iloc[-1]["X"] == pytest.approx(0.51)


def test_hold_rows_do_nothing_and_missing_prices_are_carried():
    c = [100.0, 100.0, 120.0, np.nan, 120.0, 120.0]
    p = make_panel({"X": c}, opens={"X": [100.0, 100.0, 100.0, np.nan, 120.0, 120.0]})
    res = simulate(targets_for(p, {0: {"X": 1.0}}), p, FREE)
    assert res.equity.iloc[3] == pytest.approx(1.2)
    assert res.equity.iloc[-1] == pytest.approx(1.2)


def test_episodes_and_lots():
    c = [100.0] * 3 + [150.0] * 500
    p = make_panel({"X": c})
    res = simulate(targets_for(p, {0: {"X": 1.0}, 450: {"X": 0.0}}), p, FREE, track_lots=True)
    assert len(res.episodes) == 1
    ep = res.episodes.iloc[0]
    assert ep["contrib"] == pytest.approx(0.5)
    gains = {y: v for y, v in res.realized.items() if v["st"] or v["lt"]}
    (y, g), = gains.items()
    assert g["lt"] == pytest.approx(0.5) and g["st"] == 0  # held 450 days -> long term


def test_short_term_gain_and_after_tax():
    c = [100.0] * 3 + [150.0] * 40
    p = make_panel({"X": c})
    res = simulate(targets_for(p, {0: {"X": 1.0}, 30: {"X": 0.0}}), p, FREE, track_lots=True)
    g = [v for v in res.realized.values() if v["st"]][0]
    assert g["st"] == pytest.approx(0.5)
    at = after_tax(res, st_rate=0.35, lt_rate=0.20)
    assert at["final_after_tax"] == pytest.approx(1.5 - 0.5 * 0.35)


def test_tax_netting_rules():
    assert _net_tax(100, 50, 0.4, 0.2) == (pytest.approx(50.0), 0.0, 0.0)
    assert _net_tax(-30, 50, 0.4, 0.2) == (pytest.approx(4.0), 0.0, 0.0)   # ST loss offsets LT gain
    assert _net_tax(-80, 50, 0.4, 0.2) == (0.0, -30, 0.0)                   # remaining ST loss carries
    assert _net_tax(50, -80, 0.4, 0.2) == (0.0, 0.0, -30)
    assert _net_tax(-1, -2, 0.4, 0.2) == (0.0, -1, -2)
