import numpy as np
import pandas as pd
import pytest

from swingbot.data import panel as panel_mod
from swingbot.portfolio.limits import Limits, Order, TradingHalted, check_account, check_orders, check_targets
from swingbot.research.overfit import deflated_sharpe, expected_max_sharpe, pbo_cscv, probabilistic_sharpe

LIM = Limits(0.05, 0.25, 0.5, 0.3, frozenset({"BTC", "ETH"}))


def test_limits_clip_targets_and_orders():
    t = check_targets({"BTC": 0.9, "ETH": 0.7, "DOGE": 0.5, "XRP": -1}, LIM)
    assert "DOGE" not in t and t["BTC"] == pytest.approx(0.5) and t["ETH"] == pytest.approx(0.5)
    ok, rej = check_orders([Order("BTC", "buy", 5000), Order("DOGE", "buy", 10)], equity=10000, limits=LIM)
    assert ok[0].notional == pytest.approx(3000) and len(ok) == 1 and len(rej) == 2


def test_account_halts():
    check_account(9600, 10000, 10000, LIM)
    with pytest.raises(TradingHalted):
        check_account(9400, 10000, 10000, LIM)
    with pytest.raises(TradingHalted):
        check_account(7400, 7500, 10000, LIM)


def test_holdout_requires_reason_and_is_logged(tmp_path, monkeypatch):
    monkeypatch.setattr(panel_mod, "HOLDOUT_LOG", tmp_path / "log.md")
    cfg = {"data": {"universe": ["BTC"], "source": "coinbase"}, "splits": {"holdout_start": "2024-10-01"}}
    p = panel_mod.load_crypto_panel(cfg)
    assert p.index[-1] < pd.Timestamp("2024-10-01", tz="UTC")
    with pytest.raises(PermissionError):
        panel_mod.load_crypto_panel(cfg, purpose="holdout")
    p2 = panel_mod.load_crypto_panel(cfg, purpose="holdout", holdout_reason="unit test")
    assert p2.index[-1] >= pd.Timestamp("2024-10-01", tz="UTC")
    assert "unit test" in (tmp_path / "log.md").read_text()


def test_expected_max_sharpe_grows_with_trials():
    assert expected_max_sharpe(1, 0.01) == 0
    assert expected_max_sharpe(10, 0.01) < expected_max_sharpe(1000, 0.01)


def test_dsr_penalizes_many_trials():
    rng = np.random.default_rng(3)
    r = rng.normal(0.001, 0.02, 1500)
    one = deflated_sharpe(r, 1, var_sr=0.0)["dsr"]
    many = deflated_sharpe(r, 1000, var_sr=0.03 ** 2)["dsr"]
    assert one == pytest.approx(probabilistic_sharpe(r))
    assert many < one


def test_pbo_noise_vs_signal():
    rng = np.random.default_rng(5)
    noise = pd.DataFrame(rng.normal(0, 0.01, (1600, 12)))
    assert 0.25 < pbo_cscv(noise)["pbo"] < 0.75
    sig = noise.copy()
    sig[0] += 0.002  # one configuration has a real, persistent edge
    assert pbo_cscv(sig)["pbo"] < 0.05
