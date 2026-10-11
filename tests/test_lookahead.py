"""Lookahead-bias tests: changing or deleting FUTURE data must not change any past decision."""

import numpy as np
import pandas as pd
import pytest

from swingbot.data.panel import Panel
from swingbot.portfolio.combine import build_book
from swingbot.strategies import benchmarks as bm
from swingbot.strategies.sleeves import (DipParams, RotationParams, TrendParams, dip_sleeve, rotation_sleeve,
                                         trend_sleeve)

PPY, MINH = 365, 250

SLEEVES = {
    "trend": lambda p: trend_sleeve(p, TrendParams(), PPY, MINH),
    "trend_sma": lambda p: trend_sleeve(p, TrendParams(signal="above_sma"), PPY, MINH),
    "rotation": lambda p: rotation_sleeve(p, RotationParams(), PPY, MINH),
    "dip": lambda p: dip_sleeve(p, DipParams(entry=30), PPY, MINH),
    "bh": lambda p: bm.buy_and_hold(p, {"AAA": 0.5, "BBB": 0.5}),
    "sma200": lambda p: bm.sma_trend(p, "AAA"),
}


def _scramble_future(panel: Panel, cut: pd.Timestamp, seed: int = 1) -> Panel:
    """Replace every bar after `cut` with wild values (x0.2 or x5), so any peek at the future flips decisions."""
    rng = np.random.default_rng(seed)
    out = {}
    for f in ("open", "high", "low", "close", "volume"):
        df = getattr(panel, f).copy()
        fut = df.index > cut
        df.loc[fut] = df.loc[fut] * rng.choice([0.2, 5.0], size=df.loc[fut].shape)
        out[f] = df
    return Panel(**out, filled=panel.filled)


CUTS = list(range(300, 880, 20))  # many cut points: a one-bar peek is checked at each of them


def assert_causal(fn, panel):
    full = fn(panel)
    for k, i in enumerate(CUTS):
        cut = panel.index[i]
        pd.testing.assert_frame_equal(full.loc[:cut], fn(panel.truncate(cut)), check_freq=False)
        pd.testing.assert_frame_equal(full.loc[:cut], fn(_scramble_future(panel, cut, seed=k)).loc[:cut],
                                      check_freq=False)


def test_the_test_catches_a_planted_one_day_peek(random_panel):
    leaky = lambda p: (p.close.shift(-1) > p.close).astype(float) * 0.25  # noqa: E731
    with pytest.raises(AssertionError):
        assert_causal(leaky, random_panel)


@pytest.mark.parametrize("name", list(SLEEVES))
def test_truncation_and_future_scramble_do_not_change_past(random_panel, name):
    fn = SLEEVES[name]
    full = fn(random_panel)
    assert (full.fillna(0) >= -1e-12).all().all()
    assert (full.fillna(0).sum(axis=1) <= 1 + 1e-9).all()
    assert_causal(fn, random_panel)


def test_book_is_causal(random_panel):
    cfg = {"annualization_days": PPY,
           "portfolio": {"sleeve_budget": {"trend": 0.34, "rotation": 0.33, "dip": 0.33},
                         "vol_target_annual": 0.3, "vol_lookback_days": 60,
                         "max_weight": {"default": 0.5}}}

    def book(p):
        sl = {k: SLEEVES[k](p) for k in ("trend", "rotation", "dip")}
        return build_book(sl, cfg, p.close)

    assert (book(random_panel).sum(axis=1) <= 1 + 1e-9).all()
    assert_causal(book, random_panel)
