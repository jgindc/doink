import numpy as np
import pandas as pd
import pytest

from swingbot.data.panel import Panel


def make_panel(closes: dict[str, list[float]] | pd.DataFrame, opens: dict | pd.DataFrame | None = None,
               start: str = "2020-01-01") -> Panel:
    close = pd.DataFrame(closes)
    close.index = pd.date_range(start, periods=len(close), freq="1D", tz="UTC")
    if opens is None:
        open_ = close.shift(1).fillna(close)
    else:
        open_ = pd.DataFrame(opens, index=close.index)
    high = np.maximum(open_, close)
    low = np.minimum(open_, close)
    vol = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    return Panel(open_, high, low, close, vol, pd.DataFrame(False, index=close.index, columns=close.columns))


@pytest.fixture
def random_panel():
    rng = np.random.default_rng(0)
    n = 900
    data = {}
    for k, a in enumerate(["AAA", "BBB", "CCC", "DDD"]):
        r = rng.normal(0.0005, 0.03, n)
        px = 100 * np.exp(np.cumsum(r))
        if k == 3:  # late listing
            px[:300] = np.nan
        data[a] = px
    return make_panel(data)
