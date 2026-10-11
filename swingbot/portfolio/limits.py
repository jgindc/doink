"""Hard risk limits, enforced in code OUTSIDE any strategy or AI decision path.

Every order batch for paper/live trading must pass `check_orders` and every run must pass `check_account`.
Nothing upstream (rule sleeves or the AI-decides lane) can change these limits at runtime; they come only
from the YAML config, and live mode additionally requires an explicit flag plus human confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class TradingHalted(RuntimeError):
    pass


@dataclass(frozen=True)
class Limits:
    max_daily_loss: float
    max_drawdown: float
    max_position_weight: float
    max_order_notional_frac: float
    allowed_symbols: frozenset = field(default_factory=frozenset)

    @classmethod
    def from_config(cls, cfg: dict) -> "Limits":
        lim = cfg["limits"]
        return cls(lim["max_daily_loss"], lim["max_drawdown"], lim["max_position_weight"],
                   lim["max_order_notional_frac"], frozenset(lim["allowed_symbols"]))


@dataclass
class Order:
    symbol: str
    side: str          # "buy" | "sell"
    notional: float    # USD, > 0


def check_account(equity: float, start_of_day_equity: float, peak_equity: float, limits: Limits) -> None:
    """Raise TradingHalted if the daily-loss or drawdown limit is breached. The caller must alert the owner."""
    if start_of_day_equity > 0 and equity / start_of_day_equity - 1 <= -limits.max_daily_loss:
        raise TradingHalted(f"daily loss limit hit: {equity / start_of_day_equity - 1:.2%}")
    if peak_equity > 0 and equity / peak_equity - 1 <= -limits.max_drawdown:
        raise TradingHalted(f"max drawdown limit hit: {equity / peak_equity - 1:.2%}")


def check_targets(targets: dict[str, float], limits: Limits) -> dict[str, float]:
    """Clip target weights to the limits. Unknown symbols are dropped; never adds leverage or shorts."""
    out = {}
    for s, w in targets.items():
        if s not in limits.allowed_symbols:
            continue
        out[s] = min(max(float(w), 0.0), limits.max_position_weight)
    total = sum(out.values())
    if total > 1.0:
        out = {s: w / total for s, w in out.items()}
    return out


def check_orders(orders: list[Order], equity: float, limits: Limits) -> tuple[list[Order], list[str]]:
    """Return (approved, rejections). Orders above the per-order cap are cut down, not dropped."""
    approved, rejected = [], []
    cap = limits.max_order_notional_frac * equity
    for o in orders:
        if o.symbol not in limits.allowed_symbols:
            rejected.append(f"{o.symbol}: symbol not allowed")
            continue
        if o.side not in ("buy", "sell") or o.notional <= 0:
            rejected.append(f"{o.symbol}: malformed order {o}")
            continue
        if o.side == "buy" and o.notional > cap:
            rejected.append(f"{o.symbol}: buy {o.notional:.2f} cut to cap {cap:.2f}")
            o = Order(o.symbol, o.side, cap)
        approved.append(o)
    return approved, rejected
