"""Single source of truth for markets, thresholds, and fusion weights.

Numeric values marked PROVISIONAL are literature-conventional starting points,
not measured optima; the backtest/validation layer exists to confirm or move
them. Change them here and nowhere else.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

TORCH_DEVICE = os.environ.get("STOCKINTEL_TORCH_DEVICE", "cpu")    # "cuda" inside the ZeroGPU model Space
# Trading days ahead that the volatility band and every forecast fan cover, so one chart's
# bands can be read against each other. Also one of the horizons the models are scored at.
FORECAST_DAYS = 20
FAN_LEVELS = (0.1, 0.25, 0.5, 0.75, 0.9)          # the quantiles every chart fan draws


@dataclass(frozen=True)
class Market:
    code: str
    name: str
    currency: str
    timezone: str
    suffix: str            # provider ticker suffix, e.g. ".NS"
    benchmark: str         # provider symbol of the default benchmark index
    risk_free_rate: float  # annualized; PROVISIONAL, update from current T-bill/G-sec


MARKETS: Dict[str, Market] = {
    "NSE": Market("NSE", "National Stock Exchange of India", "INR", "Asia/Kolkata",
                  ".NS", "^NSEI", risk_free_rate=0.068),
    "BSE": Market("BSE", "Bombay Stock Exchange", "INR", "Asia/Kolkata",
                  ".BO", "^BSESN", risk_free_rate=0.068),
    "US": Market("US", "United States", "USD", "America/New_York",
                 "", "^GSPC", risk_free_rate=0.042),
}

DEFAULT_MARKET = "NSE"


def resolve_symbol(symbol: str, market: Optional[str] = None) -> str:
    """Map a user symbol to the provider symbol for its market.

    'RELIANCE' + NSE -> 'RELIANCE.NS'; symbols already carrying a suffix or
    index carets pass through unchanged.
    """
    mkt = MARKETS[market or DEFAULT_MARKET]
    s = symbol.strip().upper()
    if s.startswith("^") or "." in s:
        return s
    return s + mkt.suffix


def canonical_symbol(symbol: str, market: Optional[str] = None) -> str:
    """One spelling per holding: the market's default suffix is dropped
    ('TCS.NS' -> 'TCS' on NSE); other suffixes are kept ('RELIANCE.BO')."""
    s = symbol.strip().upper()
    suffix = MARKETS[market or DEFAULT_MARKET].suffix
    return s[:-len(suffix)] if suffix and s.endswith(suffix) else s


@dataclass(frozen=True)
class FusionWeights:
    """Relative domain weights for evidence fusion. PROVISIONAL: equal-ish
    weights with technicals/fundamentals slightly above news; to be replaced
    by weights validated in backtests (scripts in backtest/)."""
    technical: float = 0.20
    candlestick: float = 0.08
    pattern: float = 0.10
    fundamental: float = 0.22
    news_sentiment: float = 0.15
    forecast: float = 0.10
    risk: float = 0.15
    historical: float = 0.05
    regime: float = 0.05

    def as_dict(self) -> Dict[str, float]:
        return {
            "technical": self.technical,
            "candlestick": self.candlestick,
            "pattern": self.pattern,
            "fundamental": self.fundamental,
            "news_sentiment": self.news_sentiment,
            "forecast": self.forecast,
            "risk": self.risk,
            "historical": self.historical,
            "regime": self.regime,
        }


@dataclass(frozen=True)
class Thresholds:
    """Analytical cut-offs. All PROVISIONAL unless noted; classic textbook
    values kept so outputs are comparable with standard charting tools."""
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    doji_body_ratio: float = 0.1        # body <= 10% of range
    long_wick_ratio: float = 2.0        # wick >= 2x body
    volume_spike_mult: float = 2.0      # vs 20d average
    price_move_alert_pct: float = 5.0
    pivot_order: int = 5                # bars each side for a local extremum
    sr_cluster_pct: float = 1.5         # pivots within 1.5% cluster into a zone
    pattern_tolerance_pct: float = 3.0  # peak/trough equality tolerance
    high_vol_percentile: float = 80.0
    low_vol_percentile: float = 20.0
    stale_days: int = 7                 # data older than this is flagged stale
    conflict_dispersion: float = 0.45   # weighted score std above this = conflict
    high_risk_score: float = -0.4       # risk domain score below this caps label


WEIGHTS = FusionWeights()
THRESHOLDS = Thresholds()


@dataclass(frozen=True)
class RiskProfile:
    """Portfolio guard-rails per risk preference. PROVISIONAL: common retail
    diversification heuristics, not optimized values."""
    name: str
    max_position_weight: float
    max_sector_weight: float
    max_holding_vol: float        # annualized volatility ceiling for a candidate
    min_median_traded_value: float  # liquidity floor, in the market's currency per day


RISK_PROFILES: Dict[str, RiskProfile] = {
    "conservative": RiskProfile("conservative", 0.15, 0.30, 0.28, 2.0e8),
    "moderate": RiskProfile("moderate", 0.20, 0.35, 0.40, 1.0e8),
    "aggressive": RiskProfile("aggressive", 0.30, 0.50, 0.65, 3.0e7),
}

# Backtest execution assumptions (India cash equities). PROVISIONAL: broker
# dependent; measured round-trip cost for discount brokers is near this.
TRANSACTION_COST_BPS = 12.0
SLIPPAGE_BPS = 5.0

ANNUALIZATION_DAYS = 252

FORECAST_HORIZONS = (1, 5, 10, 20)

@dataclass(frozen=True)
class DecisionPolicy:
    """Maps fused evidence to a decision-support label. PROVISIONAL bands:
    chosen so a label needs agreement from several domains, not one loud one;
    `stockintel validate-fusion` replays the price-derived part to test them."""
    strong: float = 0.35         # |score| at or above = strong view
    mild: float = 0.15           # |score| below = no view (HOLD)
    min_confidence: float = 0.45  # BUY/SELL need at least this confidence
    min_domains: int = 3          # fewer available domains = INSUFFICIENT_EVIDENCE


DECISION = DecisionPolicy()

# Decision-support vocabulary: research classifications, never orders.
# SELL applies only to a held position; the same evidence on a stock you do
# not own reads AVOID.
DECISION_LABELS = ("BUY", "WATCH", "HOLD", "SELL", "AVOID", "INSUFFICIENT_EVIDENCE")
