"""Technical indicators: EMA, RSI, Correlation with SPY, Keltner Bands, FTD."""

import numpy as np
import pandas as pd
from typing import Dict, Any


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

def compute_emas(close: pd.Series, periods: list[int] = [5, 10, 15, 21, 50, 112, 200]) -> pd.DataFrame:
    """Return a DataFrame of EMAs for each period."""
    result = {}
    for p in periods:
        result[f"EMA_{p}"] = close.ewm(span=p, adjust=False).mean()
    return pd.DataFrame(result, index=close.index)


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename(f"RSI_{period}")


def compute_rsis(close: pd.Series, periods: list[int] = [2, 14]) -> pd.DataFrame:
    return pd.concat([compute_rsi(close, p) for p in periods], axis=1)


# ---------------------------------------------------------------------------
# Correlation with SPY
# ---------------------------------------------------------------------------

def compute_correlations(
    close: pd.Series,
    spy_close: pd.Series,
    periods: list[int] = [5, 10, 15, 21],
) -> pd.DataFrame:
    """Rolling Pearson correlation of returns against SPY returns."""
    ret = close.pct_change()
    spy_ret = spy_close.pct_change()
    combined = pd.concat([ret, spy_ret], axis=1).dropna(how="all")
    ret_aligned = combined.iloc[:, 0]
    spy_aligned = combined.iloc[:, 1]

    result = {}
    for p in periods:
        result[f"CORR_SPY_{p}"] = ret_aligned.rolling(p).corr(spy_aligned)
    return pd.DataFrame(result, index=close.index)


# ---------------------------------------------------------------------------
# Keltner Channels
# ---------------------------------------------------------------------------

def compute_keltner(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 21,
    atr_mult: float = 2.0,
) -> pd.DataFrame:
    """
    Keltner Channels with EMA middle line and ATR-based bands.
    Returns middle, upper, lower bands and distance of close from each band.
    """
    ema = close.ewm(span=period, adjust=False).mean()

    # True Range
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()

    upper = ema + atr_mult * atr
    lower = ema - atr_mult * atr

    dist_upper = close - upper   # negative => below upper band
    dist_lower = close - lower   # positive => above lower band

    return pd.DataFrame(
        {
            f"KC_MID_{period}": ema,
            f"KC_UPPER_{period}": upper,
            f"KC_LOWER_{period}": lower,
            f"KC_DIST_UPPER_{period}": dist_upper,
            f"KC_DIST_LOWER_{period}": dist_lower,
        },
        index=close.index,
    )


# ---------------------------------------------------------------------------
# Follow-Through Day (FTD)
# ---------------------------------------------------------------------------

def compute_ftd(
    close: pd.Series,
    volume: pd.Series,
    min_day: int = 4,
    min_gain_pct: float = 1.7,
) -> pd.Series:
    """
    Follow-Through Day indicator (adapted for individual stocks from O'Neil methodology).

    A FTD is signalled when:
      1. A rally attempt starts (first up close after a prior down-trend).
      2. On day `min_day` or later of the rally attempt, close gains >= `min_gain_pct`%
         on volume higher than the previous session.

    Returns a boolean Series: True on FTD days.

    Parameters:
        min_day: Minimum day of rally attempt before a FTD can occur (default 4).
        min_gain_pct: Minimum single-day gain % for a FTD (default 1.7%).
    """
    returns = close.pct_change() * 100
    vol_higher = volume > volume.shift(1)

    # Track rally day counter
    rally_day = pd.Series(0, index=close.index)
    in_rally = False
    count = 0

    for i in range(1, len(close)):
        if returns.iloc[i] > 0:
            if not in_rally:
                in_rally = True
                count = 1
            else:
                count += 1
        else:
            in_rally = False
            count = 0
        rally_day.iloc[i] = count

    strong_day = returns >= min_gain_pct
    ftd = (rally_day >= min_day) & strong_day & vol_higher

    return ftd.rename("FTD")


# ---------------------------------------------------------------------------
# All-in-one per ticker
# ---------------------------------------------------------------------------

def compute_all_indicators(
    ohlcv: pd.DataFrame,
    spy_close: pd.Series,
) -> pd.DataFrame:
    """
    Compute all indicators for a single ticker's OHLCV DataFrame.
    Returns a DataFrame with one row per date containing all indicator values.

    Expected ohlcv columns: Open, High, Low, Close, Volume (or Adjusted_close).
    """
    close = ohlcv["Adjusted_close"] if "Adjusted_close" in ohlcv.columns else ohlcv["Close"]
    high = ohlcv["High"]
    low = ohlcv["Low"]
    volume = ohlcv["Volume"]

    emas = compute_emas(close)
    rsis = compute_rsis(close)
    corrs = compute_correlations(close, spy_close)
    kc = compute_keltner(high, low, close)
    ftd = compute_ftd(close, volume)

    return pd.concat([ohlcv[["Open", "High", "Low", "Close"]], emas, rsis, corrs, kc, ftd.to_frame()], axis=1)


# ---------------------------------------------------------------------------
# Snapshot: latest values only
# ---------------------------------------------------------------------------

def latest_snapshot(indicators_df: pd.DataFrame) -> Dict[str, Any]:
    """Return the most recent row as a dict."""
    row = indicators_df.iloc[-1]
    return row.to_dict()
