"""
Ranking engine: scores and orders tickers based on computed indicators.

Scoring philosophy (higher = better):
  - Momentum / trend:     price vs EMA200 (are we above?), EMA slope
  - Short-term strength:  RSI_2 (contrarian: lower = more oversold = buy signal)
  - Medium-term momentum: RSI_14 (higher = stronger)
  - Correlation with SPY: lower 21d corr => more independent alpha
  - Keltner position:     closer to lower band = potential mean-reversion entry
  - FTD:                  recent FTD = positive signal
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


def _safe(val, default=np.nan):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return val


def score_ticker(snapshot: Dict[str, Any], close_history: pd.Series) -> Dict[str, float]:
    """
    Compute individual component scores (each 0-100) and a weighted composite.

    Args:
        snapshot: Latest indicator values dict from indicators.latest_snapshot().
        close_history: Full close price Series for slope computation.
    """
    scores = {}

    # --- 1. Trend score: % of EMAs that close is above (200 period horizon) ---
    ema_periods = [5, 10, 15, 21, 50, 112, 200]
    last_close = _safe(snapshot.get("Close"))
    above = sum(
        1 for p in ema_periods
        if not np.isnan(_safe(snapshot.get(f"EMA_{p}", np.nan)))
        and last_close > snapshot[f"EMA_{p}"]
    )
    scores["trend_score"] = (above / len(ema_periods)) * 100

    # --- 2. EMA momentum: slope of EMA_21 over last 5 days (normalised) ---
    ema21_now = _safe(snapshot.get("EMA_21", np.nan))
    if not np.isnan(ema21_now) and len(close_history) >= 6:
        ema21_prev = close_history.ewm(span=21, adjust=False).mean().iloc[-6]
        slope_pct = (ema21_now - ema21_prev) / ema21_prev * 100 if ema21_prev else 0
        scores["ema_momentum"] = min(max(slope_pct * 10 + 50, 0), 100)
    else:
        scores["ema_momentum"] = 50

    # --- 3. RSI_14 score: higher RSI_14 = stronger (momentum regime) ---
    rsi14 = _safe(snapshot.get("RSI_14", 50))
    scores["rsi14_score"] = float(rsi14)

    # --- 4. RSI_2 contrarian: very low RSI_2 => oversold => potential bounce ---
    rsi2 = _safe(snapshot.get("RSI_2", 50))
    scores["rsi2_oversold"] = 100 - float(rsi2)  # lower RSI_2 → higher score

    # --- 5. SPY correlation: lower 21d corr = more independent ---
    corr21 = _safe(snapshot.get("CORR_SPY_21", 0.5))
    scores["independence"] = (1 - float(corr21)) * 50  # 0-100

    # --- 6. Keltner position: distance from lower band (closer = oversold) ---
    kc_dist_lower = _safe(snapshot.get("KC_DIST_LOWER_21", 0))
    kc_mid = _safe(snapshot.get("KC_MID_21", last_close or 1))
    if kc_mid and not np.isnan(kc_mid):
        # Normalise by mid; more negative => further below lower => more oversold
        norm_dist = kc_dist_lower / kc_mid * 100
        scores["kc_position"] = min(max(-norm_dist * 10 + 50, 0), 100)
    else:
        scores["kc_position"] = 50

    # --- 7. FTD bonus ---
    scores["ftd_bonus"] = 100 if snapshot.get("FTD", False) else 0

    # --- Weighted composite (weights sum to 1) ---
    weights = {
        "trend_score":    0.25,
        "ema_momentum":   0.20,
        "rsi14_score":    0.15,
        "rsi2_oversold":  0.10,
        "independence":   0.10,
        "kc_position":    0.15,
        "ftd_bonus":      0.05,
    }
    composite = sum(scores[k] * w for k, w in weights.items() if not np.isnan(scores[k]))
    scores["composite"] = composite

    return scores


def build_ranking(
    snapshots: Dict[str, Dict[str, Any]],
    histories: Dict[str, pd.Series],
) -> pd.DataFrame:
    """
    Build a ranked DataFrame for the full universe.

    Args:
        snapshots: {ticker: latest_snapshot_dict}
        histories: {ticker: close_price_series}

    Returns:
        DataFrame sorted by composite score descending, with rank column.
    """
    rows = []
    for ticker, snap in snapshots.items():
        hist = histories.get(ticker, pd.Series(dtype=float))
        try:
            s = score_ticker(snap, hist)
        except Exception:
            s = {"composite": np.nan}
        s["ticker"] = ticker
        # Attach key indicator values for display
        for col in ["Close", "EMA_21", "EMA_50", "EMA_200", "RSI_2", "RSI_14",
                    "CORR_SPY_21", "KC_DIST_UPPER_21", "KC_DIST_LOWER_21", "FTD"]:
            s[col] = snap.get(col, np.nan)
        rows.append(s)

    df = pd.DataFrame(rows).set_index("ticker")
    df = df.sort_values("composite", ascending=False)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df
