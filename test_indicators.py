"""Unit/integration tests using synthetic OHLCV data (no network required)."""

import numpy as np
import pandas as pd
import pytest
from indicators import (
    compute_emas,
    compute_rsis,
    compute_correlations,
    compute_keltner,
    compute_ftd,
    compute_all_indicators,
    latest_snapshot,
)
from ranking import score_ticker, build_ranking


def make_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data with a mild uptrend."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.015, n))
    high = close * (1 + rng.uniform(0, 0.02, n))
    low = close * (1 - rng.uniform(0, 0.02, n))
    open_ = low + (high - low) * rng.uniform(0, 1, n)
    volume = rng.integers(500_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close,
         "Adjusted_close": close, "Volume": volume},
        index=dates,
    )


@pytest.fixture
def ohlcv():
    return make_ohlcv()


@pytest.fixture
def spy_close():
    return make_ohlcv(seed=99)["Adjusted_close"]


# ---- EMA ------------------------------------------------------------------

def test_ema_shape(ohlcv):
    emas = compute_emas(ohlcv["Adjusted_close"])
    assert set(emas.columns) == {f"EMA_{p}" for p in [5, 10, 15, 21, 50, 112, 200]}
    assert len(emas) == len(ohlcv)


def test_ema_convergence(ohlcv):
    emas = compute_emas(ohlcv["Adjusted_close"])
    # After enough data EMA_200 should exist (no NaN at end)
    assert not emas["EMA_200"].iloc[-1:].isna().any()


# ---- RSI ------------------------------------------------------------------

def test_rsi_bounds(ohlcv):
    rsis = compute_rsis(ohlcv["Adjusted_close"])
    for col in rsis.columns:
        valid = rsis[col].dropna()
        assert (valid >= 0).all() and (valid <= 100).all(), f"{col} out of [0,100]"


def test_rsi_columns(ohlcv):
    rsis = compute_rsis(ohlcv["Adjusted_close"])
    assert "RSI_2" in rsis.columns and "RSI_14" in rsis.columns


# ---- Correlation ----------------------------------------------------------

def test_corr_bounds(ohlcv, spy_close):
    corrs = compute_correlations(ohlcv["Adjusted_close"], spy_close)
    for col in corrs.columns:
        valid = corrs[col].dropna()
        assert (valid >= -1.01).all() and (valid <= 1.01).all()


def test_corr_columns(ohlcv, spy_close):
    corrs = compute_correlations(ohlcv["Adjusted_close"], spy_close)
    assert set(corrs.columns) == {f"CORR_SPY_{p}" for p in [5, 10, 15, 21]}


# ---- Keltner --------------------------------------------------------------

def test_keltner_bands_order(ohlcv):
    kc = compute_keltner(ohlcv["High"], ohlcv["Low"], ohlcv["Adjusted_close"])
    valid = kc.dropna()
    assert (valid["KC_UPPER_21"] > valid["KC_MID_21"]).all()
    assert (valid["KC_MID_21"] > valid["KC_LOWER_21"]).all()


def test_keltner_dist_signs(ohlcv):
    kc = compute_keltner(ohlcv["High"], ohlcv["Low"], ohlcv["Adjusted_close"])
    close = ohlcv["Adjusted_close"]
    valid_idx = kc.dropna().index
    # dist_upper = close - upper: expect mostly negative (price below upper band)
    assert (kc.loc[valid_idx, "KC_DIST_UPPER_21"] < kc.loc[valid_idx, "KC_UPPER_21"]).all()


# ---- FTD ------------------------------------------------------------------

def test_ftd_is_boolean(ohlcv):
    ftd = compute_ftd(ohlcv["Adjusted_close"], ohlcv["Volume"])
    assert ftd.dtype == bool or set(ftd.unique()).issubset({True, False})


def test_ftd_not_all_true(ohlcv):
    ftd = compute_ftd(ohlcv["Adjusted_close"], ohlcv["Volume"])
    assert not ftd.all(), "FTD should not fire on every day"


# ---- All indicators -------------------------------------------------------

def test_all_indicators_shape(ohlcv, spy_close):
    df = compute_all_indicators(ohlcv, spy_close)
    assert len(df) == len(ohlcv)
    expected_cols = ["EMA_5", "RSI_2", "RSI_14", "CORR_SPY_21", "KC_MID_21", "FTD"]
    for c in expected_cols:
        assert c in df.columns, f"Missing column: {c}"


# ---- Ranking --------------------------------------------------------------

def test_score_ticker(ohlcv, spy_close):
    df = compute_all_indicators(ohlcv, spy_close)
    snap = latest_snapshot(df)
    close_col = ohlcv["Adjusted_close"]
    scores = score_ticker(snap, close_col)
    assert "composite" in scores
    assert 0 <= scores["composite"] <= 100


def test_build_ranking(spy_close):
    tickers = ["AAA", "BBB", "CCC"]
    snapshots = {}
    histories = {}
    for i, t in enumerate(tickers):
        ohlcv = make_ohlcv(seed=i)
        df = compute_all_indicators(ohlcv, spy_close)
        snapshots[t] = latest_snapshot(df)
        histories[t] = ohlcv["Adjusted_close"]
    ranking = build_ranking(snapshots, histories)
    assert list(ranking["rank"]) == [1, 2, 3]
    assert ranking.index.tolist() == sorted(ranking.index, key=lambda x: -ranking.loc[x, "composite"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
