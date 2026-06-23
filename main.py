"""
Quantitative ranking pipeline.

Usage:
    python main.py [--universe universe.txt] [--start YYYY-MM-DD] [--out output.csv]
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

from downloader import fetch_ohlcv, load_universe
from indicators import compute_all_indicators, latest_snapshot
from ranking import build_ranking

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SPY_TICKER = "SPY.US"


def parse_args():
    p = argparse.ArgumentParser(description="Quantitative ticker ranking pipeline")
    p.add_argument("--universe", default="universe.txt", help="Path to universe file")
    p.add_argument("--start", default="2022-01-01", help="Start date YYYY-MM-DD")
    p.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    p.add_argument("--out", default="ranking_output.csv", help="Output CSV path")
    p.add_argument("--detail-dir", default="details", help="Dir for per-ticker detail CSVs")
    return p.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------ #
    # 1. Load universe
    # ------------------------------------------------------------------ #
    universe_path = Path(args.universe)
    if not universe_path.exists():
        logger.error(f"Universe file not found: {universe_path}")
        sys.exit(1)

    tickers = load_universe(str(universe_path))
    logger.info(f"Universe: {len(tickers)} tickers")

    # ------------------------------------------------------------------ #
    # 2. Download SPY (benchmark for correlation)
    # ------------------------------------------------------------------ #
    logger.info("Downloading SPY benchmark data...")
    try:
        spy_df = fetch_ohlcv(SPY_TICKER, start_date=args.start, end_date=args.end)
        spy_close = (
            spy_df["Adjusted_close"]
            if "Adjusted_close" in spy_df.columns
            else spy_df["Close"]
        )
    except Exception as e:
        logger.error(f"Cannot download SPY: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 3. Download + compute indicators for each ticker
    # ------------------------------------------------------------------ #
    detail_dir = Path(args.detail_dir)
    detail_dir.mkdir(exist_ok=True)

    snapshots: Dict[str, dict] = {}
    histories: Dict[str, pd.Series] = {}
    failed = []

    for ticker in tickers:
        logger.info(f"Processing {ticker}...")
        try:
            ohlcv = fetch_ohlcv(ticker, start_date=args.start, end_date=args.end)
            indicators_df = compute_all_indicators(ohlcv, spy_close)
            snap = latest_snapshot(indicators_df)
            snapshots[ticker] = snap

            close_col = (
                ohlcv["Adjusted_close"]
                if "Adjusted_close" in ohlcv.columns
                else ohlcv["Close"]
            )
            histories[ticker] = close_col

            # Save per-ticker detail
            out_path = detail_dir / f"{ticker.replace('.', '_')}_indicators.csv"
            indicators_df.to_csv(out_path)
            logger.info(f"  -> Saved detail to {out_path}")

        except Exception as e:
            logger.warning(f"  -> FAILED for {ticker}: {e}")
            failed.append(ticker)

    if not snapshots:
        logger.error("No data could be downloaded. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 4. Build ranking
    # ------------------------------------------------------------------ #
    logger.info("Building ranking...")
    ranking = build_ranking(snapshots, histories)

    # ------------------------------------------------------------------ #
    # 5. Save & print
    # ------------------------------------------------------------------ #
    ranking.to_csv(args.out)
    logger.info(f"Ranking saved to {args.out}")

    print("\n" + "=" * 80)
    print(f"TICKER RANKING  (as of latest available data)")
    print("=" * 80)

    display_cols = [
        "rank", "composite", "trend_score", "ema_momentum",
        "rsi14_score", "rsi2_oversold", "kc_position", "ftd_bonus",
        "Close", "RSI_14", "RSI_2", "CORR_SPY_21", "FTD",
    ]
    display = ranking.reset_index()[
        [c for c in display_cols if c in ranking.reset_index().columns]
    ]
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", "{:.2f}".format)
    print(display.to_string(index=False))

    if failed:
        print(f"\nFailed tickers ({len(failed)}): {', '.join(failed)}")

    print("=" * 80)


if __name__ == "__main__":
    main()
