"""OHLCV data downloader using eodhd.com REST API."""

import requests
import pandas as pd
import io
import time
import logging
from typing import Optional

API_TOKEN = "679cc47dea6fa7.23803995"
BASE_URL = "https://eodhd.com/api/eod"

logger = logging.getLogger(__name__)


def fetch_ohlcv(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    retries: int = 3,
    retry_delay: float = 2.0,
) -> pd.DataFrame:
    """
    Download daily OHLCV data for a ticker from eodhd.com.

    Args:
        ticker: Ticker symbol in format TICKER.EXCHANGE (e.g. 'AAPL.US', '3SIL.LSE')
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
        retries: Number of retry attempts on failure
        retry_delay: Seconds between retries

    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Adjusted_close, Volume
        Indexed by Date (datetime).
    """
    params = {
        "api_token": API_TOKEN,
        "fmt": "csv",
    }
    if start_date:
        params["from"] = start_date
    if end_date:
        params["to"] = end_date

    url = f"{BASE_URL}/{ticker}"

    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text), parse_dates=["Date"])
            df = df.set_index("Date").sort_index()
            df.columns = [c.strip() for c in df.columns]
            logger.info(f"Downloaded {len(df)} rows for {ticker}")
            return df
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt+1}/{retries} failed for {ticker}: {e}")
            if attempt < retries - 1:
                time.sleep(retry_delay * (attempt + 1))
    raise RuntimeError(f"Failed to download data for {ticker} after {retries} attempts")


def load_universe(path: str = "universe.txt") -> list[str]:
    """Load ticker list from universe file, ignoring comments and blank lines."""
    tickers = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                tickers.append(line)
    return tickers
