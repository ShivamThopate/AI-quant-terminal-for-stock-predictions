"""
Price Data Fetcher
===================
Fetches OHLCV (Open, High, Low, Close, Volume) price data for NSE-listed equities.

Primary source : yfinance
Fallback source: jugaad-data (NSE historical)

Also provides benchmark (Nifty 50) price data for Relative Strength calculations.
"""

import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import (
    BENCHMARK_TICKER,
    DATA_CACHE_DIR,
    NSE_SUFFIX,
    PRICE_SOURCES,
)

logger = logging.getLogger(__name__)


class PriceFetcher:
    """
    Fetches historical OHLCV data with automatic fallback and local file caching.

    Usage::

        fetcher = PriceFetcher()
        df = fetcher.fetch_ohlcv("RELIANCE", period="1y", interval="1d")
        bench = fetcher.fetch_benchmark(period="1y")
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self._cache_dir = cache_dir or DATA_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a single NSE ticker.

        Parameters
        ----------
        ticker : str
            NSE symbol, with or without ``.NS`` suffix.
        period : str
            yfinance-style period string (e.g. ``"1y"``, ``"90d"``).
        interval : str
            Bar interval (default ``"1d"``).

        Returns
        -------
        pd.DataFrame
            Columns: ``[Open, High, Low, Close, Volume]`` indexed by Date.
        """
        ticker = self._normalise_ticker(ticker)

        # Check cache first
        cached = self._load_cache(ticker, period, interval)
        if cached is not None:
            logger.info("Cache hit for %s (%s / %s)", ticker, period, interval)
            return cached

        # Try each source in priority order
        for source in PRICE_SOURCES:
            try:
                if source == "yfinance":
                    df = self._fetch_yfinance(ticker, period, interval)
                elif source == "jugaad-data":
                    df = self._fetch_jugaad(ticker, period)
                else:
                    continue

                if df is not None and not df.empty:
                    self._save_cache(df, ticker, period, interval)
                    logger.info("Fetched %s from %s (%d rows)", ticker, source, len(df))
                    return df

            except Exception as exc:
                logger.warning("Source '%s' failed for %s: %s", source, ticker, exc)
                continue

        logger.error("All data sources failed for %s", ticker)
        return pd.DataFrame()

    def fetch_benchmark(
        self, period: str = "1y", interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch OHLCV for the benchmark index (Nifty 50)."""
        return self.fetch_ohlcv(BENCHMARK_TICKER, period, interval)

    def fetch_multiple(
        self,
        tickers: list,
        period: str = "1y",
        interval: str = "1d",
    ) -> dict:
        """
        Fetch OHLCV for multiple tickers.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping of ticker → OHLCV DataFrame.
        """
        results = {}
        for ticker in tickers:
            df = self.fetch_ohlcv(ticker, period, interval)
            if not df.empty:
                results[self._normalise_ticker(ticker)] = df
        return results

    # ------------------------------------------------------------------
    # Source implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _fetch_yfinance(ticker: str, period: str, interval: str) -> pd.DataFrame:
        """Fetch data via the yfinance library."""
        import yfinance as yf

        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)

        if df.empty:
            return pd.DataFrame()

        # Normalise column names
        df = df.rename(columns={
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume",
        })

        # Keep only the columns we need
        cols = ["Open", "High", "Low", "Close", "Volume"]
        available = [c for c in cols if c in df.columns]
        df = df[available].copy()

        # Ensure index is DatetimeIndex named 'Date'
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"

        return df

    @staticmethod
    def _fetch_jugaad(ticker: str, period: str) -> pd.DataFrame:
        """
        Fetch data via jugaad-data (NSE historical data).
        jugaad-data uses start/end dates rather than period strings.
        """
        from jugaad_data.nse import stock_df

        # Convert yfinance-style period to start/end dates
        end_date = datetime.today()
        period_days = {
            "7d": 7, "30d": 30, "60d": 60, "90d": 90,
            "6mo": 180, "1y": 365, "2y": 730, "5y": 1825,
        }
        days = period_days.get(period, 365)
        start_date = end_date - timedelta(days=days)

        # jugaad-data expects the raw NSE symbol (no .NS)
        symbol = ticker.replace(".NS", "").replace(".ns", "")

        df = stock_df(
            symbol=symbol,
            from_date=start_date.date(),
            to_date=end_date.date(),
            series="EQ",
        )

        if df is None or df.empty:
            return pd.DataFrame()

        # Normalise columns from jugaad-data format
        rename_map = {
            "DATE": "Date",
            "OPEN": "Open",
            "HIGH": "High",
            "LOW": "Low",
            "CLOSE": "Close",
            "TOTAL_TRADE_QUANTITY": "Volume",
            "NO OF TRADES": "Volume",  # alternate column name
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        # Fallback: try lowercase column names
        if "Open" not in df.columns:
            rename_lower = {c: c.capitalize() for c in df.columns}
            df = df.rename(columns=rename_lower)

        cols = ["Open", "High", "Low", "Close", "Volume"]
        available = [c for c in cols if c in df.columns]
        df = df[available].copy()

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
        else:
            df.index = pd.to_datetime(df.index)
            df.index.name = "Date"

        df = df.sort_index()
        return df

    # ------------------------------------------------------------------
    # Caching helpers
    # ------------------------------------------------------------------

    def _cache_key(self, ticker: str, period: str, interval: str) -> str:
        """Generate a unique cache filename."""
        raw = f"{ticker}_{period}_{interval}"
        hsh = hashlib.md5(raw.encode()).hexdigest()[:8]
        return f"{ticker.replace('^', 'IDX_')}_{period}_{interval}_{hsh}.parquet"

    def _load_cache(self, ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """Load data from local parquet cache if fresh (< 1 day old)."""
        path = self._cache_dir / self._cache_key(ticker, period, interval)
        if path.exists():
            age_hours = (datetime.now().timestamp() - path.stat().st_mtime) / 3600
            if age_hours < 24:
                try:
                    return pd.read_parquet(path)
                except Exception:
                    pass
        return None

    def _save_cache(self, df: pd.DataFrame, ticker: str, period: str, interval: str):
        """Persist DataFrame to local parquet cache."""
        path = self._cache_dir / self._cache_key(ticker, period, interval)
        try:
            df.to_parquet(path)
        except Exception as exc:
            logger.warning("Failed to cache %s: %s", ticker, exc)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_ticker(ticker: str) -> str:
        """Ensure ticker has the .NS suffix (unless it's an index, commodity, or crypto)."""
        if ticker.startswith("^") or "=" in ticker or "-" in ticker:
            return ticker.upper()
        if not ticker.upper().endswith(".NS"):
            return f"{ticker.upper()}{NSE_SUFFIX}"
        return ticker.upper()
