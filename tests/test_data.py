"""
Integration tests for data fetcher modules.
These tests make real API calls — they require internet access.
Mark them with @pytest.mark.integration to skip in CI if needed.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.price_fetcher import PriceFetcher
from data.fundamental_fetcher import FundamentalFetcher, FundamentalData
from data.macro_fetcher import MacroFetcher, MacroSnapshot
from data.news_fetcher import NewsFetcher, Headline


# -----------------------------------------------------------------------
# Price Fetcher
# -----------------------------------------------------------------------

class TestPriceFetcher:

    @pytest.fixture
    def fetcher(self):
        return PriceFetcher()

    def test_fetch_single_ticker(self, fetcher):
        """Fetch OHLCV for RELIANCE and verify DataFrame shape."""
        df = fetcher.fetch_ohlcv("RELIANCE", period="30d")
        assert not df.empty, "DataFrame should not be empty"
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col in df.columns, f"Missing column: {col}"
        assert len(df) > 10, "Should have at least 10 trading days in 30d"

    def test_ticker_normalisation(self, fetcher):
        """Tickers without .NS should be auto-normalised."""
        t1 = fetcher._normalise_ticker("RELIANCE")
        t2 = fetcher._normalise_ticker("RELIANCE.NS")
        assert t1 == t2 == "RELIANCE.NS"

    def test_benchmark_fetch(self, fetcher):
        """Fetch Nifty 50 benchmark data."""
        df = fetcher.fetch_benchmark(period="30d")
        assert not df.empty
        assert "Close" in df.columns

    def test_fetch_multiple(self, fetcher):
        """Fetch data for multiple tickers."""
        results = fetcher.fetch_multiple(["TCS", "INFY"], period="30d")
        assert len(results) >= 1, "Should fetch at least 1 ticker"

    def test_index_ticker_no_suffix(self, fetcher):
        """Index tickers like ^NSEI should not get .NS appended."""
        t = fetcher._normalise_ticker("^NSEI")
        assert t == "^NSEI"


# -----------------------------------------------------------------------
# Fundamental Fetcher
# -----------------------------------------------------------------------

class TestFundamentalFetcher:

    @pytest.fixture
    def fetcher(self):
        return FundamentalFetcher()

    def test_fetch_single(self, fetcher):
        """Fetch fundamentals for TCS."""
        data = fetcher.fetch("TCS")
        assert isinstance(data, FundamentalData)
        assert data.ticker == "TCS.NS"
        # At least one ratio should be populated
        has_data = any([
            data.pe_ratio is not None,
            data.market_cap is not None,
            data.eps is not None,
        ])
        assert has_data, "At least one fundamental field should be populated"

    def test_symbol_cleaning(self, fetcher):
        assert fetcher._clean_symbol("TCS.NS") == "TCS"
        assert fetcher._ensure_ns("tcs") == "TCS.NS"


# -----------------------------------------------------------------------
# Macro Fetcher
# -----------------------------------------------------------------------

class TestMacroFetcher:

    def test_fetch_macro(self):
        fetcher = MacroFetcher()
        snap = fetcher.fetch()
        assert isinstance(snap, MacroSnapshot)
        # At minimum, defaults should populate repo_rate
        assert snap.repo_rate is not None
        assert snap.repo_rate > 0
        # USD/INR should be fetched from yfinance
        # (may be None if network fails, but should usually work)
        assert snap.timestamp is not None


# -----------------------------------------------------------------------
# News Fetcher
# -----------------------------------------------------------------------

class TestNewsFetcher:

    @pytest.fixture
    def fetcher(self):
        return NewsFetcher(delay=1)

    def test_fetch_for_ticker(self, fetcher):
        """Fetch headlines for a major stock."""
        headlines = fetcher.fetch_for_ticker("RELIANCE", max_headlines=5)
        assert isinstance(headlines, list)
        # Headlines may be empty if RSS is blocked, but structure should be correct
        if headlines:
            assert isinstance(headlines[0], Headline)
            assert headlines[0].title

    def test_fetch_for_sector(self, fetcher):
        """Fetch sector headlines."""
        headlines = fetcher.fetch_for_sector("IT", max_headlines=5)
        assert isinstance(headlines, list)
