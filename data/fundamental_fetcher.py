"""
Fundamental Data Fetcher
=========================
Fetches fundamental ratios for NSE-listed equities.

Primary source : nsepython (direct NSE data)
Fallback source: yfinance .info dictionary
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from config.settings import FUNDAMENTAL_SOURCES, NSE_SUFFIX

logger = logging.getLogger(__name__)

# ── In-process per-ticker cache ───────────────────────────────────────────────
# Fundamentals are quarterly data; caching for 6 hours prevents hammering
# yfinance / NSE on every repeated query within the same trading session.
_FUNDAMENTAL_CACHE_TTL_SECONDS = 6 * 3600   # 6 hours
_fundamental_cache: Dict[str, "FundamentalData"] = {}
_fundamental_cache_ts: Dict[str, float] = {}  # ticker → Unix timestamp


@dataclass
class FundamentalData:
    """Standardised container for a stock's fundamental ratios."""
    ticker: str
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    roe: Optional[float] = None
    debt_to_equity: Optional[float] = None
    market_cap: Optional[float] = None
    dividend_yield: Optional[float] = None
    eps: Optional[float] = None
    book_value: Optional[float] = None
    face_value: Optional[float] = None
    industry: Optional[str] = None


def _safe_float(value) -> Optional[float]:
    """Safely convert a value to float, returning None on failure."""
    if value is None:
        return None
    try:
        result = float(value)
        if result != result or result == float("inf") or result == float("-inf"):
            return None
        return result
    except (ValueError, TypeError):
        return None


class FundamentalFetcher:
    """
    Fetches fundamental data with automatic fallback.

    Usage::

        fetcher = FundamentalFetcher()
        data = fetcher.fetch("TCS")
    """

    def fetch(self, ticker: str, force_refresh: bool = False) -> "FundamentalData":
        """Fetch fundamental ratios for a single ticker, with 6-hour cache."""
        symbol = self._clean_symbol(ticker)
        ns_ticker = self._ensure_ns(ticker)

        now = time.time()
        cached = _fundamental_cache.get(ns_ticker)
        cache_age = now - _fundamental_cache_ts.get(ns_ticker, 0.0)

        if not force_refresh and cached is not None and cache_age < _FUNDAMENTAL_CACHE_TTL_SECONDS:
            logger.debug(
                "Fundamental cache hit for %s (age=%.0fs)", ns_ticker, cache_age
            )
            return cached

        # ── Cache miss — fetch from network ──────────────────────────────
        result = None
        for source in FUNDAMENTAL_SOURCES:
            try:
                if source == "nsepython":
                    result = self._fetch_nsepython(symbol, ns_ticker)
                elif source == "yfinance":
                    result = self._fetch_yfinance(ns_ticker)
                else:
                    continue
                if result is not None:
                    logger.info("Fetched fundamentals for %s from %s", ticker, source)
                    break
            except Exception as exc:
                logger.warning("Source '%s' failed for %s: %s", source, ticker, exc)
                continue

        if result is None:
            logger.error("All fundamental sources failed for %s", ticker)
            result = FundamentalData(ticker=ns_ticker)

        # ── Store in cache ───────────────────────────────────────────────
        _fundamental_cache[ns_ticker] = result
        _fundamental_cache_ts[ns_ticker] = now

        return result

    def fetch_multiple(self, tickers: List[str]) -> Dict[str, FundamentalData]:
        """Fetch fundamentals for a list of tickers."""
        return {self._ensure_ns(t): self.fetch(t) for t in tickers}

    @staticmethod
    def _fetch_nsepython(symbol: str, ns_ticker: str) -> Optional[FundamentalData]:
        """
        Fetch via nsepython.

        nse_eq() returns a nested dict with keys like:
            metadata, securityInfo, priceInfo, industryInfo, etc.
        We extract as many fundamental fields as possible.
        """
        try:
            from nsepython import nse_eq
            info = nse_eq(symbol)
            if info is None:
                return None

            metadata = info.get("metadata", {})
            security_info = info.get("securityInfo", {})
            price_info = info.get("priceInfo", {})
            industry_info = info.get("industryInfo", {})

            # Extract P/E from multiple possible locations
            pe = _safe_float(metadata.get("pdSymbolPe")) or _safe_float(info.get("pe"))
            pb = _safe_float(metadata.get("pdSectorPe")) or _safe_float(info.get("pb"))

            # Market cap: NSE sometimes provides it under securityInfo
            market_cap = _safe_float(security_info.get("issuedSize"))

            # Industry from industryInfo or metadata
            industry = (
                industry_info.get("basicIndustry")
                or industry_info.get("industry")
                or metadata.get("industry")
            )

            return FundamentalData(
                ticker=ns_ticker,
                pe_ratio=pe,
                pb_ratio=pb,
                face_value=_safe_float(security_info.get("faceValue")),
                market_cap=market_cap,
                industry=industry,
                # ROE, D/E, EPS, book_value not available from nse_eq —
                # these require the yfinance fallback
            )
        except ImportError:
            logger.warning("nsepython not installed, skipping")
            return None

    @staticmethod
    def _fetch_yfinance(ns_ticker: str) -> Optional[FundamentalData]:
        """Fetch via yfinance .info dict."""
        import yfinance as yf
        stock = yf.Ticker(ns_ticker)
        info = stock.info
        if not info or info.get("regularMarketPrice") is None:
            return None
        return FundamentalData(
            ticker=ns_ticker,
            pe_ratio=_safe_float(info.get("trailingPE") or info.get("forwardPE")),
            pb_ratio=_safe_float(info.get("priceToBook")),
            roe=_safe_float(info.get("returnOnEquity")),
            debt_to_equity=_safe_float(info.get("debtToEquity")),
            market_cap=_safe_float(info.get("marketCap")),
            dividend_yield=_safe_float(info.get("dividendYield")),
            eps=_safe_float(info.get("trailingEps")),
            book_value=_safe_float(info.get("bookValue")),
            industry=info.get("industry"),
        )

    @staticmethod
    def _clean_symbol(ticker: str) -> str:
        return ticker.replace(".NS", "").replace(".ns", "").upper()

    @staticmethod
    def _ensure_ns(ticker: str) -> str:
        clean = ticker.replace(".NS", "").replace(".ns", "").upper()
        return f"{clean}{NSE_SUFFIX}"
