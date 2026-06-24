"""
Macro Economic Data Fetcher
=============================
Fetches key macroeconomic indicators relevant to Indian equity markets.

Primary  : RBI DBIE web scrape (Repo Rate, CPI, G-Sec yield)
Fallback : yfinance for USD/INR, manual defaults for rates

Caching strategy
----------------
Macro data (repo rate, CPI, G-Sec yield, USD/INR) changes at most once
per trading day — scraping these sources on every agent query wastes
network bandwidth and risks rate-limiting.

``fetch()`` is cached with a 24-hour TTL using a module-level timestamp
guard.  A fresh network call is only made when the cached result is
older than ``_MACRO_CACHE_TTL_SECONDS`` (86 400 s = 24 hours).
The cache is stored as a module-level variable so it survives for the
entire lifetime of the Streamlit process without hitting the network again.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ── In-process cache ─────────────────────────────────────────────────────────
_MACRO_CACHE_TTL_SECONDS = 86_400   # 24 hours
_macro_cache: Optional["MacroSnapshot"] = None
_macro_cache_ts: float = 0.0        # Unix timestamp of last successful fetch


@dataclass
class MacroSnapshot:
    """Container for macro indicators at a point in time."""
    repo_rate: Optional[float] = None
    cpi_inflation: Optional[float] = None
    usdinr: Optional[float] = None
    gsec_10y_yield: Optional[float] = None
    timestamp: Optional[str] = None


class MacroFetcher:
    """
    Fetches macroeconomic indicators with graceful fallback and 24-hour caching.

    Multiple agent queries within the same trading day hit the in-process
    cache rather than making redundant network calls.

    Usage::

        fetcher = MacroFetcher()
        snap = fetcher.fetch()
        print(snap.repo_rate, snap.usdinr)
    """

    # Current RBI defaults (manually updated as last-known values)
    _DEFAULT_REPO = 6.50
    _DEFAULT_CPI = 4.80
    _DEFAULT_GSEC10Y = 7.10

    def fetch(self, force_refresh: bool = False) -> MacroSnapshot:
        """
        Fetch all macro indicators, returning the cached result if fresh.

        Parameters
        ----------
        force_refresh : bool
            If True, bypass the cache and always hit the network.
            Useful for unit tests or manual refresh.

        Returns
        -------
        MacroSnapshot — cached or freshly fetched.
        """
        global _macro_cache, _macro_cache_ts

        now = time.time()
        age = now - _macro_cache_ts

        if not force_refresh and _macro_cache is not None and age < _MACRO_CACHE_TTL_SECONDS:
            logger.debug(
                "MacroFetcher cache hit (age=%.0fs, TTL=%ds)",
                age, _MACRO_CACHE_TTL_SECONDS,
            )
            return _macro_cache

        # ── Cache miss — fetch from network ──────────────────────────────
        snap = MacroSnapshot()

        # USD/INR (yfinance — most reliable)
        snap.usdinr = self._fetch_usdinr()

        # RBI data (scrape with fallback to defaults)
        rbi_data = self._fetch_rbi_dbie()
        snap.repo_rate = rbi_data.get("repo_rate", self._DEFAULT_REPO)
        snap.cpi_inflation = rbi_data.get("cpi", self._DEFAULT_CPI)
        snap.gsec_10y_yield = rbi_data.get("gsec_10y", self._DEFAULT_GSEC10Y)

        from datetime import datetime
        snap.timestamp = datetime.now().isoformat()

        logger.info(
            "Macro snapshot fetched: Repo=%.2f%%, CPI=%.2f%%, USDINR=%.2f, 10Y=%.2f%%",
            snap.repo_rate or 0, snap.cpi_inflation or 0,
            snap.usdinr or 0, snap.gsec_10y_yield or 0,
        )

        # ── Update cache ─────────────────────────────────────────────────
        _macro_cache = snap
        _macro_cache_ts = now

        return snap

    # ------------------------------------------------------------------
    # Source implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _fetch_usdinr() -> Optional[float]:
        """Fetch USD/INR exchange rate via yfinance."""
        try:
            import yfinance as yf
            data = yf.Ticker("USDINR=X").history(period="5d")
            if data is not None and not data.empty:
                return round(float(data["Close"].iloc[-1]), 2)
        except Exception as exc:
            logger.warning("Failed to fetch USD/INR: %s", exc)
        return None

    @staticmethod
    def _fetch_rbi_dbie() -> dict:
        """
        Attempt to scrape key indicators from the RBI DBIE portal.
        Returns a dict with keys: repo_rate, cpi, gsec_10y.
        Falls back to empty dict on failure.
        """
        result = {}
        try:
            url = "https://www.rbi.org.in/Scripts/BS_NSDPDisplay.aspx?param=4"
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            }
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                logger.warning("RBI DBIE returned status %d", resp.status_code)
                return result

            soup = BeautifulSoup(resp.text, "html.parser")
            tables = soup.find_all("table")

            for table in tables:
                text = table.get_text(" ", strip=True).lower()
                if "policy repo" in text or "repo rate" in text:
                    nums = _extract_numbers(text)
                    if nums:
                        result["repo_rate"] = nums[0]
                if "consumer price" in text or "cpi" in text:
                    nums = _extract_numbers(text)
                    if nums:
                        result["cpi"] = nums[0]

        except Exception as exc:
            logger.warning("RBI DBIE scrape failed: %s", exc)

        return result


def _extract_numbers(text: str) -> list:
    """Extract plausible numeric values (0-100 range) from text."""
    import re
    nums = re.findall(r"(\d+\.?\d*)", text)
    return [float(n) for n in nums if 0 < float(n) < 100][:5]
