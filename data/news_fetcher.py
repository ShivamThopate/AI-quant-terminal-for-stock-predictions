"""
News / Headlines Fetcher
==========================
Scrapes recent financial headlines via Google News RSS for sentiment analysis.
Rate-limited to avoid blocks.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from config.settings import NEWS_REQUEST_DELAY_SECONDS

logger = logging.getLogger(__name__)


@dataclass
class Headline:
    """A single news headline."""
    title: str
    source: Optional[str] = None
    published_date: Optional[datetime] = None
    ticker: Optional[str] = None
    url: Optional[str] = None


class NewsFetcher:
    """
    Fetches recent financial headlines from Google News RSS.

    Usage::

        fetcher = NewsFetcher()
        headlines = fetcher.fetch_for_ticker("RELIANCE")
        headlines = fetcher.fetch_for_sector("IT")
    """

    _GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}+stock+NSE&hl=en-IN&gl=IN&ceid=IN:en"

    def __init__(self, delay: float = NEWS_REQUEST_DELAY_SECONDS):
        self._delay = delay

    def fetch_for_ticker(self, ticker: str, max_headlines: int = 20) -> List[Headline]:
        """Fetch headlines for a specific NSE ticker."""
        symbol = ticker.replace(".NS", "").replace(".ns", "")
        return self._fetch(symbol, ticker, max_headlines)

    def fetch_for_sector(self, sector: str, max_headlines: int = 20) -> List[Headline]:
        """Fetch headlines for a sector name (e.g. 'IT', 'Banking')."""
        query = f"India {sector} sector stocks"
        return self._fetch(query, sector, max_headlines)

    def fetch_for_tickers(self, tickers: List[str], max_per_ticker: int = 10) -> List[Headline]:
        """Fetch headlines for multiple tickers with rate limiting."""
        all_headlines = []
        for ticker in tickers:
            headlines = self.fetch_for_ticker(ticker, max_per_ticker)
            all_headlines.extend(headlines)
            time.sleep(self._delay)
        return all_headlines

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fetch(self, query: str, tag: str, max_items: int) -> List[Headline]:
        """Fetch and parse RSS feed."""
        try:
            import feedparser
        except ImportError:
            logger.error("feedparser not installed — cannot fetch news")
            return []

        url = self._GOOGLE_NEWS_RSS.format(query=query.replace(" ", "+"))

        try:
            feed = feedparser.parse(url)
        except Exception as exc:
            logger.warning("RSS fetch failed for '%s': %s", query, exc)
            return []

        headlines = []
        for entry in feed.entries[:max_items]:
            pub_date = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                try:
                    pub_date = datetime(*entry.published_parsed[:6])
                except Exception:
                    pass

            source = None
            if hasattr(entry, "source") and hasattr(entry.source, "title"):
                source = entry.source.title
            elif " - " in entry.get("title", ""):
                source = entry.title.rsplit(" - ", 1)[-1]

            title = entry.get("title", "").strip()
            # Remove source suffix from title if present
            if source and title.endswith(f" - {source}"):
                title = title[: -(len(source) + 3)].strip()

            headlines.append(Headline(
                title=title,
                source=source,
                published_date=pub_date,
                ticker=tag,
                url=entry.get("link"),
            ))

        logger.info("Fetched %d headlines for '%s'", len(headlines), tag)
        return headlines
