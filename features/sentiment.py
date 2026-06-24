"""
Sentiment Scorer
=================
Scores financial news headlines using **FinBERT** (ProsusAI/finbert), a
transformer model pre-trained on financial text. Unlike VADER (dictionary-
based word counting), FinBERT understands financial context — e.g. it knows
"profit fell less than expected" is actually positive.

Falls back to VADER automatically if FinBERT can't load (missing torch, etc.).

Aggregates per-ticker, per-day sentiment scores.
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

from config.settings import SENTIMENT_HEADLINE_LOOKBACK_DAYS

logger = logging.getLogger(__name__)

# Lazy-loaded FinBERT scorer singleton
_finbert: Optional["FinBERTScorer"] = None


def _get_scorer():
    """Lazy-initialise the FinBERT sentiment scorer."""
    global _finbert
    if _finbert is None:
        from nlp.finbert_scorer import FinBERTScorer
        _finbert = FinBERTScorer()
    return _finbert


class SentimentScorer:
    """
    Score a list of headlines and return per-ticker daily sentiment.

    Now powered by FinBERT (deep learning) instead of VADER (dictionary).
    Falls back to VADER automatically if FinBERT can't load.

    Usage::

        scorer = SentimentScorer()
        df = scorer.score_headlines(headlines)
        # df columns: [date, ticker, sentiment_score, headline_count]
    """

    def score_single(self, text: str) -> float:
        """
        Score a single headline using FinBERT.

        Returns a compound score in [-1.0, +1.0].
        """
        scorer = _get_scorer()
        return scorer.score(text)

    def score_headlines(
        self,
        headlines: list,
        lookback_days: int = SENTIMENT_HEADLINE_LOOKBACK_DAYS,
    ) -> pd.DataFrame:
        """
        Score a list of ``Headline`` objects and aggregate by ticker + date.

        Uses FinBERT batch inference for efficiency when processing
        multiple headlines at once.

        Parameters
        ----------
        headlines : List[Headline]
            Each must have attributes: title, ticker, published_date.
        lookback_days : int
            Only consider headlines published within this many days.

        Returns
        -------
        pd.DataFrame
            Columns: [date, ticker, sentiment_score, headline_count].
            ``sentiment_score`` is the mean FinBERT score for that ticker-day.
        """
        if not headlines:
            return pd.DataFrame(columns=["date", "ticker", "sentiment_score", "headline_count"])

        cutoff = datetime.now() - timedelta(days=lookback_days)

        # Filter headlines by date first
        valid_headlines = []
        for h in headlines:
            if h.published_date and h.published_date < cutoff:
                continue
            valid_headlines.append(h)

        if not valid_headlines:
            return pd.DataFrame(columns=["date", "ticker", "sentiment_score", "headline_count"])

        # Batch-score all headlines at once (much faster than one-by-one)
        scorer = _get_scorer()
        texts = [h.title for h in valid_headlines]
        scores = scorer.score_batch(texts)

        # Group scores by (ticker, date)
        groups: Dict[tuple, List[float]] = defaultdict(list)

        for h, score in zip(valid_headlines, scores):
            date_key = (
                h.published_date.date() if h.published_date else datetime.now().date()
            )
            ticker_key = h.ticker or "UNKNOWN"
            groups[(ticker_key, date_key)].append(score)

        # Build DataFrame
        rows = []
        for (ticker, date), group_scores in groups.items():
            rows.append({
                "date": date,
                "ticker": ticker,
                "sentiment_score": sum(group_scores) / len(group_scores),
                "headline_count": len(group_scores),
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        return df

    def aggregate_by_ticker(self, sentiment_df: pd.DataFrame) -> Dict[str, float]:
        """
        Collapse the daily sentiment DataFrame into a single score per ticker.

        Returns
        -------
        dict[str, float] — ticker → mean sentiment score.
        """
        if sentiment_df.empty:
            return {}
        return (
            sentiment_df.groupby("ticker")["sentiment_score"]
            .mean()
            .to_dict()
        )
