"""
Feature Matrix Builder
========================
Merges all data sources (price + technicals + sentiment + fundamentals + macro)
into a single, clean, normalised DataFrame ready for ML consumption.

Normalisation strategy (look-ahead-safe):
  Per-ticker rolling Z-score with a 252-day window (≈ 1 trading year).
  No global mean/std is ever computed across the full dataset — every
  normalised value at date T only uses data from [T-252, T-1].
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from features.technical import TechnicalIndicators
from features.sentiment import SentimentScorer
from data.fundamental_fetcher import FundamentalData
from data.macro_fetcher import MacroSnapshot

logger = logging.getLogger(__name__)

# Columns that must never be normalised (raw prices, identifiers, counts)
_NORMALISE_EXCLUDE = {
    "ticker", "Open", "High", "Low", "Close", "Volume", "headline_count"
}

# Rolling window for Z-score normalisation (≈ 1 trading year)
_ROLL_WINDOW = 252
_ROLL_MIN_PERIODS = 30


class FeatureMatrixBuilder:
    """
    Builds a unified feature matrix from heterogeneous data sources.

    Usage::

        builder = FeatureMatrixBuilder()
        matrix = builder.build(
            price_data={"TCS.NS": df_tcs, "INFY.NS": df_infy},
            benchmark_close=nifty_close,
            fundamentals={"TCS.NS": fund_tcs, ...},
            macro=macro_snapshot,
            sentiment_df=sentiment_dataframe,
        )
    """

    def __init__(self):
        self.tech = TechnicalIndicators()
        self.scorer = SentimentScorer()

    def build(
        self,
        price_data: Dict[str, pd.DataFrame],
        benchmark_close: pd.Series = None,
        fundamentals: Optional[Dict[str, FundamentalData]] = None,
        macro: Optional[MacroSnapshot] = None,
        sentiment_df: Optional[pd.DataFrame] = None,
        normalize: bool = True,
    ) -> pd.DataFrame:
        """
        Build and return the unified feature matrix.

        Parameters
        ----------
        price_data : dict[str, pd.DataFrame]
            Ticker → OHLCV DataFrame.
        benchmark_close : pd.Series
            Nifty 50 close prices.
        fundamentals : dict[str, FundamentalData], optional
        macro : MacroSnapshot, optional
        sentiment_df : pd.DataFrame, optional
            Columns: [date, ticker, sentiment_score, headline_count].
        normalize : bool
            Whether to apply rolling Z-score normalisation per ticker.
            Pass ``normalize=False`` during training so the labeler can
            operate on raw prices (LSTM gets normalised features separately
            via the DataLoader pipeline).

        Returns
        -------
        pd.DataFrame — clean, unified feature matrix.
        """
        frames = []

        for ticker, ohlcv in price_data.items():
            if ohlcv.empty:
                continue

            # 1. Compute technical indicators
            df = self.tech.compute_all(ohlcv, benchmark_close)
            df["ticker"] = ticker

            # 2. Merge fundamentals (static — broadcast across all rows)
            if fundamentals and ticker in fundamentals:
                fund = fundamentals[ticker]
                df["pe_ratio"] = fund.pe_ratio
                df["pb_ratio"] = fund.pb_ratio
                df["roe"] = fund.roe
                df["debt_to_equity"] = fund.debt_to_equity

            # 3. Merge macro (static — broadcast across all rows)
            if macro:
                df["repo_rate"] = macro.repo_rate
                df["cpi"] = macro.cpi_inflation
                df["usdinr"] = macro.usdinr
                df["gsec_10y"] = macro.gsec_10y_yield

            # 4. Merge sentiment (daily — join on date)
            if sentiment_df is not None and not sentiment_df.empty:
                ticker_sent = sentiment_df[sentiment_df["ticker"] == ticker].copy()
                if not ticker_sent.empty:
                    ticker_sent["date"] = pd.to_datetime(ticker_sent["date"])
                    ticker_sent = ticker_sent.set_index("date")[
                        ["sentiment_score", "headline_count"]
                    ]
                    ticker_sent = ticker_sent.reindex(df.index, method="ffill")
                    df["sentiment_score"] = ticker_sent["sentiment_score"]
                    df["headline_count"] = ticker_sent["headline_count"]

            if "sentiment_score" not in df.columns:
                df["sentiment_score"] = 0.0
                df["headline_count"] = 0

            # 5. Add lag features
            df = self._add_lag_features(df)

            # 6. Drop leaky absolute Bollinger Band price columns
            #    (keep only bb_pct_b which is scale-invariant)
            for col in ["bb_upper", "bb_middle", "bb_lower"]:
                if col in df.columns:
                    df = df.drop(columns=[col])

            frames.append(df)

        if not frames:
            logger.warning("No frames to build feature matrix from")
            return pd.DataFrame()

        # Concatenate all tickers
        matrix = pd.concat(frames, axis=0)

        # 7. Handle missing data
        matrix = self._handle_missing(matrix)

        # 8. Normalise numeric columns (rolling Z-score, per-ticker, look-ahead-safe)
        if normalize:
            matrix = self._normalise(matrix)

        logger.info(
            "Feature matrix built: %d rows × %d cols, %d tickers",
            len(matrix),
            len(matrix.columns),
            matrix["ticker"].nunique() if "ticker" in matrix.columns else 0,
        )
        return matrix

    # ──────────────────────────────────────────────────────────────────────
    # Lag & rate-of-change features
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _add_lag_features(df: pd.DataFrame, lag: int = 5) -> pd.DataFrame:
        """
        Add lagged and rate-of-change features to give the model
        trend-over-time context.

        Lag features: what was RSI / MACD_hist / returns_1d N days ago?
        ROC features: how much did RSI / volume change over N days?
        """
        lag_cols = ["rsi", "macd_hist", "returns_1d"]
        for col in lag_cols:
            if col in df.columns:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)

        if "rsi" in df.columns:
            df["rsi_roc"] = df["rsi"] - df["rsi"].shift(lag)
        if "Volume" in df.columns:
            avg_vol = df["Volume"].rolling(lag).mean()
            df["volume_roc"] = (df["Volume"] - avg_vol) / avg_vol.replace(0, 1)
        if "atr" in df.columns:
            df["atr_roc"] = df["atr"].pct_change(lag)

        return df

    # ──────────────────────────────────────────────────────────────────────
    # Missing data handling
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _handle_missing(df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill → backward-fill → fill remaining with 0 → drop critical NaN."""
        non_numeric = ["ticker"]
        numeric_cols = [c for c in df.columns if c not in non_numeric]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df[numeric_cols] = df[numeric_cols].ffill().bfill()

        critical = ["Close", "rsi", "macd"]
        critical_present = [c for c in critical if c in df.columns]
        if critical_present:
            df = df.dropna(subset=critical_present)

        df[numeric_cols] = df[numeric_cols].fillna(0.0)
        return df

    # ──────────────────────────────────────────────────────────────────────
    # Look-ahead-safe normalisation
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise(df: pd.DataFrame) -> pd.DataFrame:
        """
        Rolling Z-score normalisation — per ticker, strictly look-ahead-safe.

        For each numeric feature column and each ticker, every value at date T
        is standardised using only the rolling mean and std computed from the
        preceding _ROLL_WINDOW trading days:

            z_t = (x_t - mean(x_{t-W}..x_{t-1})) / std(x_{t-W}..x_{t-1})

        This is achieved by setting ``closed="left"`` (or equivalently
        ``shift(1)`` before rolling), but pandas rolling() with default
        ``closed="right"`` includes the current value.  To avoid this we
        expand the rolling window and use a 1-period shifted series so T is
        not included in its own normalisation statistics.

        ``min_periods=_ROLL_MIN_PERIODS`` (30 days): rows before the warm-up
        period will have their z-score set to 0.0 (handled via fillna).

        Columns excluded from normalisation:
          - ``ticker`` (string identifier)
          - Raw OHLCV prices (``Open``, ``High``, ``Low``, ``Close``, ``Volume``)
          - ``headline_count`` (integer count — kept raw)

        Returns
        -------
        pd.DataFrame with the same shape; normalised features in-place.
        """
        exclude = _NORMALISE_EXCLUDE
        norm_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude
        ]

        if not norm_cols:
            return df

        ticker_col = df["ticker"].copy()

        def _rolling_zscore_group(group: pd.DataFrame) -> pd.DataFrame:
            for col in norm_cols:
                if col not in group.columns:
                    continue
                series = group[col]

                # Shift by 1 so each point's mean/std only uses *past* values
                shifted = series.shift(1)
                roll = shifted.rolling(window=_ROLL_WINDOW, min_periods=_ROLL_MIN_PERIODS)
                mean_ = roll.mean()
                std_ = roll.std()

                # Avoid division by zero (constant features)
                std_safe = std_.replace(0.0, np.nan)

                z = (series - mean_) / std_safe

                # Replace inf / NaN with 0.0 (warm-up period or constant series)
                z = z.replace([np.inf, -np.inf], 0.0).fillna(0.0)
                group = group.copy()
                group[col] = z
            return group

        df = df.groupby("ticker", group_keys=False).apply(_rolling_zscore_group)

        # Restore ticker column if groupby consumed it
        if "ticker" not in df.columns:
            df["ticker"] = ticker_col

        return df

    # ──────────────────────────────────────────────────────────────────────
    # Summary helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def get_feature_names(matrix: pd.DataFrame) -> List[str]:
        """Return the list of feature column names (excluding identifiers)."""
        exclude = {"ticker", "Open", "High", "Low", "Close", "Volume"}
        return [c for c in matrix.columns if c not in exclude]
