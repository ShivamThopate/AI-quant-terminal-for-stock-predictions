"""
Unit & integration tests for feature engineering modules.
Tests technical indicators against known values and verifies feature matrix integrity.
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from features.technical import TechnicalIndicators
from features.sentiment import SentimentScorer
from features.feature_matrix import FeatureMatrixBuilder
from data.macro_fetcher import MacroSnapshot
from data.fundamental_fetcher import FundamentalData


# -----------------------------------------------------------------------
# Helpers — create synthetic OHLCV data
# -----------------------------------------------------------------------

def _make_ohlcv(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = 100 + np.cumsum(rng.randn(n) * 0.5)
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    open_ = close + rng.uniform(-1, 1, n)
    volume = rng.randint(100000, 1000000, n)

    return pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }, index=dates)


# -----------------------------------------------------------------------
# Technical Indicators
# -----------------------------------------------------------------------

class TestMACD:
    def test_macd_shape(self):
        df = _make_ohlcv()
        ti = TechnicalIndicators()
        result = ti.macd(df["Close"])
        assert "macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_hist" in result.columns
        assert len(result) == len(df)

    def test_macd_hist_is_difference(self):
        df = _make_ohlcv()
        ti = TechnicalIndicators()
        result = ti.macd(df["Close"])
        np.testing.assert_array_almost_equal(
            result["macd_hist"].values,
            (result["macd"] - result["macd_signal"]).values,
            decimal=10,
        )


class TestRSI:
    def test_rsi_range(self):
        df = _make_ohlcv()
        ti = TechnicalIndicators()
        rsi = ti.rsi(df["Close"])
        valid = rsi.dropna()
        assert valid.min() >= 0, "RSI should not go below 0"
        assert valid.max() <= 100, "RSI should not exceed 100"

    def test_rsi_length(self):
        df = _make_ohlcv()
        ti = TechnicalIndicators()
        rsi = ti.rsi(df["Close"])
        assert len(rsi) == len(df)


class TestATR:
    def test_atr_positive(self):
        df = _make_ohlcv()
        ti = TechnicalIndicators()
        atr = ti.atr(df["High"], df["Low"], df["Close"])
        valid = atr.dropna()
        assert (valid >= 0).all(), "ATR should always be non-negative"


class TestBollingerBands:
    def test_band_order(self):
        df = _make_ohlcv()
        ti = TechnicalIndicators()
        bb = ti.bollinger_bands(df["Close"])
        valid = bb.dropna()
        assert (valid["bb_upper"] >= valid["bb_middle"]).all()
        assert (valid["bb_middle"] >= valid["bb_lower"]).all()


class TestRelativeStrength:
    def test_rs_identity(self):
        """When stock == benchmark, RS should be ~1.0 everywhere."""
        df = _make_ohlcv()
        ti = TechnicalIndicators()
        rs = ti.relative_strength(df["Close"], df["Close"])
        np.testing.assert_array_almost_equal(rs.values, 1.0, decimal=10)

    def test_rs_outperformance(self):
        """A stock that grows faster should have RS > 1.0."""
        df = _make_ohlcv()
        stock_close = df["Close"] * np.linspace(1, 1.5, len(df))  # 50% boost
        bench_close = df["Close"]
        ti = TechnicalIndicators()
        rs = ti.relative_strength(stock_close, bench_close)
        # Last RS value should be > 1.0
        assert rs.iloc[-1] > 1.0


class TestComputeAll:
    def test_all_columns_present(self):
        df = _make_ohlcv()
        bench = _make_ohlcv(seed=99)
        ti = TechnicalIndicators()
        result = ti.compute_all(df, bench["Close"])

        expected = [
            "macd", "macd_signal", "macd_hist",
            "rsi", "atr",
            "bb_upper", "bb_middle", "bb_lower",
            "rs_vs_nifty",
            "returns_1d", "returns_5d", "volatility_20d",
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"


# -----------------------------------------------------------------------
# Sentiment Scorer
# -----------------------------------------------------------------------

class TestSentimentScorer:
    def test_positive_headline(self):
        scorer = SentimentScorer()
        score = scorer.score_single("Company reports record profits and strong growth")
        assert score > 0, "Positive headline should have positive score"

    def test_negative_headline(self):
        scorer = SentimentScorer()
        score = scorer.score_single("Stock crashes amid fraud allegations and losses")
        assert score < 0, "Negative headline should have negative score"

    def test_score_range(self):
        scorer = SentimentScorer()
        score = scorer.score_single("Neutral market update")
        assert -1.0 <= score <= 1.0


class TestSentimentAggregation:
    def test_empty_headlines(self):
        scorer = SentimentScorer()
        df = scorer.score_headlines([])
        assert df.empty

    def test_aggregate_by_ticker(self):
        scorer = SentimentScorer()
        # Create a small DataFrame
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "ticker": ["TCS.NS", "TCS.NS"],
            "sentiment_score": [0.5, 0.3],
            "headline_count": [1, 1],
        })
        agg = scorer.aggregate_by_ticker(df)
        assert "TCS.NS" in agg
        assert abs(agg["TCS.NS"] - 0.4) < 0.01


# -----------------------------------------------------------------------
# Feature Matrix Builder
# -----------------------------------------------------------------------

class TestFeatureMatrixBuilder:
    def test_build_basic(self):
        """Build a feature matrix from synthetic data."""
        builder = FeatureMatrixBuilder()

        price_data = {
            "TCS.NS": _make_ohlcv(seed=1),
            "INFY.NS": _make_ohlcv(seed=2),
        }
        bench = _make_ohlcv(seed=99)

        fundamentals = {
            "TCS.NS": FundamentalData(
                ticker="TCS.NS", pe_ratio=30.0, pb_ratio=10.0,
                roe=0.35, debt_to_equity=0.1,
            ),
            "INFY.NS": FundamentalData(
                ticker="INFY.NS", pe_ratio=25.0, pb_ratio=8.0,
                roe=0.28, debt_to_equity=0.05,
            ),
        }

        macro = MacroSnapshot(
            repo_rate=6.5, cpi_inflation=4.8, usdinr=83.5, gsec_10y_yield=7.1,
        )

        matrix = builder.build(
            price_data=price_data,
            benchmark_close=bench["Close"],
            fundamentals=fundamentals,
            macro=macro,
        )

        assert not matrix.empty
        assert "ticker" in matrix.columns
        assert matrix["ticker"].nunique() == 2
        assert "rsi" in matrix.columns
        assert "rs_vs_nifty" in matrix.columns

    def test_no_nan_after_build(self):
        """Feature matrix should have no NaN in critical columns after cleaning."""
        builder = FeatureMatrixBuilder()
        price_data = {"TCS.NS": _make_ohlcv(n=200)}
        bench = _make_ohlcv(n=200, seed=99)

        matrix = builder.build(price_data=price_data, benchmark_close=bench["Close"])

        critical = ["Close", "rsi", "macd"]
        for col in critical:
            if col in matrix.columns:
                assert matrix[col].isna().sum() == 0, f"NaN found in {col}"

    def test_feature_names(self):
        """get_feature_names should exclude identifier columns."""
        builder = FeatureMatrixBuilder()
        price_data = {"TCS.NS": _make_ohlcv()}
        matrix = builder.build(price_data=price_data, normalize=False)
        names = builder.get_feature_names(matrix)
        assert "Close" not in names
        assert "ticker" not in names
        assert "rsi" in names
