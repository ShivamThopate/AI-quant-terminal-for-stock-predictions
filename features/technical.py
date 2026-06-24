"""
Technical Indicators
=====================
Pure-pandas implementations of common technical analysis indicators.
No TA-Lib dependency required.

Indicators:
    - MACD (Moving Average Convergence Divergence)
    - RSI  (Relative Strength Index — Wilder's smoothing)
    - ATR  (Average True Range)
    - Bollinger Bands (upper, middle, lower)
    - RS   (Relative Strength vs benchmark — Nifty 50)
"""

import pandas as pd
import numpy as np

from config.settings import INDICATOR_DEFAULTS


class TechnicalIndicators:
    """
    Compute technical indicators from OHLCV DataFrames.

    All methods are stateless and return pd.Series or pd.DataFrame.

    Usage::

        ti = TechnicalIndicators()
        macd_df = ti.macd(df["Close"])
        rsi = ti.rsi(df["Close"])
        rs = ti.relative_strength(df["Close"], benchmark_close)
    """

    # ------------------------------------------------------------------
    # MACD
    # ------------------------------------------------------------------

    @staticmethod
    def macd(
        close: pd.Series,
        fast: int = INDICATOR_DEFAULTS["macd_fast"],
        slow: int = INDICATOR_DEFAULTS["macd_slow"],
        signal: int = INDICATOR_DEFAULTS["macd_signal"],
    ) -> pd.DataFrame:
        """
        Moving Average Convergence Divergence.

        Parameters
        ----------
        close : pd.Series — closing prices.
        fast  : int — fast EMA period (default 12).
        slow  : int — slow EMA period (default 26).
        signal: int — signal line EMA period (default 9).

        Returns
        -------
        pd.DataFrame with columns [macd, macd_signal, macd_hist].
        """
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return pd.DataFrame({
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_hist": histogram,
        }, index=close.index)

    # ------------------------------------------------------------------
    # RSI (Wilder's smoothing)
    # ------------------------------------------------------------------

    @staticmethod
    def rsi(
        close: pd.Series,
        period: int = INDICATOR_DEFAULTS["rsi_period"],
    ) -> pd.Series:
        """
        Relative Strength Index using Wilder's smoothing method.

        Parameters
        ----------
        close  : pd.Series — closing prices.
        period : int — lookback period (default 14).

        Returns
        -------
        pd.Series — RSI values (0–100).
        """
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        # Wilder's smoothing (equivalent to EMA with alpha = 1/period)
        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi_values = 100.0 - (100.0 / (1.0 + rs))

        rsi_series = pd.Series(rsi_values, index=close.index, name="rsi")
        return rsi_series

    # ------------------------------------------------------------------
    # ATR (Average True Range)
    # ------------------------------------------------------------------

    @staticmethod
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = INDICATOR_DEFAULTS["atr_period"],
    ) -> pd.Series:
        """
        Average True Range.

        Parameters
        ----------
        high, low, close : pd.Series — OHLC price series.
        period : int — smoothing period (default 14).

        Returns
        -------
        pd.Series — ATR values.
        """
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_values = true_range.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        return pd.Series(atr_values, index=close.index, name="atr")

    # ------------------------------------------------------------------
    # Bollinger Bands
    # ------------------------------------------------------------------

    @staticmethod
    def bollinger_bands(
        close: pd.Series,
        period: int = INDICATOR_DEFAULTS["bb_period"],
        std_dev: int = INDICATOR_DEFAULTS["bb_std"],
    ) -> pd.DataFrame:
        """
        Bollinger Bands (simple moving average ± N standard deviations).

        Returns
        -------
        pd.DataFrame with columns [bb_upper, bb_middle, bb_lower].
        """
        middle = close.rolling(window=period).mean()
        rolling_std = close.rolling(window=period).std()

        upper = middle + (rolling_std * std_dev)
        lower = middle - (rolling_std * std_dev)

        return pd.DataFrame({
            "bb_upper": upper,
            "bb_middle": middle,
            "bb_lower": lower,
        }, index=close.index)

    # ------------------------------------------------------------------
    # Volume Indicators (VWAP, OBV, Volume Ratio)
    # ------------------------------------------------------------------

    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Volume Weighted Average Price (20-day rolling approximation)."""
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).rolling(window=20).sum() / volume.rolling(window=20).sum()
        return pd.Series(vwap, name="vwap")

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume (OBV)."""
        direction = np.sign(close.diff()).fillna(1)
        obv = (volume * direction).cumsum()
        return pd.Series(obv, name="obv")

    @staticmethod
    def volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
        """Volume ratio: current volume vs N-day average volume."""
        avg_vol = volume.rolling(window=period).mean()
        ratio = volume / avg_vol.replace(0, np.nan)
        return pd.Series(ratio, name="volume_ratio")

    # ------------------------------------------------------------------
    # Relative Strength vs Benchmark
    # ------------------------------------------------------------------

    @staticmethod
    def relative_strength(
        stock_close: pd.Series,
        benchmark_close: pd.Series,
    ) -> pd.Series:
        """
        Relative Strength (RS) ratio: stock performance / benchmark performance.

        A rising RS means the stock is outperforming the benchmark (Nifty 50).
        RS = 1.0 when stock and benchmark have identical returns.
        RS > 1.0 means stock outperforms; RS < 1.0 means underperformance.

        Parameters
        ----------
        stock_close     : pd.Series — stock closing prices.
        benchmark_close : pd.Series — benchmark (Nifty 50) closing prices.

        Returns
        -------
        pd.Series — RS ratio (normalised so first value = 1.0).
        """
        # Align the two series on their common dates
        stock_aligned, bench_aligned = stock_close.align(benchmark_close, join="inner")

        if stock_aligned.empty or bench_aligned.empty:
            return pd.Series(dtype=float, name="rs_vs_nifty")

        # Cumulative return series (base = 1.0)
        stock_cum = stock_aligned / stock_aligned.iloc[0]
        bench_cum = bench_aligned / bench_aligned.iloc[0]

        rs = stock_cum / bench_cum

        return pd.Series(rs, name="rs_vs_nifty")

    # ------------------------------------------------------------------
    # Convenience: compute all indicators at once
    # ------------------------------------------------------------------

    def compute_all(
        self,
        ohlcv: pd.DataFrame,
        benchmark_close: pd.Series = None,
    ) -> pd.DataFrame:
        """
        Compute all technical indicators and merge them into the OHLCV frame.

        Parameters
        ----------
        ohlcv : pd.DataFrame
            Must contain columns: Open, High, Low, Close, Volume.
        benchmark_close : pd.Series, optional
            Nifty 50 close prices for RS calculation.

        Returns
        -------
        pd.DataFrame — original OHLCV + all indicator columns.
        """
        df = ohlcv.copy()
        close = df["Close"]

        # MACD
        macd_df = self.macd(close)
        df = df.join(macd_df)

        # RSI
        df["rsi"] = self.rsi(close)

        # ATR & ATR%
        df["atr"] = self.atr(df["High"], df["Low"], close)
        df["atr_pct"] = df["atr"] / close.replace(0, np.nan)

        # Bollinger Bands & %B
        bb_df = self.bollinger_bands(close)
        df = df.join(bb_df)
        bb_range = df["bb_upper"] - df["bb_lower"]
        df["bb_pct_b"] = np.where(bb_range == 0, 0.5, (close - df["bb_lower"]) / bb_range)

        # Volume indicators
        df["vwap"] = self.vwap(df["High"], df["Low"], close, df["Volume"])
        df["obv"] = self.obv(close, df["Volume"])
        df["volume_ratio"] = self.volume_ratio(df["Volume"])

        # Relative Strength vs benchmark
        if benchmark_close is not None:
            df["rs_vs_nifty"] = self.relative_strength(close, benchmark_close)
        else:
            df["rs_vs_nifty"] = np.nan

        # Derived features
        df["returns_1d"] = close.pct_change(1)
        df["returns_5d"] = close.pct_change(5)
        df["volatility_20d"] = close.pct_change().rolling(20).std()

        return df
