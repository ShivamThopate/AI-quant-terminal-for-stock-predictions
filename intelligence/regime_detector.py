"""
Market Regime Detector
========================
Uses K-Means clustering on historical returns and volatility to classify
the current market into one of three regimes:
    - "bull"             — high returns, moderate volatility
    - "bear"             — negative returns, moderate-to-high volatility
    - "high_volatility"  — extreme volatility regardless of direction

The detected regime is passed as a global context variable to the ML model
and used by the risk filter for circuit-breaker logic.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Default lookback windows for feature computation
_RETURNS_WINDOW = 20   # 20-day cumulative returns
_VOLATILITY_WINDOW = 20  # 20-day rolling volatility


class RegimeDetector:
    """
    Detect market regime via K-Means clustering.

    Usage::

        detector = RegimeDetector()
        regime_df = detector.fit_predict(benchmark_close)
        current = detector.current_regime(benchmark_close)
    """

    REGIME_LABELS = ("bull", "bear", "high_volatility")

    def __init__(self, n_regimes: int = 3, random_state: int = 42):
        self._n_regimes = n_regimes
        self._random_state = random_state
        self._kmeans: Optional[KMeans] = None
        self._scaler: Optional[StandardScaler] = None
        self._label_map: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_predict(self, close: pd.Series) -> pd.DataFrame:
        """
        Fit the regime model on a price series and return a DataFrame
        with regime labels for each date.

        Parameters
        ----------
        close : pd.Series
            Closing prices (e.g. Nifty 50), indexed by date.

        Returns
        -------
        pd.DataFrame
            Columns: [returns_20d, volatility_20d, regime_cluster, regime]
        """
        features = self._compute_features(close)
        features = features.dropna()

        if len(features) < self._n_regimes * 2:
            logger.warning("Not enough data for regime detection (%d rows)", len(features))
            features["regime_cluster"] = 0
            features["regime"] = "bull"
            return features

        # Scale features
        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(features[["returns_20d", "volatility_20d"]])

        # Fit K-Means
        self._kmeans = KMeans(
            n_clusters=self._n_regimes,
            random_state=self._random_state,
            n_init=10,
        )
        features["regime_cluster"] = self._kmeans.fit_predict(X)

        # Auto-label clusters based on centroid characteristics
        self._label_map = self._auto_label_clusters(features)
        features["regime"] = features["regime_cluster"].map(self._label_map)

        logger.info(
            "Regime detection complete: %s",
            features["regime"].value_counts().to_dict(),
        )
        return features

    def current_regime(self, close: pd.Series) -> str:
        """
        Return the regime label for the most recent date.

        If the model has not been fit, fits it first.
        """
        if self._kmeans is None:
            df = self.fit_predict(close)
            if df.empty:
                return "bull"  # safe default
            return df["regime"].iloc[-1]
        
        # Already fit, just predict the latest
        features = self._compute_features(close).dropna()
        if features.empty:
            return "bull"
        last_row = features.iloc[-1]
        return self.predict_regime(last_row["returns_20d"], last_row["volatility_20d"])

    def predict_regime(self, returns_20d: float, volatility_20d: float) -> str:
        """
        Predict regime for a single observation.

        Parameters
        ----------
        returns_20d : float
        volatility_20d : float

        Returns
        -------
        str — one of ("bull", "bear", "high_volatility")
        """
        if self._kmeans is None or self._scaler is None:
            raise RuntimeError("RegimeDetector has not been fit yet. Call fit_predict first.")

        X = self._scaler.transform([[returns_20d, volatility_20d]])
        cluster = self._kmeans.predict(X)[0]
        return self._label_map.get(cluster, "bull")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_features(close: pd.Series) -> pd.DataFrame:
        """Compute regime classification features from a price series."""
        df = pd.DataFrame(index=close.index)
        df["returns_20d"] = close.pct_change(_RETURNS_WINDOW)
        df["volatility_20d"] = close.pct_change().rolling(_VOLATILITY_WINDOW).std()
        return df

    def _auto_label_clusters(self, df: pd.DataFrame) -> dict:
        """
        Assign human-readable labels to clusters based on centroid stats.

        Logic:
        1. Cluster with highest mean volatility → high_volatility
        2. Among remaining, cluster with highest mean returns → bull
        3. The other → bear
        """
        cluster_stats = df.groupby("regime_cluster").agg(
            mean_ret=("returns_20d", "mean"),
            mean_vol=("volatility_20d", "mean"),
        )

        # Identify high-volatility cluster
        hv_cluster = cluster_stats["mean_vol"].idxmax()

        # Among the other clusters, find bull (highest returns) and bear (lowest)
        remaining = cluster_stats.drop(index=hv_cluster)
        if len(remaining) >= 2:
            bull_cluster = remaining["mean_ret"].idxmax()
            bear_cluster = remaining["mean_ret"].idxmin()
        elif len(remaining) == 1:
            # Only two clusters total: one is HV, other is bull or bear
            only_cluster = remaining.index[0]
            if remaining.loc[only_cluster, "mean_ret"] >= 0:
                bull_cluster = only_cluster
                bear_cluster = hv_cluster  # reassign HV as bear
            else:
                bear_cluster = only_cluster
                bull_cluster = hv_cluster
        else:
            bull_cluster = hv_cluster
            bear_cluster = hv_cluster

        label_map = {
            bull_cluster: "bull",
            bear_cluster: "bear",
            hv_cluster: "high_volatility",
        }

        logger.info("Regime cluster mapping: %s", label_map)
        logger.info("Cluster centroids:\n%s", cluster_stats.to_string())

        return label_map
