"""
Triple Barrier Method Labeler
===============================
Implements the Triple Barrier labeling scheme from Marcos López de Prado's
*Advances in Financial Machine Learning*.

For each observation, looks forward up to ``max_days`` trading days:
    - Upper barrier (+take_profit_pct): label = 1  (TP hit first)
    - Lower barrier (-stop_loss_pct):   label = -1 (SL hit first)
    - Vertical barrier (time expires):  label = 0  (neutral / timeout)

The ML model predicts P(label=1) — the probability of a profitable trade.
"""

import logging

import numpy as np
import pandas as pd

from config.settings import TRIPLE_BARRIER

logger = logging.getLogger(__name__)


class TripleBarrierLabeler:
    """
    Generate Triple Barrier labels for a price series.

    Usage::

        labeler = TripleBarrierLabeler()
        labels = labeler.label(close_prices)
        # labels: pd.Series with values in {-1, 0, 1}
    """

    def __init__(
        self,
        take_profit_pct: float = TRIPLE_BARRIER["take_profit_pct"],
        stop_loss_pct: float = TRIPLE_BARRIER["stop_loss_pct"],
        max_holding_days: int = TRIPLE_BARRIER["max_holding_days"],
    ):
        self.tp = take_profit_pct   # +3%
        self.sl = stop_loss_pct     # -2%
        self.max_days = max_holding_days  # 5

    def label(self, close: pd.Series) -> pd.Series:
        """
        Generate triple barrier labels for every bar in the series.

        Parameters
        ----------
        close : pd.Series
            Closing prices indexed by date.

        Returns
        -------
        pd.Series
            Label values: 1 (TP hit), -1 (SL hit), 0 (timeout).
            NaN for the last ``max_days`` bars (insufficient forward data).
        """
        n = len(close)
        if n == 0:
            return pd.Series(dtype=float)

        close_values = close.values
        labels = np.zeros(n, dtype=float)

        upper = close_values * (1.0 + self.tp)
        lower = close_values * (1.0 - self.sl)

        hit_tp = np.full(n, self.max_days + 1)
        hit_sl = np.full(n, self.max_days + 1)

        # Look forward up to max_days using shifted arrays
        for shift in range(1, self.max_days + 1):
            future = np.full(n, np.nan)
            if n > shift:
                future[:-shift] = close_values[shift:]

            # Where does it hit TP first?
            mask_tp = (future >= upper) & (hit_tp == self.max_days + 1)
            hit_tp[mask_tp] = shift

            # Where does it hit SL first?
            mask_sl = (future <= lower) & (hit_sl == self.max_days + 1)
            hit_sl[mask_sl] = shift

        hit_any_tp = hit_tp <= self.max_days
        hit_any_sl = hit_sl <= self.max_days

        labels[hit_any_tp & (hit_tp <= hit_sl)] = 1.0
        labels[hit_any_sl & (hit_sl < hit_tp)] = -1.0

        # Invalid entries (NaN, <= 0)
        invalid = np.isnan(close_values) | (close_values <= 0)
        labels[invalid] = np.nan

        # Mark last max_days bars as NaN (insufficient forward data)
        if n >= self.max_days:
            labels[-self.max_days:] = np.nan

        labels_series = pd.Series(labels, index=close.index)

        # Log distribution
        valid = labels_series.dropna()
        if len(valid) > 0:
            dist = valid.value_counts().to_dict()
            logger.info(
                "Triple Barrier labels: TP(+1)=%d, SL(-1)=%d, Neutral(0)=%d, "
                "total=%d, NaN=%d",
                dist.get(1.0, 0), dist.get(-1.0, 0), dist.get(0.0, 0),
                len(valid), labels_series.isna().sum(),
            )

        return labels_series

    def label_dataframe(self, df: pd.DataFrame, close_col: str = "Close") -> pd.DataFrame:
        """
        Add a ``barrier_label`` column to a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain a close price column.
        close_col : str
            Name of the close price column.

        Returns
        -------
        pd.DataFrame — original df with ``barrier_label`` column added.
        """
        result = df.copy()
        result["barrier_label"] = self.label(result[close_col])
        return result
