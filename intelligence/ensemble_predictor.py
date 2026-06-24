"""
Ensemble Predictor
====================
Combines the snapshot-based XGBoost model with the temporal-based LSTM model.

Key design decisions:
  - Both models expose the same ``predict_latest`` interface returning
    {p_take_profit, p_stop_loss, p_neutral, predicted_label} per ticker.
  - Dynamic weights are loaded from ``models/ensemble_weights.json`` and are
    computed during training by comparing chronological-holdout accuracies.
  - If the LSTM returns an empty result (e.g., insufficient sequence length),
    the ensemble falls back gracefully to XGBoost-only predictions.
  - The final predicted_label is re-derived from the blended probabilities so
    it is consistent with the ensemble output (not taken from either sub-model).
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config.settings import MODELS_DIR, SL_DETECTION_THRESHOLD
from intelligence.predictor import DirectionalPredictor
from intelligence.lstm_predictor import LSTMPredictor

logger = logging.getLogger(__name__)

# Default weights if no persisted weights file exists
_DEFAULT_XGB_WEIGHT = 0.6
_DEFAULT_LSTM_WEIGHT = 0.4


class EnsemblePredictor:
    """
    Weighted ensemble of XGBoost and LSTM predictors.

    Usage::

        # Training (run once via retrain.py):
        ensemble = EnsemblePredictor()
        # Individual training is done by retrain.py; ensemble just loads weights.

        # Inference:
        ensemble = EnsemblePredictor()
        results = ensemble.predict_latest(feature_matrix)
        # results: {"RELIANCE.NS": {"p_take_profit": 0.52, ...}, ...}
    """

    def __init__(
        self,
        xgb_path: Optional[Path] = None,
        lstm_path: Optional[Path] = None,
        weights_path: Optional[Path] = None,
    ):
        self.xgb = DirectionalPredictor(model_path=xgb_path)
        self.lstm = LSTMPredictor(model_path=lstm_path)
        self._weights_path = weights_path or MODELS_DIR / "ensemble_weights.json"

        # Defaults — overwritten by _load_weights()
        self.xgb_weight = _DEFAULT_XGB_WEIGHT
        self.lstm_weight = _DEFAULT_LSTM_WEIGHT
        self._load_weights()

    # ──────────────────────────────────────────────────────────────────────
    # Weight management
    # ──────────────────────────────────────────────────────────────────────

    def _load_weights(self) -> None:
        """Load dynamic weights from the JSON file persisted by retrain.py."""
        if self._weights_path.exists():
            try:
                with open(self._weights_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                self.xgb_weight = float(data.get("xgb_weight", _DEFAULT_XGB_WEIGHT))
                self.lstm_weight = float(data.get("lstm_weight", _DEFAULT_LSTM_WEIGHT))
                # Renormalise in case of rounding drift
                total = self.xgb_weight + self.lstm_weight
                if total > 0:
                    self.xgb_weight /= total
                    self.lstm_weight /= total
                logger.info(
                    "Loaded ensemble weights: XGB=%.4f, LSTM=%.4f",
                    self.xgb_weight, self.lstm_weight,
                )
            except Exception as exc:
                logger.warning(
                    "Could not load ensemble weights from %s: %s. "
                    "Using defaults XGB=%.2f / LSTM=%.2f.",
                    self._weights_path, exc,
                    _DEFAULT_XGB_WEIGHT, _DEFAULT_LSTM_WEIGHT,
                )
        else:
            logger.warning(
                "No ensemble weights file at %s. "
                "Using defaults XGB=%.2f / LSTM=%.2f.",
                self._weights_path, _DEFAULT_XGB_WEIGHT, _DEFAULT_LSTM_WEIGHT,
            )

    # ──────────────────────────────────────────────────────────────────────
    # Model availability
    # ──────────────────────────────────────────────────────────────────────

    def ensure_model(self) -> bool:
        """
        Return True if at least the XGBoost model is available.

        The LSTM is optional — if it is missing the ensemble falls back
        to XGBoost-only mode (xgb_weight=1.0).
        """
        xgb_ready = self.xgb.ensure_model()
        lstm_ready = self.lstm.ensure_model()

        if xgb_ready and not lstm_ready:
            logger.warning(
                "LSTM model unavailable — running XGBoost-only mode (weight=1.0)."
            )
        return xgb_ready

    # ──────────────────────────────────────────────────────────────────────
    # Core prediction
    # ──────────────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate blended probability predictions for the full feature matrix.

        Returns a DataFrame (same index/rows as XGBoost output) with columns:
            p_stop_loss, p_neutral, p_take_profit, predicted_label

        Blending strategy
        -----------------
        For each row that exists in *both* model outputs, the probabilities
        are combined as a weighted average:

            p_blend = xgb_weight * p_xgb + lstm_weight * p_lstm

        For rows only in the XGBoost output (first seq_length-1 rows per
        ticker, where the LSTM has no prediction), the XGBoost probabilities
        are used at full weight.
        """
        if not self.ensure_model():
            raise RuntimeError(
                "XGBoost model not available. Run 'python retrain.py' first."
            )

        # ── XGBoost predictions (always present) ────────────────────────
        xgb_df = self.xgb.predict(df)

        # ── LSTM predictions (may be empty / fewer rows) ─────────────────
        lstm_available = self.lstm.ensure_model()
        lstm_df = pd.DataFrame()

        if lstm_available:
            try:
                lstm_df = self.lstm.predict(df)
            except Exception as exc:
                logger.warning("LSTM predict failed: %s. Falling back to XGB.", exc)
                lstm_df = pd.DataFrame()

        # ── Blend ────────────────────────────────────────────────────────
        result = xgb_df.copy()
        prob_cols = ["p_stop_loss", "p_neutral", "p_take_profit"]

        if not lstm_df.empty:
            # Align LSTM probs on the shared index
            lstm_aligned = lstm_df[prob_cols].reindex(result.index)
            lstm_mask = lstm_aligned.notna().all(axis=1)  # rows where LSTM has a pred

            if lstm_mask.any():
                # Effective weights: where LSTM is present use both, else XGB-only
                w_xgb = np.where(lstm_mask, self.xgb_weight, 1.0)
                w_lstm = np.where(lstm_mask, self.lstm_weight, 0.0)

                for col in prob_cols:
                    xgb_prob = result[col].values
                    lstm_prob = lstm_aligned[col].fillna(0.0).values
                    result[col] = w_xgb * xgb_prob + w_lstm * lstm_prob

                # Re-normalise rows to sum to 1.0 (guard against floating drift)
                row_sums = result[prob_cols].sum(axis=1).replace(0, 1.0)
                for col in prob_cols:
                    result[col] = result[col] / row_sums

        # ── Re-derive predicted_label from blended probs ─────────────────
        prob_arr = result[prob_cols].values
        pred_classes = np.argmax(prob_arr, axis=1)
        inv_map = {0: -1, 1: 0, 2: 1}
        labels = np.array([inv_map[c] for c in pred_classes])

        # SL override: flag as SL when P(SL) >= threshold even if not argmax
        sl_override = prob_arr[:, 0] >= SL_DETECTION_THRESHOLD
        labels[sl_override & (pred_classes != 0)] = -1

        result["predicted_label"] = labels
        return result

    def predict_latest(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Get blended predictions for the most recent row per ticker.

        This is the primary interface consumed by ``agent/tools.py``.

        Returns
        -------
        dict[str, dict] — ticker → {p_take_profit, p_stop_loss, p_neutral, predicted_label}
        """
        pred_df = self.predict(df)
        results = {}

        if "ticker" in pred_df.columns:
            for ticker in pred_df["ticker"].unique():
                row = pred_df[pred_df["ticker"] == ticker].iloc[-1]
                results[ticker] = {
                    "p_take_profit": float(row["p_take_profit"]),
                    "p_stop_loss": float(row["p_stop_loss"]),
                    "p_neutral": float(row["p_neutral"]),
                    "predicted_label": int(row["predicted_label"]),
                }
        elif not pred_df.empty:
            row = pred_df.iloc[-1]
            results["unknown"] = {
                "p_take_profit": float(row["p_take_profit"]),
                "p_stop_loss": float(row["p_stop_loss"]),
                "p_neutral": float(row["p_neutral"]),
                "predicted_label": int(row["predicted_label"]),
            }

        return results
