"""
Directional Predictor — XGBoost Triple Barrier Model
======================================================
Predicts the probability of a stock hitting its +3% take-profit barrier
before its -2% stop-loss barrier within a 5-day horizon.

Key design decisions:
    - Target: Triple Barrier labels {-1, 0, 1} → 3-class classification
    - Validation: Walk-forward (expanding window) to prevent lookahead bias
    - Output: per-ticker probabilities (p_take_profit, p_stop_loss, p_neutral)
    - Regime is injected as a feature column
    - Asymmetric cost: missing a stop-loss event is penalized 2.5× more
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from config.settings import (
    MODELS_DIR,
    XGBOOST_PARAMS,
    RETRAIN_ACCURACY_THRESHOLD,
    SL_COST_MULTIPLIER,
    SL_DETECTION_THRESHOLD,
)

logger = logging.getLogger(__name__)

# Features to exclude from ML input
_EXCLUDE_COLS = {
    "ticker", "Open", "High", "Low", "Close", "Volume",
    "barrier_label", "regime", "regime_cluster", "regime_label",
    "returns_20d", "headline_count", "target",
    # Leaky absolute-price features (only bb_pct_b is kept)
    "bb_upper", "bb_middle", "bb_lower",
}

# Warn the user if the saved model is older than this many days
_MODEL_STALENESS_DAYS = 90


class DirectionalPredictor:
    """
    XGBoost classifier predicting Triple Barrier outcomes.

    Usage::

        # Training (run once via retrain.py):
        predictor = DirectionalPredictor()
        metrics = predictor.train(feature_matrix)

        # Prediction (fast — loads saved model):
        predictor = DirectionalPredictor()
        predictions = predictor.predict_latest(feature_matrix)
    """

    def __init__(self, model_path: Optional[Path] = None):
        self._model: Optional[XGBClassifier] = None
        self._model_path = model_path or MODELS_DIR / "xgb_triple_barrier.joblib"
        self._feature_names: List[str] = []
        self._metrics: Dict = {}

    # ------------------------------------------------------------------
    # Model loading (auto-loads saved model for predictions)
    # ------------------------------------------------------------------

    def ensure_model(self) -> bool:
        """
        Ensure a trained model is available for predictions.

        Loads the saved model from disk if it exists.
        Returns True if a model is ready, False otherwise.
        Also warns if the model is older than 90 days.
        """
        if self._model is not None:
            return True

        if self._model_path.exists():
            self._load_model()

            # Check model age and warn if stale
            age_days = (
                datetime.now().timestamp() - self._model_path.stat().st_mtime
            ) / 86400
            if age_days > _MODEL_STALENESS_DAYS:
                logger.warning(
                    "⚠️ Model is %.0f days old. Consider running "
                    "'python retrain.py' to refresh it.", age_days
                )

            return self._model is not None

        logger.warning(
            "No saved model found at %s. "
            "Run 'python retrain.py' to train one.", self._model_path
        )
        return False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
        n_splits: int = 7,
    ) -> Dict:
        """
        Train the XGBoost model using walk-forward validation.

        Uses asymmetric cost-sensitive weights that penalize missing
        stop-loss events 2.5× more than other errors. This forces the
        model to be more aggressive at detecting crashes.

        Parameters
        ----------
        df : pd.DataFrame
            Feature matrix with a ``barrier_label`` column.
        n_splits : int
            Number of time-series splits for walk-forward validation.

        Returns
        -------
        dict — metrics from the final fold (accuracy, precision, recall, f1,
               classification_report, feature_importance).
        """
        # Prepare data
        df_clean = df.dropna(subset=["barrier_label"]).copy()

        # Map labels: {-1, 0, 1} → {0, 1, 2} for XGBoost
        label_map = {-1.0: 0, 0.0: 1, 1.0: 2}
        df_clean["target"] = df_clean["barrier_label"].map(label_map)
        df_clean = df_clean.dropna(subset=["target"])
        df_clean["target"] = df_clean["target"].astype(int)

        # Select feature columns
        self._feature_names = [
            c for c in df_clean.columns
            if c not in _EXCLUDE_COLS and c != "target"
            and df_clean[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]
        ]

        X = df_clean[self._feature_names].values
        y = df_clean["target"].values

        logger.info(
            "Training XGBoost: %d samples, %d features, %d splits",
            len(X), len(self._feature_names), n_splits,
        )
        logger.info("Label distribution: %s", dict(zip(*np.unique(y, return_counts=True))))

        # Walk-forward validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        all_y_true = []
        all_y_pred = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Asymmetric cost-sensitive weights
            sample_weights = self._compute_class_weights(y_train)

            model = XGBClassifier(**XGBOOST_PARAMS, early_stopping_rounds=20)
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                sample_weight=sample_weights,
                verbose=False,
            )

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            logger.info("Fold %d: accuracy=%.4f, train=%d, test=%d",
                        fold + 1, acc, len(train_idx), len(test_idx))

            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)

        # Final model: train on all data with asymmetric weights
        sample_weights = self._compute_class_weights(y)

        self._model = XGBClassifier(**XGBOOST_PARAMS)
        self._model.fit(X, y, sample_weight=sample_weights, verbose=False)

        # Save model
        self._save_model()

        # Compute aggregate metrics
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)

        inv_map = {0: "SL_hit(-1)", 1: "Neutral(0)", 2: "TP_hit(+1)"}
        target_names = [inv_map[i] for i in sorted(inv_map.keys())]

        final_accuracy = float(accuracy_score(all_y_true, all_y_pred))

        self._metrics = {
            "accuracy": final_accuracy,
            "precision_weighted": float(precision_score(all_y_true, all_y_pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(all_y_true, all_y_pred, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(all_y_true, all_y_pred, average="weighted", zero_division=0)),
            "classification_report": classification_report(
                all_y_true, all_y_pred,
                target_names=target_names,
                zero_division=0,
                output_dict=True,
            ),
            "feature_importance": self._get_feature_importance(),
            "n_samples": len(X),
            "n_features": len(self._feature_names),
            "n_splits": n_splits,
            "needs_retrain": final_accuracy < RETRAIN_ACCURACY_THRESHOLD,
        }

        logger.info(
            "Walk-forward results: accuracy=%.4f, f1=%.4f",
            self._metrics["accuracy"], self._metrics["f1_weighted"],
        )

        return self._metrics

    # ------------------------------------------------------------------
    # Asymmetric class weights
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_class_weights(y: np.ndarray) -> np.ndarray:
        """
        Compute asymmetric sample weights that penalize missing SL events more.

        Class 0 (SL_hit) gets SL_COST_MULTIPLIER × the base weight.
        This forces the model to pay more attention to crash patterns
        and improves stop-loss recall.
        """
        classes, counts = np.unique(y, return_counts=True)
        # Base weights: inverse frequency (standard class balancing)
        base_weights = {
            c: len(y) / (len(classes) * count)
            for c, count in zip(classes, counts)
        }

        # Boost class 0 (SL_hit) by the SL cost multiplier
        if 0 in base_weights:
            base_weights[0] *= SL_COST_MULTIPLIER

        return np.array([base_weights.get(label, 1.0) for label in y])

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for the feature matrix.

        Returns a DataFrame with columns:
            - p_take_profit: P(label = +1)
            - p_stop_loss:   P(label = -1)
            - p_neutral:     P(label = 0)
            - predicted_label: threshold-calibrated label
        """
        # Auto-load saved model if needed
        if not self.ensure_model():
            raise RuntimeError(
                "No trained model available. Run 'python retrain.py' first."
            )

        # Ensure feature columns exist (handle minor mismatches gracefully)
        available = [c for c in self._feature_names if c in df.columns]
        if len(available) < len(self._feature_names):
            missing = set(self._feature_names) - set(available)
            logger.warning("Missing features for prediction: %s", missing)

        X = df[available].values
        probas = self._model.predict_proba(X)

        # Map back: class 0→SL(-1), class 1→Neutral(0), class 2→TP(+1)
        result = df.copy()
        result["p_stop_loss"] = probas[:, 0]    # P(label=-1)
        result["p_neutral"] = probas[:, 1]      # P(label=0)
        result["p_take_profit"] = probas[:, 2]  # P(label=+1)

        # Threshold-calibrated prediction:
        # If P(stop_loss) >= SL_DETECTION_THRESHOLD, flag as SL even if
        # it's not the argmax class. This catches crashes the model is
        # uncertain about (e.g., p_sl=0.30, p_neutral=0.40, p_tp=0.30).
        inv_map = {0: -1, 1: 0, 2: 1}
        pred_classes = np.argmax(probas, axis=1)
        labels = np.array([inv_map[c] for c in pred_classes])

        # Override: flag SL if probability exceeds threshold
        sl_override = probas[:, 0] >= SL_DETECTION_THRESHOLD
        labels[sl_override & (pred_classes != 0)] = -1

        result["predicted_label"] = labels

        return result

    def predict_latest(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Get predictions for the latest row per ticker.

        Returns
        -------
        dict[str, dict] — ticker → {p_take_profit, p_stop_loss, p_neutral, predicted_label}
        """
        pred_df = self.predict(df)
        results = {}

        if "ticker" in pred_df.columns:
            for ticker in pred_df["ticker"].unique():
                ticker_data = pred_df[pred_df["ticker"] == ticker].iloc[-1]
                results[ticker] = {
                    "p_take_profit": float(ticker_data["p_take_profit"]),
                    "p_stop_loss": float(ticker_data["p_stop_loss"]),
                    "p_neutral": float(ticker_data["p_neutral"]),
                    "predicted_label": int(ticker_data["predicted_label"]),
                }
        else:
            last = pred_df.iloc[-1]
            results["unknown"] = {
                "p_take_profit": float(last["p_take_profit"]),
                "p_stop_loss": float(last["p_stop_loss"]),
                "p_neutral": float(last["p_neutral"]),
                "predicted_label": int(last["predicted_label"]),
            }

        return results

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def _save_model(self):
        """Serialize model + feature names to disk."""
        payload = {
            "model": self._model,
            "feature_names": self._feature_names,
        }
        joblib.dump(payload, self._model_path)
        logger.info("Model saved to %s", self._model_path)

    def _load_model(self):
        """Load model from disk."""
        if self._model_path.exists():
            payload = joblib.load(self._model_path)
            self._model = payload["model"]
            self._feature_names = payload["feature_names"]
            logger.info("Model loaded from %s", self._model_path)
        else:
            logger.warning("No saved model found at %s", self._model_path)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def _get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance as a sorted dict."""
        if self._model is None:
            return {}
        importances = self._model.feature_importances_
        pairs = sorted(
            zip(self._feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        )
        return {name: round(float(imp), 6) for name, imp in pairs}

    @property
    def metrics(self) -> Dict:
        return self._metrics
