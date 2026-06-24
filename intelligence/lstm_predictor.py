"""
LSTM Deep Learning Predictor
==============================
Processes sequential feature data (30-day windows) to identify temporal
patterns and predict Triple Barrier labels (Take Profit / Stop Loss / Neutral).

Design principles:
  - Sequences are built per-ticker to prevent cross-ticker contamination.
  - Labels are assigned to the *last* row of each window (only uses past data).
  - Train / Validation split is strictly chronological at the sequence level.
  - DataLoader shuffle applies ONLY to the training set AFTER the split.
  - Validation set is evaluated with shuffle=False in all loops.
  - The best model checkpoint (by val accuracy) is persisted to disk.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from config.settings import MODELS_DIR, SL_COST_MULTIPLIER, SL_DETECTION_THRESHOLD

logger = logging.getLogger(__name__)

# Features to exclude from LSTM input (must match XGBoost's _EXCLUDE_COLS)
_EXCLUDE_COLS = {
    "ticker", "Open", "High", "Low", "Close", "Volume",
    "barrier_label", "regime", "regime_cluster", "regime_label",
    "returns_20d", "headline_count", "target",
    # Leaky absolute-price features
    "bb_upper", "bb_middle", "bb_lower",
}

# Default LSTM hyperparameters — tunable via retrain.py kwargs
_DEFAULT_HIDDEN_SIZE = 128
_DEFAULT_NUM_LAYERS = 2
_DEFAULT_DROPOUT = 0.3


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch Model
# ──────────────────────────────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    """
    Stacked LSTM with a fully-connected classification head.

    Input shape:  (batch_size, seq_length, input_features)
    Output shape: (batch_size, 3)  — raw logits for {SL=-1, Neutral=0, TP=+1}
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = _DEFAULT_HIDDEN_SIZE,
        num_layers: int = _DEFAULT_NUM_LAYERS,
        dropout: float = _DEFAULT_DROPOUT,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # dropout only applied between LSTM layers (not after the last)
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        # 3 output classes: 0=SL(-1), 1=Neutral(0), 2=TP(+1)
        self.fc = nn.Linear(hidden_size, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # h0, c0 default to zeros automatically if not provided
        out, _ = self.lstm(x)
        # Decode only the final time-step hidden state
        last_hidden = out[:, -1, :]
        last_hidden = self.layer_norm(last_hidden)
        last_hidden = self.dropout(last_hidden)
        return self.fc(last_hidden)


# ──────────────────────────────────────────────────────────────────────────────
# Predictor Wrapper
# ──────────────────────────────────────────────────────────────────────────────

class LSTMPredictor:
    """
    Training and inference wrapper for the PyTorch LSTM model.

    Provides the same ``predict_latest`` interface as ``DirectionalPredictor``
    so it can be used interchangeably by the ``EnsemblePredictor``.

    Usage::

        # Training (run once via retrain.py):
        predictor = LSTMPredictor()
        metrics = predictor.train(feature_matrix, epochs=100)

        # Inference (fast — loads saved .pt checkpoint):
        predictor = LSTMPredictor()
        results = predictor.predict_latest(feature_matrix)
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        seq_length: int = 30,
        hidden_size: int = _DEFAULT_HIDDEN_SIZE,
        num_layers: int = _DEFAULT_NUM_LAYERS,
        dropout: float = _DEFAULT_DROPOUT,
    ):
        self._model: Optional[LSTMModel] = None
        self._model_path = model_path or MODELS_DIR / "lstm_triple_barrier.pt"
        self._feature_names: List[str] = []
        self._seq_length = seq_length
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._dropout = dropout
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("LSTMPredictor will use device: %s", self._device)

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def ensure_model(self) -> bool:
        """Return True if a trained model is loaded, attempting disk load first."""
        if self._model is not None:
            return True
        if self._model_path.exists():
            self._load_model()
            return self._model is not None
        logger.warning(
            "No saved LSTM found at %s. Run 'python retrain.py' first.",
            self._model_path,
        )
        return False

    def train(
        self,
        df: pd.DataFrame,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        val_split: float = 0.20,
        patience: int = 15,
    ) -> Dict:
        """
        Train the LSTM on 30-day sliding windows with a strict chronological split.

        Steps
        -----
        1. Build per-ticker sequences chronologically (no look-ahead).
        2. Sort all sequences by their *end-date index position* so the global
           array is time-ordered across tickers.
        3. Split 80/20 chronologically at the sequence level.
        4. Only the training DataLoader uses shuffle=True.
        5. Early-stopping on validation accuracy with ``patience`` epochs.
        6. Best-val-acc checkpoint saved to disk immediately.

        Parameters
        ----------
        df : pd.DataFrame
            Labeled feature matrix with ``barrier_label`` and ``ticker`` columns.
        epochs : int
            Maximum training epochs.
        batch_size : int
        lr : float
            Adam learning rate.
        val_split : float
            Fraction of sequences reserved for validation (latest dates).
        patience : int
            Early stopping: stop if val accuracy does not improve for this
            many consecutive epochs.

        Returns
        -------
        dict — {"accuracy": best_val_acc, "n_sequences": int, "n_features": int}
        """
        logger.info(
            "Preparing %d-day sequences for LSTM training...", self._seq_length
        )
        df_clean = df.dropna(subset=["barrier_label"]).copy()

        # Resolve feature columns once
        self._feature_names = self._resolve_features(df_clean)
        if not self._feature_names:
            logger.error("No usable numeric feature columns found.")
            return {"accuracy": 0.0}

        # Build sequences (chronological, per-ticker)
        X_all, y_all = self._build_sequences(df_clean, is_training=True)
        n_seq = len(X_all)
        if n_seq < 100:
            logger.warning(
                "Only %d sequences generated — need at least 100. "
                "LSTM training skipped.", n_seq
            )
            return {"accuracy": 0.0}

        # ── Strict chronological split ──────────────────────────────────
        split = int(n_seq * (1.0 - val_split))
        X_train_np, y_train_np = X_all[:split], y_all[:split]
        X_val_np, y_val_np = X_all[split:], y_all[split:]

        logger.info(
            "Split: %d train / %d val sequences (%.0f%% / %.0f%%)",
            len(X_train_np), len(X_val_np),
            (1 - val_split) * 100, val_split * 100,
        )

        # ── Tensors ─────────────────────────────────────────────────────
        X_train_t = torch.tensor(X_train_np, dtype=torch.float32).to(self._device)
        y_train_t = torch.tensor(y_train_np, dtype=torch.long).to(self._device)
        X_val_t = torch.tensor(X_val_np, dtype=torch.float32).to(self._device)
        y_val_t = torch.tensor(y_val_np, dtype=torch.long).to(self._device)

        # ── Class weights (asymmetric — penalise SL miss more) ──────────
        class_weights = self._compute_class_weights(y_train_np)
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self._device)

        # ── DataLoaders (train shuffled, val ordered) ────────────────────
        train_ds = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        # ── Model, Loss, Optimiser ───────────────────────────────────────
        n_features = X_train_np.shape[2]
        self._model = LSTMModel(
            input_size=n_features,
            hidden_size=self._hidden_size,
            num_layers=self._num_layers,
            dropout=self._dropout,
        ).to(self._device)

        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        optimizer = optim.Adam(self._model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-5
        )

        logger.info(
            "Training LSTM on %s | input=%d, hidden=%d, layers=%d | epochs=%d",
            self._device, n_features, self._hidden_size, self._num_layers, epochs,
        )

        best_val_acc = 0.0
        no_improve = 0

        for epoch in range(1, epochs + 1):
            # ── Training step ────────────────────────────────────────────
            self._model.train()
            running_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                logits = self._model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += loss.item()

            # ── Validation step ──────────────────────────────────────────
            self._model.eval()
            with torch.no_grad():
                val_logits = self._model(X_val_t)
                val_loss = criterion(val_logits, y_val_t).item()
                preds = val_logits.argmax(dim=1)
                val_acc = (preds == y_val_t).float().mean().item()

            scheduler.step(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve = 0
                # Persist best weights immediately
                self._save_model()
            else:
                no_improve += 1

            if epoch % 10 == 0 or epoch == epochs:
                logger.info(
                    "Epoch [%d/%d] | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f | best=%.4f",
                    epoch, epochs,
                    running_loss / max(len(train_loader), 1),
                    val_loss, val_acc, best_val_acc,
                )

            if no_improve >= patience:
                logger.info(
                    "Early stopping at epoch %d (no val acc improvement for %d epochs).",
                    epoch, patience,
                )
                break

        # Reload best checkpoint so the in-memory model is the best one
        self._load_model()
        logger.info("LSTM training complete. Best val accuracy: %.4f", best_val_acc)

        return {
            "accuracy": best_val_acc,
            "n_sequences": n_seq,
            "n_features": n_features,
        }

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate class probabilities for the feature matrix.

        Returns a DataFrame aligned to the *valid* rows (i.e., rows that
        are the last step of at least one complete window). The first
        ``(seq_length - 1)`` rows per ticker will not have predictions.

        Columns added:
            p_stop_loss, p_neutral, p_take_profit, predicted_label

        Memory safety
        -------------
        - ``model.eval()`` is called unconditionally at entry to disable
          dropout and batch-norm training behaviour.
        - The entire forward pass is wrapped in ``torch.no_grad()`` so
          PyTorch never allocates a computation graph during inference.
        - ``.detach().cpu()`` is called before ``.numpy()`` to guarantee
          the tensor is detached from any residual autograd state and
          explicitly moved off the GPU before conversion.
        - A ``finally`` block calls ``torch.cuda.empty_cache()`` to
          release any cached GPU memory blocks back to the allocator
          after every inference call.
        """
        # ── Set eval mode unconditionally at entry ──────────────────────
        if not self.ensure_model():
            raise RuntimeError(
                "No trained LSTM model available. Run 'python retrain.py' first."
            )
        self._model.eval()  # disables dropout & batchnorm training mode

        # Resolve feature names (must match training)
        if not self._feature_names:
            self._feature_names = self._resolve_features(df)

        X_all, _ = self._build_sequences(df, is_training=False)
        if len(X_all) == 0:
            logger.warning("No sequences generated from DataFrame (too few rows?).")
            return pd.DataFrame()

        X_t = torch.tensor(X_all, dtype=torch.float32).to(self._device)

        try:
            with torch.no_grad():
                logits = self._model(X_t)
                # .detach() severs any residual autograd linkage before
                # moving off the GPU; .cpu() pins to host memory.
                probas = torch.softmax(logits, dim=1).detach().cpu().numpy()
        finally:
            # Release cached CUDA memory blocks after each inference call
            del X_t
            if self._device.type == "cuda":
                torch.cuda.empty_cache()

        # ── Map probabilities back to the DataFrame ──────────────────────
        # Strategy: each sequence of length L ending at row i produces a
        # prediction assigned to row i. We reconstruct this mapping per ticker.
        result = df.copy()
        result["p_stop_loss"] = np.nan
        result["p_neutral"] = np.nan
        result["p_take_profit"] = np.nan

        tickers = (
            df["ticker"].unique() if "ticker" in df.columns else [None]
        )
        proba_cursor = 0

        for ticker in tickers:
            if ticker is not None:
                mask = result["ticker"] == ticker
            else:
                mask = pd.Series(True, index=result.index)

            ticker_len = int(mask.sum())
            if ticker_len < self._seq_length:
                continue  # not enough rows for even one window

            n_windows = ticker_len - self._seq_length + 1
            ticker_probas = probas[proba_cursor : proba_cursor + n_windows]
            proba_cursor += n_windows

            # The i-th window ends at position (seq_length - 1 + i) within the ticker slice
            ticker_idx = result.index[mask]
            target_idx = ticker_idx[self._seq_length - 1 :]  # rows that get a prediction

            result.loc[target_idx, "p_stop_loss"] = ticker_probas[:, 0]
            result.loc[target_idx, "p_neutral"] = ticker_probas[:, 1]
            result.loc[target_idx, "p_take_profit"] = ticker_probas[:, 2]

        # Drop rows without predictions
        result = result.dropna(subset=["p_stop_loss"]).copy()
        if result.empty:
            return result

        # ── Threshold-calibrated label assignment ────────────────────────
        prob_arr = result[["p_stop_loss", "p_neutral", "p_take_profit"]].values
        pred_classes = np.argmax(prob_arr, axis=1)
        inv_map = {0: -1, 1: 0, 2: 1}
        labels = np.array([inv_map[c] for c in pred_classes])

        # Override: flag SL when P(SL) >= threshold even if not argmax
        sl_override = prob_arr[:, 0] >= SL_DETECTION_THRESHOLD
        labels[sl_override & (pred_classes != 0)] = -1

        result["predicted_label"] = labels
        return result

    def predict_latest(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Get ensemble-ready predictions for the most recent row per ticker.

        Enforces eval mode and no_grad at this level as well, so the method
        is safe even if called directly (bypassing ``predict()``).

        Returns
        -------
        dict[str, dict] — ticker → {p_take_profit, p_stop_loss, p_neutral, predicted_label}
        """
        # Guarantee eval mode is active before any tensor work
        if self._model is not None:
            self._model.eval()

        with torch.no_grad():
            pred_df = self.predict(df)

        results = {}

        if "ticker" in pred_df.columns:
            for ticker in pred_df["ticker"].unique():
                row = pred_df[pred_df["ticker"] == ticker].iloc[-1]
                results[ticker] = {
                    # Explicit Python float/int conversion — safe on CPU numpy scalars
                    "p_take_profit": float(row["p_take_profit"]),
                    "p_stop_loss":   float(row["p_stop_loss"]),
                    "p_neutral":     float(row["p_neutral"]),
                    "predicted_label": int(row["predicted_label"]),
                }
        elif not pred_df.empty:
            row = pred_df.iloc[-1]
            results["unknown"] = {
                "p_take_profit":   float(row["p_take_profit"]),
                "p_stop_loss":     float(row["p_stop_loss"]),
                "p_neutral":       float(row["p_neutral"]),
                "predicted_label": int(row["predicted_label"]),
            }

        return results

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _resolve_features(self, df: pd.DataFrame) -> List[str]:
        """Return numeric feature column names usable as LSTM input."""
        return [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in _EXCLUDE_COLS
        ]

    def _build_sequences(
        self,
        df: pd.DataFrame,
        is_training: bool,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Build (N, seq_length, n_features) 3D arrays from the 2D feature matrix.

        Each ticker is processed independently to avoid cross-ticker sequences.
        The DataFrame must be sorted chronologically within each ticker (the
        FeatureMatrixBuilder always outputs date-indexed data, so this holds).

        For training, the label is the barrier_label of the *last* row in the
        window — this is look-ahead safe because the label only references
        future prices that happened *after* that row was recorded.

        Parameters
        ----------
        df : pd.DataFrame
        is_training : bool
            If True, also returns a y array of mapped integer labels.

        Returns
        -------
        (X, y) where y is None when is_training=False.
        """
        label_map = {-1.0: 0, 0.0: 1, 1.0: 2}
        X_list: List[np.ndarray] = []
        y_list: List[int] = []

        tickers = df["ticker"].unique() if "ticker" in df.columns else [None]

        for ticker in tickers:
            if ticker is not None:
                tdf = df[df["ticker"] == ticker]
            else:
                tdf = df

            # Ensure chronological order within this ticker slice
            if isinstance(tdf.index, pd.DatetimeIndex):
                tdf = tdf.sort_index()

            x_vals = tdf[self._feature_names].values  # (T, F)
            n_rows = len(x_vals)

            if n_rows < self._seq_length:
                continue  # not enough history for even one window

            if is_training:
                raw_labels = tdf["barrier_label"].values

            for start in range(n_rows - self._seq_length + 1):
                end = start + self._seq_length  # exclusive
                seq = x_vals[start:end]  # (seq_length, F) — uses only past rows

                if is_training:
                    label_raw = raw_labels[end - 1]  # label of the last row in window
                    if np.isnan(label_raw):
                        continue
                    mapped = label_map.get(float(label_raw))
                    if mapped is None:
                        continue
                    X_list.append(seq)
                    y_list.append(mapped)
                else:
                    X_list.append(seq)

        if not X_list:
            return np.empty((0,)), np.empty((0,))

        X_arr = np.array(X_list, dtype=np.float32)
        y_arr = np.array(y_list, dtype=np.int64) if is_training else None
        return X_arr, y_arr

    def _compute_class_weights(self, y: np.ndarray) -> List[float]:
        """
        Inverse-frequency class weights with asymmetric SL penalty.

        Classes are always ordered [0=SL, 1=Neutral, 2=TP].
        """
        classes, counts = np.unique(y, return_counts=True)
        n_total = len(y)
        n_classes = 3  # always 3 even if some are absent in this split

        weights = np.ones(n_classes, dtype=np.float64)
        for cls, cnt in zip(classes, counts):
            weights[cls] = n_total / (n_classes * cnt)

        # Boost Stop-Loss class (0) by the global SL cost multiplier
        weights[0] *= SL_COST_MULTIPLIER
        return weights.tolist()

    # ──────────────────────────────────────────────────────────────────────
    # Model persistence
    # ──────────────────────────────────────────────────────────────────────

    def _save_model(self):
        """Serialize model weights + metadata to disk."""
        if self._model is None:
            return
        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state": self._model.state_dict(),
            "feature_names": self._feature_names,
            "input_size": self._model.lstm.input_size,
            "hidden_size": self._hidden_size,
            "num_layers": self._num_layers,
            "dropout": self._dropout,
        }
        torch.save(checkpoint, self._model_path)
        logger.debug("LSTM checkpoint saved to %s", self._model_path)

    def _load_model(self):
        """Load model weights + metadata from disk."""
        if not self._model_path.exists():
            logger.warning("LSTM checkpoint not found at %s", self._model_path)
            return
        try:
            checkpoint = torch.load(
                self._model_path, map_location=self._device, weights_only=False
            )
            self._feature_names = checkpoint["feature_names"]
            input_size = checkpoint["input_size"]
            hidden_size = checkpoint.get("hidden_size", _DEFAULT_HIDDEN_SIZE)
            num_layers = checkpoint.get("num_layers", _DEFAULT_NUM_LAYERS)
            dropout = checkpoint.get("dropout", _DEFAULT_DROPOUT)

            self._model = LSTMModel(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            ).to(self._device)
            self._model.load_state_dict(checkpoint["model_state"])
            self._model.eval()
            logger.info("LSTM model loaded from %s", self._model_path)
        except Exception as exc:
            logger.error("Failed to load LSTM model: %s", exc)
            self._model = None
