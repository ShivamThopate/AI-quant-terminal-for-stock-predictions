"""
Model Retraining Script — XGBoost + LSTM Ensemble
====================================================
Fetches historical data for a diversified set of Nifty stocks, builds the
full feature matrix, trains both models, evaluates them on a chronological
holdout set, and serialises the dynamic ensemble weight ratios.

Pipeline
--------
  1. Fetch OHLCV + benchmark + fundamentals + macro
  2. Build feature matrix (normalize=False so labeler sees raw prices)
  3. Inject market regime as an encoded feature column
  4. Label with Triple Barrier (+2% TP / -2% SL / 5-day horizon)
  5. Chronological 80/20 split for holdout evaluation
  6. Train XGBoost (walk-forward CV on 80% train set)
  7. Train LSTM (chronological 80/20 on sequence-level data)
  8. Evaluate both models on the holdout 20%
  9. Compute accuracy-based ensemble weights → save to ensemble_weights.json

Usage:
    python retrain.py

Run from the ``algo_trading_system/`` directory or any parent directory —
sys.path is adjusted automatically.
"""

import sys
import os
import io
import json
import logging
import time
from pathlib import Path

# ── Path bootstrap ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Windows console UTF-8 ────────────────────────────────────────────────────
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(stream=sys.stderr)],
)
logger = logging.getLogger("retrain")


# ── Training Universe ────────────────────────────────────────────────────────
# Diversified set of Nifty 50 stocks spanning multiple sectors
TRAINING_TICKERS = [
    "RELIANCE.NS",    # Oil & Gas / Conglomerate
    "TCS.NS",         # IT
    "HDFCBANK.NS",    # Banking
    "INFY.NS",        # IT
    "ICICIBANK.NS",   # Banking
    "SBIN.NS",        # PSU Banking
    "BAJFINANCE.NS",  # NBFC
    "BHARTIARTL.NS",  # Telecom
    "ITC.NS",         # FMCG
    "KOTAKBANK.NS",   # Banking
    "LT.NS",          # Infrastructure
    "HINDUNILVR.NS",  # FMCG
    "MARUTI.NS",      # Auto
    "SUNPHARMA.NS",   # Pharma
    "AXISBANK.NS",    # Banking
    "M&M.NS",         # Auto
    "TITAN.NS",       # Consumer
    "ASIANPAINT.NS",  # Consumer
    "BAJAJFINSV.NS",  # Financials
    "HCLTECH.NS",     # IT
]


# ── Holdout evaluation helper ────────────────────────────────────────────────

def _evaluate_xgb_on_holdout(predictor, X_holdout, y_holdout):
    """
    Run the trained XGBoost model on pre-split holdout arrays.

    Returns accuracy as a float.
    """
    import numpy as np
    from sklearn.metrics import accuracy_score

    if not predictor.ensure_model():
        return 0.0

    probas = predictor._model.predict_proba(X_holdout)
    preds = probas.argmax(axis=1)
    return float(accuracy_score(y_holdout, preds))


def _evaluate_lstm_on_holdout(lstm_predictor, X_holdout_np, y_holdout_np):
    """
    Run the trained LSTM on pre-built 3D numpy arrays (sequences).

    Returns accuracy as a float.
    """
    import numpy as np
    import torch

    if not lstm_predictor.ensure_model():
        return 0.0

    device = lstm_predictor._device
    X_t = torch.tensor(X_holdout_np, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_holdout_np, dtype=torch.long).to(device)

    lstm_predictor._model.eval()
    with torch.no_grad():
        logits = lstm_predictor._model(X_t)
        preds = logits.argmax(dim=1)
        acc = (preds == y_t).float().mean().item()

    return float(acc)


# ── Main retraining pipeline ─────────────────────────────────────────────────

def retrain():
    """Run the full retraining pipeline (XGBoost + LSTM Ensemble)."""
    import numpy as np

    print("\n" + "=" * 65)
    print("  XGBOOST + LSTM ENSEMBLE RETRAINING")
    print("=" * 65)

    start_time = time.time()

    from config.settings import TRAINING_DATA_PERIOD, MODELS_DIR
    from data.price_fetcher import PriceFetcher
    from data.fundamental_fetcher import FundamentalFetcher
    from data.macro_fetcher import MacroFetcher
    from features.feature_matrix import FeatureMatrixBuilder
    from intelligence.regime_detector import RegimeDetector
    from intelligence.labeler import TripleBarrierLabeler
    from intelligence.predictor import DirectionalPredictor
    from intelligence.lstm_predictor import LSTMPredictor

    pf = PriceFetcher()
    ff = FundamentalFetcher()
    mf = MacroFetcher()
    builder = FeatureMatrixBuilder()
    regime_detector = RegimeDetector()
    labeler = TripleBarrierLabeler()

    # ── Step 1: Fetch price data ─────────────────────────────────────────
    print(f"\n[1/6] Fetching {TRAINING_DATA_PERIOD} of OHLCV data for "
          f"{len(TRAINING_TICKERS)} stocks...")

    price_data = {}
    for ticker in TRAINING_TICKERS:
        try:
            df = pf.fetch_ohlcv(ticker, period=TRAINING_DATA_PERIOD)
            if not df.empty:
                price_data[ticker] = df
                print(f"  ✓ {ticker}: {len(df)} days")
            else:
                print(f"  ✗ {ticker}: no data returned")
        except Exception as exc:
            print(f"  ✗ {ticker}: {exc}")

    if len(price_data) < 5:
        print("\n❌ Fewer than 5 stocks loaded. Aborting.")
        sys.exit(1)

    print(f"\n  → {len(price_data)} stocks loaded successfully")

    # ── Step 2: Fetch benchmark, fundamentals, macro ─────────────────────
    print(f"\n[2/6] Fetching benchmark (Nifty 50) + macro + fundamentals...")

    benchmark = pf.fetch_benchmark(period=TRAINING_DATA_PERIOD)
    if benchmark.empty:
        print("❌ Could not fetch Nifty 50 benchmark data. Aborting.")
        sys.exit(1)
    print(f"  ✓ Nifty 50: {len(benchmark)} days")

    fundamentals = ff.fetch_multiple(list(price_data.keys()))
    print(f"  ✓ Fundamentals: {len(fundamentals)} stocks")

    macro = mf.fetch()
    print(
        f"  ✓ Macro: Repo={macro.repo_rate}%, "
        f"CPI={macro.cpi_inflation}%, USD/INR={macro.usdinr}"
    )

    # ── Step 3: Build feature matrix (raw prices for labeling) ───────────
    print(
        f"\n[3/6] Building feature matrix "
        f"(technicals + fundamentals + macro + lag features)..."
    )

    # normalize=False: keep raw Close for Triple Barrier labeling.
    # Both XGBoost and LSTM will see normalised features later (XGB normalises
    # internally via retrain; LSTM sequences are z-scored in FeatureMatrixBuilder
    # when normalize=True is used for inference).
    matrix = builder.build(
        price_data=price_data,
        benchmark_close=benchmark["Close"],
        fundamentals=fundamentals,
        macro=macro,
        normalize=False,
    )

    # Inject market regime as an encoded feature column
    regime_df = regime_detector.fit_predict(benchmark["Close"])
    regime_map = regime_df["regime"].to_dict()
    matrix["regime_label"] = matrix.index.map(lambda dt: regime_map.get(dt, "bull"))
    regime_encoding = {"bull": 1, "bear": -1, "high_volatility": 0}
    matrix["regime_encoded"] = (
        matrix["regime_label"].map(regime_encoding).fillna(0).astype(int)
    )

    print(f"  → Matrix shape: {matrix.shape[0]} rows × {matrix.shape[1]} columns")
    print(f"  → Tickers: {matrix['ticker'].nunique()}")

    # ── Step 4: Triple Barrier labeling ──────────────────────────────────
    print(
        f"\n[4/6] Labeling with Triple Barrier "
        f"(+2% TP / -2% SL / 5-day horizon)..."
    )

    matrix = labeler.label_dataframe(matrix, close_col="Close")
    valid_labels = matrix["barrier_label"].dropna()

    if len(valid_labels) < 100:
        print(f"❌ Only {len(valid_labels)} labeled samples — need ≥ 100. Aborting.")
        sys.exit(1)

    counts = valid_labels.value_counts()
    print(
        f"  → Labels: TP(+1)={counts.get(1.0, 0)}, "
        f"SL(-1)={counts.get(-1.0, 0)}, "
        f"Neutral(0)={counts.get(0.0, 0)}"
    )
    print(f"  → Total labeled samples: {len(valid_labels)}")

    # ── Step 5: Chronological 80/20 split for holdout evaluation ─────────
    # Sort by date, then split deterministically so the holdout is always
    # the *most recent* 20% of data — the hardest test for a time-series model.
    print(f"\n[5/6] Creating chronological 80/20 holdout split...")

    df_labeled = matrix.dropna(subset=["barrier_label"]).copy()
    df_labeled = df_labeled.sort_index()

    split_row = int(len(df_labeled) * 0.8)
    df_train_full = df_labeled.iloc[:split_row]
    df_holdout = df_labeled.iloc[split_row:]

    print(f"  → Train: {len(df_train_full)} rows | Holdout: {len(df_holdout)} rows")

    # ── Step 6: Train XGBoost ─────────────────────────────────────────────
    print(f"\n[6a/6] Training XGBoost (7-fold walk-forward CV on train set)...")
    print(f"  → SL events penalised asymmetrically (SL_COST_MULTIPLIER)")

    xgb_predictor = DirectionalPredictor()
    xgb_metrics = xgb_predictor.train(df_train_full, n_splits=7)

    print(
        f"  → Walk-forward accuracy: {xgb_metrics.get('accuracy', 0):.4f}  "
        f"| F1: {xgb_metrics.get('f1_weighted', 0):.4f}"
    )

    # Evaluate XGBoost on chronological holdout
    # Build holdout feature arrays matching XGBoost's stored feature names
    xgb_exclude = getattr(xgb_predictor, '_feature_names', None)
    label_map_int = {-1.0: 0, 0.0: 1, 1.0: 2}
    df_holdout_labeled = df_holdout.copy()
    df_holdout_labeled["target"] = df_holdout_labeled["barrier_label"].map(label_map_int)
    df_holdout_labeled = df_holdout_labeled.dropna(subset=["target"])
    df_holdout_labeled["target"] = df_holdout_labeled["target"].astype(int)

    if xgb_exclude:
        holdout_features = [c for c in xgb_exclude if c in df_holdout_labeled.columns]
        X_holdout_xgb = df_holdout_labeled[holdout_features].values
        y_holdout_xgb = df_holdout_labeled["target"].values
        xgb_holdout_acc = _evaluate_xgb_on_holdout(xgb_predictor, X_holdout_xgb, y_holdout_xgb)
    else:
        xgb_holdout_acc = xgb_metrics.get("accuracy", 0.0)

    print(f"  → Holdout accuracy (XGBoost): {xgb_holdout_acc:.4f}")

    # ── Step 7: Train LSTM ────────────────────────────────────────────────
    print(f"\n[6b/6] Training PyTorch LSTM (30-day chronological sequences)...")

    lstm_predictor = LSTMPredictor()
    lstm_metrics = lstm_predictor.train(
        df_train_full,
        epochs=100,
        batch_size=64,
        lr=1e-3,
        val_split=0.20,
        patience=15,
    )

    lstm_val_acc = lstm_metrics.get("accuracy", 0.0)
    print(
        f"  → Best validation accuracy (LSTM): {lstm_val_acc:.4f}"
    )

    # Evaluate LSTM on chronological holdout using pre-built sequences
    lstm_predictor._feature_names = lstm_predictor._feature_names or []
    if lstm_predictor._feature_names and len(df_holdout) >= lstm_predictor._seq_length:
        X_holdout_lstm, y_holdout_lstm = lstm_predictor._build_sequences(
            df_holdout_labeled, is_training=True
        )
        if len(X_holdout_lstm) >= 1:
            lstm_holdout_acc = _evaluate_lstm_on_holdout(
                lstm_predictor, X_holdout_lstm, y_holdout_lstm
            )
        else:
            lstm_holdout_acc = lstm_val_acc
    else:
        lstm_holdout_acc = lstm_val_acc

    print(f"  → Holdout accuracy (LSTM):    {lstm_holdout_acc:.4f}")

    # ── Step 8: Compute & persist dynamic ensemble weights ────────────────
    print(f"\n[Ensemble] Computing accuracy-based dynamic weights...")

    total_acc = xgb_holdout_acc + lstm_holdout_acc
    if total_acc > 0:
        xgb_weight = xgb_holdout_acc / total_acc
        lstm_weight = lstm_holdout_acc / total_acc
    else:
        xgb_weight, lstm_weight = 0.6, 0.4  # sensible default if both score 0

    weights_path = MODELS_DIR / "ensemble_weights.json"
    weights_payload = {
        "xgb_weight": round(xgb_weight, 6),
        "lstm_weight": round(lstm_weight, 6),
        "xgb_holdout_acc": round(xgb_holdout_acc, 6),
        "lstm_holdout_acc": round(lstm_holdout_acc, 6),
    }
    with open(weights_path, "w", encoding="utf-8") as fh:
        json.dump(weights_payload, fh, indent=2)

    elapsed = time.time() - start_time

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  TRAINING COMPLETE — XGBoost + LSTM Ensemble")
    print("=" * 65)

    print(f"\n  [XGBoost]")
    print(f"    Walk-forward accuracy:  {xgb_metrics.get('accuracy', 0):.4f}")
    print(f"    Walk-forward F1:        {xgb_metrics.get('f1_weighted', 0):.4f}")
    print(f"    Holdout accuracy:       {xgb_holdout_acc:.4f}")

    print(f"\n  [LSTM]")
    print(f"    Best val accuracy:      {lstm_val_acc:.4f}")
    print(f"    Holdout accuracy:       {lstm_holdout_acc:.4f}")
    print(f"    Sequences trained on:   {lstm_metrics.get('n_sequences', 'N/A')}")

    print(f"\n  [Ensemble Weights]")
    print(f"    XGBoost weight:         {xgb_weight:.4f}")
    print(f"    LSTM weight:            {lstm_weight:.4f}")
    print(f"    Saved to:               {weights_path}")

    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  All future predictions will use the Ensemble model automatically.\n")


if __name__ == "__main__":
    retrain()
