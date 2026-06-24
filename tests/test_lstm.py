"""
Tests for the LSTM and Ensemble predictors.
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intelligence.lstm_predictor import LSTMPredictor
from intelligence.ensemble_predictor import EnsemblePredictor

def _make_dummy_df(n_rows: int = 100, tickers: list = ["TCS.NS"]) -> pd.DataFrame:
    """Create a dummy feature matrix."""
    dfs = []
    for ticker in tickers:
        df = pd.DataFrame({
            "ticker": ticker,
            "Close": np.linspace(100, 200, n_rows),
            "rsi": np.random.uniform(30, 70, n_rows),
            "macd": np.random.uniform(-1, 1, n_rows),
            "returns_1d": np.random.normal(0, 0.01, n_rows),
            "barrier_label": np.random.choice([-1.0, 0.0, 1.0], size=n_rows)
        }, index=pd.date_range("2024-01-01", periods=n_rows))
        dfs.append(df)
    return pd.concat(dfs)


class TestLSTMPredictor:
    
    def test_sequence_generation_shape(self):
        """Test that sequence generation creates correct 3D tensors."""
        predictor = LSTMPredictor(seq_length=5)
        df = _make_dummy_df(n_rows=20, tickers=["A", "B"])
        
        # 20 rows per ticker, seq_length=5.
        # Number of sequences per ticker = 20 - 5 + 1 = 16
        # Total sequences = 32
        
        X, y = predictor._create_sequences(df, is_training=True)
        assert X.shape[0] == 32
        assert X.shape[1] == 5
        # Features: rsi, macd, returns_1d (3 features)
        assert X.shape[2] == 3
        
        assert y.shape[0] == 32

    def test_chronological_sequence_mapping(self):
        """Test that sequences map correctly to the chronological label without look-ahead."""
        predictor = LSTMPredictor(seq_length=3)
        # Create predictable data
        df = pd.DataFrame({
            "ticker": "A",
            "f1": [1, 2, 3, 4, 5],
            "barrier_label": [-1.0, 0.0, 1.0, -1.0, 0.0]
        })
        
        X, y = predictor._create_sequences(df, is_training=True)
        
        # Seq 1: f1 = [1, 2, 3]. Label should be the label at idx 2 (which is 1.0 -> mapped to 2)
        assert np.array_equal(X[0, :, 0], [1, 2, 3])
        assert y[0] == 2
        
        # Seq 2: f1 = [2, 3, 4]. Label should be at idx 3 (which is -1.0 -> mapped to 0)
        assert np.array_equal(X[1, :, 0], [2, 3, 4])
        assert y[1] == 0


class TestEnsemblePredictor:
    
    def test_ensemble_weights_handling(self, tmp_path):
        """Test that ensemble properly loads dynamic weights."""
        import json
        weight_file = tmp_path / "ensemble_weights.json"
        
        with open(weight_file, "w") as f:
            json.dump({"xgb_weight": 0.8, "lstm_weight": 0.2}, f)
            
        ensemble = EnsemblePredictor(weights_path=weight_file)
        assert ensemble.xgb_weight == 0.8
        assert ensemble.lstm_weight == 0.2
        
    def test_ensemble_fallback_weights(self, tmp_path):
        """Test fallback to 50/50 if weights file is missing."""
        missing_file = tmp_path / "missing.json"
        ensemble = EnsemblePredictor(weights_path=missing_file)
        assert ensemble.xgb_weight == 0.5
        assert ensemble.lstm_weight == 0.5
