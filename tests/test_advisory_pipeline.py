import pytest
import json
import pandas as pd

from data.price_fetcher import PriceFetcher
from data.fundamental_fetcher import FundamentalFetcher
from data.macro_fetcher import MacroFetcher
from data.news_fetcher import NewsFetcher
from features.feature_matrix import FeatureMatrixBuilder
from features.sentiment import SentimentScorer
from intelligence.labeler import TripleBarrierLabeler
from intelligence.predictor import DirectionalPredictor
from intelligence.regime_detector import RegimeDetector
from agent.tools import (
    get_stock_prediction,
    get_market_regime,
    get_latest_sentiment,
    get_macro_snapshot,
    scan_top_momentum_stocks
)

# =====================================================================
# 1. Data Ingestion Tests
# =====================================================================

def test_price_fetcher():
    pf = PriceFetcher()
    df = pf.fetch_ohlcv("ITC.NS", period="1mo")
    assert not df.empty, "Price data should not be empty"
    assert "Close" in df.columns, "Price data must have a Close column"
    assert len(df) > 10, "Should fetch at least a few weeks of data"

def test_news_fetcher():
    nf = NewsFetcher()
    headlines = nf.fetch_for_tickers(["ITC.NS"])
    assert isinstance(headlines, list), "Headlines should be a list"
    if headlines:
        assert hasattr(headlines[0], "title"), "Headline must have a title"

def test_macro_fetcher():
    mf = MacroFetcher()
    macro = mf.fetch()
    assert macro.repo_rate > 0, "Repo rate should be a positive number"
    assert macro.usdinr > 50, "USD/INR should be realistic"

# =====================================================================
# 2. VADER Sentiment Tests
# =====================================================================

def test_sentiment_scorer():
    scorer = SentimentScorer()
    
    pos_score = scorer.score_single("Company reports record high profits and massive growth!")
    neg_score = scorer.score_single("Terrible disaster. Horrible losses and awful revenue drop.")
    neu_score = scorer.score_single("Company holds annual meeting on Tuesday.")
    
    assert pos_score > 0, "Positive headline should have score > 0"
    assert neg_score < 0, "Negative headline should have score < 0"
    assert -0.2 < neu_score < 0.2, "Neutral headline should have score near 0"

# =====================================================================
# 3. Machine Learning Engine Tests
# =====================================================================

def test_ml_engine_pipeline():
    # Fetch Data
    pf = PriceFetcher()
    df = pf.fetch_ohlcv("TCS.NS", period="3mo")
    benchmark = pf.fetch_benchmark(period="3mo")
    
    # Build Features
    builder = FeatureMatrixBuilder()
    matrix = builder.build(
        price_data={"TCS.NS": df},
        benchmark_close=benchmark["Close"],
        fundamentals={},
        macro=None,
        normalize=False
    )
    assert not matrix.empty, "Feature matrix should not be empty"
    assert "RSI_14" in matrix.columns or "rsi" in matrix.columns, "Should have technical features"
    
    # Label Data
    labeler = TripleBarrierLabeler()
    matrix = labeler.label_dataframe(matrix, close_col="Close")
    assert "barrier_label" in matrix.columns, "Should have barrier_label column"
    
    # Predict
    predictor = DirectionalPredictor()
    train_data = matrix.dropna(subset=["barrier_label"])
    if len(train_data) > 10:  # Need minimum data to train
        predictor.train(train_data, n_splits=2)
        preds = predictor.predict_latest(matrix)
        assert "TCS.NS" in preds, "Predictor should return dict with ticker"
        assert "p_take_profit" in preds["TCS.NS"], "Must output take profit probability"
        assert "p_stop_loss" in preds["TCS.NS"], "Must output stop loss probability"

# =====================================================================
# 4. Agent API Tools Tests
# =====================================================================

def test_get_stock_prediction_tool():
    result_json = get_stock_prediction("HINDUNILVR")
    data = json.loads(result_json)
    
    if "error" not in data:
        assert "advisory_snapshot" in data, "Must return an advisory snapshot"
        assert "ml_probabilities" in data, "Must include ML probabilities"
        assert "sentiment" in data, "Must include sentiment"
        assert "action_signal" in data, "Must include unified action signal"

def test_scan_top_momentum_stocks_tool():
    result_json = scan_top_momentum_stocks(3)
    data = json.loads(result_json)
    
    assert "error" not in data, "Momentum scanner failed"
    assert "top_stocks" in data, "Must return top stocks list"
    assert len(data["top_stocks"]) == 3, "Must return exactly 3 stocks"

def test_get_market_regime_tool():
    result_json = get_market_regime()
    data = json.loads(result_json)
    
    assert "error" not in data, "Regime detector failed"
    assert "current_regime" in data, "Must output the current_regime"
    assert data["current_regime"] in ["bull", "bear", "high_volatility", "Unknown"], "Invalid regime string"
