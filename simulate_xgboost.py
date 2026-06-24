import pandas as pd
from data.price_fetcher import PriceFetcher
from data.fundamental_fetcher import FundamentalFetcher
from data.macro_fetcher import MacroFetcher
from features.feature_matrix import FeatureMatrixBuilder
from intelligence.labeler import TripleBarrierLabeler
from intelligence.predictor import DirectionalPredictor
import json

def run_simulation(ticker="RELIANCE.NS"):
    print(f"\n--- 🚀 SIMULATING XGBOOST ENGINE FOR {ticker} ---\n")
    
    # 1. FETCH RAW DATA
    print("STEP 1: Fetching Raw Market Data...")
    pf = PriceFetcher()
    prices = pf.fetch_ohlcv(ticker, period="6mo")
    benchmark = pf.fetch_benchmark(period="6mo")
    print(f"-> Fetched {len(prices)} days of price data.")
    
    # 2. BUILD FEATURE MATRIX
    print("\nSTEP 2: Building the Feature Matrix (The 'Inputs')...")
    builder = FeatureMatrixBuilder()
    fundamentals = FundamentalFetcher().fetch_multiple([ticker])
    macro = MacroFetcher().fetch()
    
    matrix = builder.build(
        price_data={ticker: prices},
        benchmark_close=benchmark["Close"],
        fundamentals=fundamentals,
        macro=macro,
        normalize=False
    )
    
    print("-> The engine calculates dozens of technical and fundamental features.")
    print("-> Sample Features for the latest day:")
    latest_features = matrix.iloc[-1]
    
    # Dynamically print 5 technical features instead of hardcoding names
    tech_cols = [c for c in matrix.columns if c not in ["Open", "High", "Low", "Close", "Volume", "ticker"]]
    for col in ["Close"] + tech_cols[:4]:
        val = latest_features.get(col, 0.0)
        print(f"   - {col}: {val:.4f}")
    
    # 3. LABEL HISTORICAL DATA
    print("\nSTEP 3: Labeling History (The 'Answers')...")
    labeler = TripleBarrierLabeler()
    matrix = labeler.label_dataframe(matrix, close_col="Close")
    
    print("-> The Triple Barrier algorithm looks at the past 6 months.")
    print("-> It asks: If I bought on this day, did it hit +3% first (1), -2% first (-1), or time out (0)?")
    counts = matrix['barrier_label'].value_counts()
    print(f"   Historical Results: {counts.get(1.0, 0)} Take-Profits, {counts.get(-1.0, 0)} Stop-Losses, {counts.get(0.0, 0)} Timeouts.")
    
    # 4. TRAIN XGBOOST
    print("\nSTEP 4: Training XGBoost...")
    predictor = DirectionalPredictor()
    # Drop rows where we don't know the future yet (the last 5 days)
    train_data = matrix.dropna(subset=['barrier_label'])
    print(f"-> Feeding {len(train_data)} labeled examples into the XGBoost Decision Tree algorithm...")
    predictor.train(train_data, n_splits=2)
    print("-> Model trained! It has learned which combinations of RSI, MACD, and Volatility lead to breakouts vs crashes.")
    
    # 5. PREDICT THE FUTURE
    print("\nSTEP 5: Predicting Tomorrow...")
    predictions = predictor.predict_latest(matrix)
    pred = predictions[ticker]
    
    print(f"-> Based on today's specific features (RSI={latest_features['rsi']:.2f}, etc.), XGBoost outputs:")
    print(f"   🟢 Probability of +3% Take-Profit : {pred['p_take_profit']*100:.2f}%")
    print(f"   🔴 Probability of -2% Stop-Loss   : {pred['p_stop_loss']*100:.2f}%")
    print(f"   ⚪ Probability of Choppy sideways : {pred['p_neutral']*100:.2f}%")
    
    if pred['p_take_profit'] > pred['p_stop_loss']:
        print("\n=> XGBOOST RAW VERDICT: BUY (Upside probability is higher)")
    else:
        print("\n=> XGBOOST RAW VERDICT: AVOID/SELL (Downside risk is higher)")
        
if __name__ == "__main__":
    run_simulation()
