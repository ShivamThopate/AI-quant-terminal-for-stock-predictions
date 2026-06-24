"""
Agent Tool Abstraction Layer
===============================
Wraps the algo trading system's Python modules into standardized "tools"
that an LLM agent can invoke via function calling.

Each tool:
  - Accepts simple string/numeric parameters
  - Validates and type-casts all inputs in a guarded block BEFORE the main
    logic — a malformed LLM argument returns a descriptive error JSON that
    the ReAct loop can self-correct from, never crashing the app thread.
  - Returns a clean JSON-serializable dict or string
  - Handles ALL internal exceptions gracefully (no unhandled raise)
"""

import json
import logging
import traceback
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Lazy-loaded singleton instances to avoid re-initializing on every call
_price_fetcher = None
_fundamental_fetcher = None
_macro_fetcher = None
_news_fetcher = None
_feature_builder = None
_regime_detector = None
_labeler = None
_predictor = None           # EnsemblePredictor
_risk_filter = None
_paper_trader = None
_sentiment_scorer = None
_initialized = False        # explicit flag guards against partially-constructed state


def _init_modules():
    """Lazy-initialize all modules once per process lifetime."""
    global _price_fetcher, _fundamental_fetcher, _macro_fetcher, _news_fetcher
    global _feature_builder, _regime_detector, _labeler, _predictor
    global _sentiment_scorer, _initialized

    if _initialized:
        return  # already done

    from data.price_fetcher import PriceFetcher
    from data.fundamental_fetcher import FundamentalFetcher
    from data.macro_fetcher import MacroFetcher
    from data.news_fetcher import NewsFetcher
    from features.feature_matrix import FeatureMatrixBuilder
    from features.sentiment import SentimentScorer
    from intelligence.regime_detector import RegimeDetector
    from intelligence.labeler import TripleBarrierLabeler
    from intelligence.ensemble_predictor import EnsemblePredictor

    _price_fetcher = PriceFetcher()
    _fundamental_fetcher = FundamentalFetcher()
    _macro_fetcher = MacroFetcher()
    _news_fetcher = NewsFetcher()
    _feature_builder = FeatureMatrixBuilder()
    _sentiment_scorer = SentimentScorer()
    _regime_detector = RegimeDetector()
    _labeler = TripleBarrierLabeler()
    _predictor = EnsemblePredictor()

    _initialized = True
    logger.info("All agent tool modules initialized (EnsemblePredictor backend).")


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _arg_error(tool_name: str, detail: str) -> str:
    """
    Return a structured JSON error string back to the ReAct agent when argument
    parsing or type-casting fails.  The agent uses this to self-correct and
    retry with fixed parameters — the app thread never crashes.
    """
    msg = (
        f"Error in tool '{tool_name}': {detail}. "
        "Please check argument types and retry."
    )
    logger.warning("Argument error in %s: %s", tool_name, detail)
    return json.dumps({"error": msg})


def _normalise_ticker(raw: str) -> str:
    """Clean and uppercase a raw ticker string, appending .NS if needed."""
    return raw.strip().upper().replace(".NS", "") + ".NS"


# =====================================================================
# TOOL 1: Stock Prediction
# =====================================================================

def get_stock_prediction(ticker) -> str:
    """
    Get the XGBoost+LSTM Ensemble Triple Barrier prediction for a single stock,
    combined with current market regime and VADER news sentiment.

    Returns the probability of hitting +2% take-profit vs -2% stop-loss
    within a 5-day horizon, along with an Advisory Action Signal.

    Args:
        ticker: NSE stock symbol (e.g., 'RELIANCE', 'TCS', 'INFY').

    Returns:
        JSON string with an Advisory Snapshot.
    """
    # ── Input validation & type coercion ─────────────────────────────────
    try:
        if ticker is None or str(ticker).strip() == "":
            return _arg_error("get_stock_prediction", "ticker must be a non-empty string")
        ticker = str(ticker).strip()
    except Exception as e:
        return _arg_error("get_stock_prediction", f"could not parse ticker argument: {e}")

    _init_modules()
    try:
        from config.settings import TRAINING_DATA_PERIOD, SL_DETECTION_THRESHOLD

        t = _normalise_ticker(ticker)

        price_df = _price_fetcher.fetch_ohlcv(t, period=TRAINING_DATA_PERIOD)
        if price_df.empty:
            return json.dumps({"error": f"No price data available for {t}"})

        benchmark = _price_fetcher.fetch_benchmark(period=TRAINING_DATA_PERIOD)
        if benchmark.empty:
            return json.dumps({"error": "Could not fetch Nifty 50 benchmark data"})

        fundamentals = _fundamental_fetcher.fetch_multiple([t])
        macro = _macro_fetcher.fetch()
        matrix = _feature_builder.build(
            price_data={t: price_df},
            benchmark_close=benchmark["Close"],
            fundamentals=fundamentals,
            macro=macro,
            normalize=False,
        )

        regime_df = _regime_detector.fit_predict(benchmark["Close"])
        regime_map = regime_df["regime"].to_dict()
        matrix["regime_label"] = matrix.index.map(
            lambda dt: regime_map.get(dt, "bull")
        )
        regime_encoding = {"bull": 1, "bear": -1, "high_volatility": 0}
        matrix["regime_encoded"] = (
            matrix["regime_label"].map(regime_encoding).fillna(0).astype(int)
        )

        if _predictor.ensure_model():
            predictions = _predictor.predict_latest(matrix)
            pred = predictions.get(t, {})
        else:
            logger.warning("No trained model found. Run 'python retrain.py' first.")
            pred = {"p_take_profit": 0.33, "p_stop_loss": 0.33, "p_neutral": 0.34}

        p_tp = pred.get("p_take_profit", 0.33)
        p_sl = pred.get("p_stop_loss", 0.33)
        p_neutral = pred.get("p_neutral", 0.34)

        try:
            sentiment_json = get_latest_sentiment(t)
            sentiment_data = json.loads(sentiment_json)
        except Exception:
            sentiment_data = {"sentiment_score": 0.0, "interpretation": "Unknown"}

        sentiment_score = sentiment_data.get("sentiment_score", 0.0)
        sentiment_interp = sentiment_data.get("interpretation", "Neutral")

        current_regime = (
            regime_df["regime"].iloc[-1] if not regime_df.empty else "Unknown"
        )

        latest_price = float(price_df["Close"].iloc[-1])

        rsi = float(matrix["rsi"].iloc[-1]) if "rsi" in matrix.columns else 50.0
        sma_200 = (
            float(price_df["Close"].rolling(200).mean().iloc[-1])
            if len(price_df) >= 200
            else 0.0
        )
        tech_confluence_met = (rsi > 40) and (latest_price > sma_200)

        fund = fundamentals.get(t)
        roe = fund.roe if fund and fund.roe is not None else 0.0
        de_ratio = (
            fund.debt_to_equity if fund and fund.debt_to_equity is not None else 0.0
        )
        fund_met = (roe > 0.15) and (de_ratio < 200.0)

        regime_met = current_regime == "bull"
        edge = p_tp - p_sl
        conf_met = edge >= 0.10 and p_tp >= 0.40

        if p_tp > p_sl:
            if (
                edge >= 0.10
                and p_tp >= 0.40
                and regime_met
                and (tech_confluence_met or fund_met or sentiment_score > 0.05)
            ):
                action_signal = "Strong Buy"
            elif edge >= 0.03 and (
                regime_met or tech_confluence_met or sentiment_score >= 0.0
            ):
                action_signal = "Buy"
            else:
                action_signal = "Lean Buy"
        elif p_sl > p_tp and p_sl >= SL_DETECTION_THRESHOLD:
            if (p_sl - p_tp) >= 0.10 and sentiment_score < -0.05:
                action_signal = "Strong Sell"
            else:
                action_signal = "Sell / Avoid"
        else:
            action_signal = "Hold"

        result = {
            "advisory_snapshot": "Advisory Snapshot",
            "ticker": t,
            "latest_price": round(latest_price, 2),
            "ml_probabilities": {
                "take_profit_2pct": round(p_tp, 4),
                "stop_loss_2pct": round(p_sl, 4),
                "neutral": round(p_neutral, 4),
            },
            "market_regime": current_regime,
            "sentiment": {
                "score": round(sentiment_score, 4),
                "interpretation": sentiment_interp,
            },
            "filters_passed": {
                "high_confidence": conf_met,
                "bull_regime": regime_met,
                "strong_fundamentals": fund_met,
                "technical_confluence": tech_confluence_met,
            },
            "action_signal": action_signal,
            "model": "XGBoost+LSTM Ensemble",
        }
        return json.dumps(result)

    except Exception as e:
        logger.error("get_stock_prediction failed: %s", traceback.format_exc())
        return json.dumps({"error": f"Prediction failed for {ticker}: {str(e)}"})


# =====================================================================
# TOOL 2: Market Regime
# =====================================================================

def get_market_regime() -> str:
    """
    Get the current market regime (Bull, Bear, or High_Volatility).

    Uses K-Means clustering on 20-day returns and volatility of the
    Nifty 50 benchmark index to classify the current market state.

    Returns:
        JSON string with current_regime, regime_distribution, and last_5_labels.
    """
    # No user-supplied arguments — no arg validation needed.
    _init_modules()
    try:
        benchmark = _price_fetcher.fetch_benchmark(period="1y")
        if benchmark.empty:
            return json.dumps({"error": "Could not fetch Nifty 50 data"})

        regime_df = _regime_detector.fit_predict(benchmark["Close"])
        current = regime_df["regime"].iloc[-1]
        dist = regime_df["regime"].value_counts().to_dict()
        last_5 = regime_df["regime"].iloc[-5:].tolist()
        nifty_close = float(benchmark["Close"].iloc[-1])

        result = {
            "current_regime": current,
            "nifty_50_close": round(nifty_close, 2),
            "regime_distribution": dist,
            "last_5_days_regime": last_5,
            "interpretation": {
                "bull": "Markets trending up with moderate volatility. Favor equity exposure.",
                "bear": "Markets trending down. Reduce exposure, consider cash/LIQUIDBEES.",
                "high_volatility": "Extreme volatility. Cap sector exposure at 20%, hold cash.",
            }.get(current, "Unknown"),
        }
        return json.dumps(result)

    except Exception as e:
        logger.error("get_market_regime failed: %s", traceback.format_exc())
        return json.dumps({"error": f"Regime detection failed: {str(e)}"})


# =====================================================================
# TOOL 3: Latest Sentiment
# =====================================================================

def get_latest_sentiment(ticker) -> str:
    """
    Get the VADER sentiment score from recent news headlines for a stock.

    Fetches RSS news headlines and computes an aggregate sentiment score
    ranging from -1.0 (extremely negative) to +1.0 (extremely positive).

    Args:
        ticker: NSE stock symbol (e.g., 'RELIANCE', 'TCS').

    Returns:
        JSON string with sentiment_score, headline_count, and sample headlines.
    """
    # ── Input validation ──────────────────────────────────────────────────
    try:
        if ticker is None or str(ticker).strip() == "":
            return _arg_error("get_latest_sentiment", "ticker must be a non-empty string")
        ticker = str(ticker).strip()
    except Exception as e:
        return _arg_error("get_latest_sentiment", f"could not parse ticker argument: {e}")

    _init_modules()
    try:
        t = _normalise_ticker(ticker)

        headlines = _news_fetcher.fetch_for_tickers([t])
        if not headlines:
            return json.dumps({
                "ticker": t,
                "sentiment_score": 0.0,
                "headline_count": 0,
                "note": "No recent news headlines found for this ticker.",
            })

        scored = []
        for h in headlines:
            score = _sentiment_scorer.score_single(h.title)
            scored.append({"title": h.title[:100], "score": round(score, 4)})

        avg_score = sum(s["score"] for s in scored) / len(scored) if scored else 0.0

        if avg_score > 0.1:
            interpretation = "Positive sentiment — bullish news flow"
        elif avg_score < -0.1:
            interpretation = "Negative sentiment — bearish news flow"
        else:
            interpretation = "Neutral sentiment — mixed or bland news"

        result = {
            "ticker": t,
            "sentiment_score": round(avg_score, 4),
            "headline_count": len(scored),
            "interpretation": interpretation,
            "top_headlines": scored[:5],
        }
        return json.dumps(result)

    except Exception as e:
        logger.error("get_latest_sentiment failed: %s", traceback.format_exc())
        return json.dumps({"error": f"Sentiment analysis failed for {ticker}: {str(e)}"})


# =====================================================================
# TOOL 4: Macro Snapshot
# =====================================================================

def get_macro_snapshot() -> str:
    """
    Get the current Indian macroeconomic snapshot.

    Returns RBI repo rate, CPI inflation, USD/INR exchange rate,
    and 10-year government bond yield.

    Returns:
        JSON string with repo_rate, cpi_inflation, usdinr, gsec_10y_yield.
    """
    # No user-supplied arguments — no arg validation needed.
    _init_modules()
    try:
        macro = _macro_fetcher.fetch()

        result = {
            "repo_rate_pct": macro.repo_rate,
            "cpi_inflation_pct": macro.cpi_inflation,
            "usdinr": macro.usdinr,
            "gsec_10y_yield_pct": macro.gsec_10y_yield,
            "interpretation": (
                f"RBI repo rate at {macro.repo_rate}%. "
                f"CPI inflation at {macro.cpi_inflation}%. "
                f"Rupee trading at {macro.usdinr} per USD. "
                f"10Y government bond yield at {macro.gsec_10y_yield}%."
            ),
        }
        return json.dumps(result)

    except Exception as e:
        logger.error("get_macro_snapshot failed: %s", traceback.format_exc())
        return json.dumps({"error": f"Macro data fetch failed: {str(e)}"})


# =====================================================================
# TOOL 5: Top Momentum Stocks
# =====================================================================

def scan_top_momentum_stocks(limit=5) -> str:
    """
    Scans the top Nifty stocks to find the leaders in momentum.
    Uses 5-day return as the momentum indicator.

    Args:
        limit: Number of top stocks to return (default 5). Accepts int or
               numeric string (e.g. '5').

    Returns:
        JSON string with the top momentum stocks.
    """
    # ── Input validation & type coercion ─────────────────────────────────
    try:
        limit = int(str(limit).strip()) if limit not in (None, "") else 5
        if limit <= 0 or limit > 50:
            return _arg_error(
                "scan_top_momentum_stocks",
                f"limit must be a positive integer ≤ 50, got '{limit}'"
            )
    except (ValueError, TypeError) as e:
        return _arg_error(
            "scan_top_momentum_stocks",
            f"could not parse 'limit' as integer: {e}"
        )

    _init_modules()
    try:
        tickers = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
            "SBIN.NS", "BAJFINANCE.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
            "LT.NS", "HINDUNILVR.NS", "AXISBANK.NS", "MARUTI.NS", "SUNPHARMA.NS",
        ]

        price_data = _price_fetcher.fetch_multiple(tickers, period="5d")

        momentum = []
        for t, df in price_data.items():
            if not df.empty and len(df) >= 2:
                last = df["Close"].iloc[-1]
                first = df["Close"].iloc[0]
                ret = (last - first) / first * 100
                momentum.append({"ticker": t, "5_day_return_pct": round(ret, 2)})

        momentum = sorted(momentum, key=lambda x: x["5_day_return_pct"], reverse=True)
        top_stocks = momentum[:limit]

        result = {
            "screener": "Top 5-Day Momentum",
            "top_stocks": top_stocks,
            "interpretation": "These stocks are showing the strongest recent upward momentum.",
        }
        return json.dumps(result)

    except Exception as e:
        logger.error("scan_top_momentum_stocks failed: %s", traceback.format_exc())
        return json.dumps({"error": f"Screener failed: {str(e)}"})


# =====================================================================
# TOOL 6: Sector Scanner with ML Predictions
# =====================================================================

def scan_sector_predictions(sector, limit=5) -> str:
    """
    Scans a specific sector and returns ML predictions for the top stocks.

    Args:
        sector: The sector name (e.g., 'IT', 'BANKING', 'AUTO', 'FMCG').
                Accepts any capitalisation; common aliases (BANK, AUTOMOBILE)
                are automatically resolved.
        limit: Number of top stocks to run predictions for (default: 5).
               Accepts int or numeric string.

    Returns:
        JSON string with aggregated predictions.
    """
    # ── Input validation & type coercion ─────────────────────────────────
    try:
        if sector is None or str(sector).strip() == "":
            return _arg_error("scan_sector_predictions", "sector must be a non-empty string")
        sector = str(sector).strip()
    except Exception as e:
        return _arg_error("scan_sector_predictions", f"could not parse 'sector' argument: {e}")

    try:
        limit = int(str(limit).strip()) if limit not in (None, "") else 5
        if limit <= 0 or limit > 20:
            return _arg_error(
                "scan_sector_predictions",
                f"limit must be a positive integer ≤ 20, got '{limit}'"
            )
    except (ValueError, TypeError) as e:
        return _arg_error(
            "scan_sector_predictions",
            f"could not parse 'limit' as integer: {e}"
        )

    _init_modules()
    try:
        from config.settings import SECTOR_MAP

        sector_upper = sector.upper()

        # Handle common aliases
        _aliases = {"BANK": "BANKING", "AUTOMOBILE": "AUTO", "PHARMA": "HEALTHCARE"}
        sector_upper = _aliases.get(sector_upper, sector_upper)

        tickers = SECTOR_MAP.get(sector_upper, [])
        if not tickers:
            available = list(SECTOR_MAP.keys())
            return json.dumps({
                "error": (
                    f"Sector '{sector}' not found. "
                    f"Available sectors: {available}"
                )
            })

        tickers_to_scan = tickers[:limit]

        results = []
        for t in tickers_to_scan:
            try:
                pred_json = get_stock_prediction(t)
                pred_data = json.loads(pred_json)

                if "error" not in pred_data:
                    pred_data.pop("advisory_snapshot", None)
                    results.append(pred_data)
            except Exception as exc:
                logger.warning(
                    "Failed to predict for %s during sector scan: %s", t, str(exc)
                )
                continue

        return json.dumps({
            "sector": sector_upper,
            "scanned_count": len(results),
            "predictions": results,
        })

    except Exception as e:
        logger.error("scan_sector_predictions failed: %s", traceback.format_exc())
        return json.dumps({"error": f"Sector scan failed: {str(e)}"})


# =====================================================================
# Tool Registry (for the LLM orchestrator)
# =====================================================================

TOOL_REGISTRY = {
    "get_stock_prediction": get_stock_prediction,
    "get_market_regime": get_market_regime,
    "get_latest_sentiment": get_latest_sentiment,
    "get_macro_snapshot": get_macro_snapshot,
    "scan_top_momentum_stocks": scan_top_momentum_stocks,
    "scan_sector_predictions": scan_sector_predictions,
}
