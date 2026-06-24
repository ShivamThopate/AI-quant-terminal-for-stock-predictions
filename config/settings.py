"""
Global configuration for the Algorithmic Trading & Financial Analyst System.

All system-wide constants, thresholds, and mappings are defined here.
Module-level imports should reference these settings rather than hardcoding values.
"""

import os
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_CACHE_DIR = PROJECT_ROOT / "data" / "cache"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure runtime directories exist
DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Market & Currency
# ---------------------------------------------------------------------------
MARKET = "NSE"
CURRENCY = "INR"
NSE_SUFFIX = ".NS"

# ---------------------------------------------------------------------------
# Capital
# ---------------------------------------------------------------------------
STARTING_CAPITAL = 10_00_000  # ₹10 Lakh

# ---------------------------------------------------------------------------
# Data Source Priority (first = primary, rest = fallbacks)
# ---------------------------------------------------------------------------
PRICE_SOURCES = ["yfinance", "jugaad-data"]
FUNDAMENTAL_SOURCES = ["nsepython", "yfinance"]

# ---------------------------------------------------------------------------
# Timeframe Mapping (NLP → yfinance period strings)
# ---------------------------------------------------------------------------
TIMEFRAME_MAP = {
    "short-term": "30d",
    "short": "30d",
    "1m": "30d",
    "medium-term": "90d",
    "medium": "90d",
    "3m": "90d",
    "long-term": "1y",
    "long": "1y",
    "1y": "1y",
    "year": "1y",
}

DEFAULT_TIMEFRAME = "1y"

# ---------------------------------------------------------------------------
# Risk Profiles
# ---------------------------------------------------------------------------
RISK_MAP = {
    "low": {
        "max_volatility": 0.15,
        "max_sector_pct": 0.30,
        "max_drawdown": 0.25,      # Indian equities regularly draw down 15-25%
        "strategy": "min_volatility",
    },
    "medium": {
        "max_volatility": 0.25,
        "max_sector_pct": 0.40,
        "max_drawdown": 0.20,
        "strategy": "max_sharpe",
    },
    "aggressive": {
        "max_volatility": 0.40,
        "max_sector_pct": 0.50,
        "max_drawdown": 0.35,
        "strategy": "max_sharpe",
    },
}

DEFAULT_RISK = "medium"

# ---------------------------------------------------------------------------
# Triple Barrier Parameters
# ---------------------------------------------------------------------------
TRIPLE_BARRIER = {
    "take_profit_pct": 0.02,   # +2%
    "stop_loss_pct": 0.02,     # -2%
    "max_holding_days": 5,
}

# ---------------------------------------------------------------------------
# Benchmark Index (for Relative Strength calculation)
# ---------------------------------------------------------------------------
BENCHMARK_TICKER = "^NSEI"  # Nifty 50

# ---------------------------------------------------------------------------
# ML / Retraining
# ---------------------------------------------------------------------------
RETRAIN_SCHEDULE = "manual"           # Run retrain.py whenever you want to refresh the model
RETRAIN_ACCURACY_THRESHOLD = 0.42    # Flag model as needing retrain if accuracy < 42%
TRAINING_DATA_PERIOD = "10y"           # Use 10 years of data for training (was 3y)

XGBOOST_PARAMS = {
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.75,
    "colsample_bytree": 0.7,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.5,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "random_state": 42,
}

# ---------------------------------------------------------------------------
# Stop-Loss Detection (asymmetric risk)
# ---------------------------------------------------------------------------
SL_COST_MULTIPLIER = 1.5             # Penalize missing SL events 1.5x more than other errors
SL_DETECTION_THRESHOLD = 0.25        # Flag SL warning if P(stop_loss) >= 25% (vs argmax default)

# ---------------------------------------------------------------------------
# Technical Indicator Defaults
# ---------------------------------------------------------------------------
INDICATOR_DEFAULTS = {
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "rsi_period": 14,
    "atr_period": 14,
    "bb_period": 20,
    "bb_std": 2,
}

# ---------------------------------------------------------------------------
# Sentiment
# ---------------------------------------------------------------------------
SENTIMENT_HEADLINE_LOOKBACK_DAYS = 7

# ---------------------------------------------------------------------------
# News Fetching
# ---------------------------------------------------------------------------
NEWS_REQUEST_DELAY_SECONDS = 2  # rate-limit between requests

# ---------------------------------------------------------------------------
# Paper Trading Database
# ---------------------------------------------------------------------------
TRADES_DB_PATH = PROJECT_ROOT / "trades.db"

# ---------------------------------------------------------------------------
# Sector Mappings (loaded from sectors.json)
# ---------------------------------------------------------------------------
_SECTORS_FILE = CONFIG_DIR / "sectors.json"


def load_sector_map() -> dict:
    """Load the sector → ticker-list mapping from sectors.json."""
    if _SECTORS_FILE.exists():
        with open(_SECTORS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


SECTOR_MAP = load_sector_map()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("ALGO_LOG_LEVEL", "INFO")
