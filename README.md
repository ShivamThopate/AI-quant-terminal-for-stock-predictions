# 🏦 AI Quant Terminal for Stock Predictions

An AI-powered quantitative trading terminal built for the Indian stock market (NSE). It combines **machine learning models** (XGBoost, LSTM), **NLP sentiment analysis** (FinBERT, VADER), and a **ReAct LLM Agent** (powered by Groq) — all served through a sleek Bloomberg-lite Streamlit interface.

---

## ✨ Features

- **AI Agent Desk** — Chat with a ReAct LLM agent that can analyze stocks, sectors, and build optimized portfolios in real-time
- **Live Market Intelligence** — Track Nifty 50, Bank Nifty, Gold, Silver, and Bitcoin prices in INR
- **Interactive Chart Explorer** — Candlestick and line charts with multiple timeframes
- **Sentiment Scanner** — VADER-based news sentiment analysis with gauge visualization
- **Macro Dashboard** — Live RBI Repo Rate, CPI Inflation, USD/INR, and 10Y G-Sec yields
- **Portfolio Optimization** — Mean-variance optimization using PyPortfolioOpt
- **Triple Barrier Labeling** — ML models trained with take-profit, stop-loss, and time-based exit signals
- **Ensemble Predictions** — Combines XGBoost and LSTM model outputs for robust signals

---

## 🚀 How to Run (Step by Step)

### Step 1: Install Python

Download and install **Python 3.10 or above** from [python.org](https://www.python.org/downloads/).

> ⚠️ During installation, make sure to **check the box** that says **"Add Python to PATH"**.

---

### Step 2: Clone the Repository

Open a terminal (Command Prompt / PowerShell / Terminal) and run:

```bash
git clone https://github.com/ShivamThopate/AI-quant-terminal-for-stock-predictions.git
```

---

### Step 3: Enter the Project Folder

```bash
cd AI-quant-terminal-for-stock-predictions
```

---

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all the libraries the project needs (yfinance, streamlit, xgboost, torch, etc.)

---

### Step 5: Get a Free Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Create a new API key and copy it

---

### Step 6: Create the `.env` File

Copy the example environment file:

```bash
# On Windows
copy .env.example .env

# On Mac/Linux
cp .env.example .env
```

Then open the `.env` file in any text editor and replace `gsk_your_groq_api_key_here` with your actual Groq API key:

```
GROQ_API_KEY=gsk_your_actual_key_here
```

---

### Step 7: Launch the App

```bash
streamlit run app.py
```

The **AI Quant Terminal** will open automatically in your browser at `http://localhost:8501`.

---

## 🖥️ Screenshots

Once launched, you'll see two main tabs:

| Tab | Description |
|-----|-------------|
| 💬 **Agent Desk** | Chat with the AI agent — ask it to analyze stocks, sectors, or build portfolios |
| 📈 **Live Market Intelligence** | Real-time indices, interactive charts, macro data, and sentiment scanner |

---

## 📁 Project Structure

```
AI-quant-terminal-for-stock-predictions/
├── app.py                  # Streamlit web interface (main entry point)
├── main.py                 # CLI entry point
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── agent/
│   ├── orchestrator.py     # ReAct LLM agent loop
│   └── tools.py            # Agent tool definitions
├── config/
│   ├── settings.py         # Global configuration & constants
│   └── sectors.json        # NSE sector → ticker mappings
├── data/
│   ├── price_fetcher.py    # OHLCV price data (yfinance)
│   ├── fundamental_fetcher.py  # Fundamental data (NSEPython)
│   ├── macro_fetcher.py    # RBI macro indicators
│   └── news_fetcher.py     # News headline scraper
├── features/
│   ├── technical.py        # Technical indicators (RSI, MACD, Bollinger, ATR)
│   ├── sentiment.py        # VADER sentiment scoring
│   └── feature_matrix.py   # Feature matrix builder
├── intelligence/
│   ├── predictor.py        # XGBoost classifier
│   ├── lstm_predictor.py   # LSTM neural network
│   ├── ensemble_predictor.py   # Ensemble model combiner
│   ├── labeler.py          # Triple barrier labeling
│   └── regime_detector.py  # Market regime detection
├── nlp/
│   └── finbert_scorer.py   # FinBERT sentiment model
├── models/                 # Saved model weights
├── tests/                  # Unit tests
└── .streamlit/
    └── config.toml         # Streamlit theme configuration
```

---

## ⚙️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **ML/AI** | XGBoost, PyTorch (LSTM), scikit-learn |
| **NLP** | FinBERT (HuggingFace Transformers), NLTK VADER |
| **LLM Agent** | Groq API (via OpenAI SDK) |
| **Data** | yfinance, jugaad-data, NSEPython |
| **Portfolio** | PyPortfolioOpt |
| **Web UI** | Streamlit, Plotly |
| **Market** | NSE (Indian Stock Market) |

---

## 📄 License

This project is open source and available for educational and research purposes.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.
