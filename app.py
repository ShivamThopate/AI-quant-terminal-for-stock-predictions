"""
Streamlit Web Interface — Phase 4
=====================================
A Bloomberg-lite terminal serving as the primary ReAct Agent interface.
"""

import sys
import os
import json
import logging
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.orchestrator import AgentOrchestrator
from data.price_fetcher import PriceFetcher
from data.macro_fetcher import MacroFetcher
from agent.tools import get_latest_sentiment

load_dotenv()

# =====================================================================
# State Management & Initialization
# =====================================================================

st.set_page_config(layout="wide", page_title="AI Quant Terminal", page_icon="🏦")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    api_key = os.getenv("GROQ_API_KEY")
    st.session_state.agent = AgentOrchestrator(api_key=api_key)

# =====================================================================
# Data Fetchers (Cached)
# =====================================================================

@st.cache_data(ttl=300)
def fetch_synthetic_inr_commodity(commodity_ticker: str, period: str, interval: str = "1d"):
    """Converts USD Comex prices to INR physical market prices using live USDINR."""
    pf = PriceFetcher()
    comm_df = pf.fetch_ohlcv(commodity_ticker, period=period, interval=interval)
    inr_df = pf.fetch_ohlcv("INR=X", period=period, interval=interval)
    
    if comm_df.empty or inr_df.empty:
        return pd.DataFrame()
        
    # Align dates (forward fill FX data to match commodity trading hours)
    inr_series = inr_df["Close"].reindex(comm_df.index, method='ffill')
    
    # Fill any leading NaNs with the first valid FX rate
    if inr_series.isna().any():
        inr_series = inr_series.bfill()
    
    # 1 Troy Ounce = 31.1034768 grams
    if commodity_ticker == "GC=F":
        multiplier = 10 / 31.1034768 # Indian Gold is per 10 grams
    elif commodity_ticker == "SI=F":
        multiplier = 1000 / 31.1034768 # Indian Silver is per 1 kilogram
    else:
        multiplier = 1.0
        
    synthetic_df = comm_df.copy()
    for col in ["Open", "High", "Low", "Close"]:
        synthetic_df[col] = synthetic_df[col] * inr_series * multiplier
        
    return synthetic_df

@st.cache_data(ttl=300)
def fetch_live_indices():
    """Fetch live Nifty 50, Bank Nifty, Gold, Silver, and Bitcoin."""
    pf = PriceFetcher()
    data = {}
    
    symbols = {
        "NIFTY 50": "^NSEI",
        "BANK NIFTY": "^NSEBANK",
        "GOLD (10g)": "GOLD_INR",
        "SILVER (1kg)": "SILVER_INR",
        "BITCOIN": "BTC-INR"
    }
    
    for name, ticker in symbols.items():
        try:
            if ticker in ["GOLD_INR", "SILVER_INR"]:
                underlying = "GC=F" if ticker == "GOLD_INR" else "SI=F"
                df = fetch_synthetic_inr_commodity(underlying, period="5d", interval="1d")
            else:
                df = pf.fetch_ohlcv(ticker, period="5d")
                
            if not df.empty and len(df) >= 2:
                last = df["Close"].iloc[-1]
                prev = df["Close"].iloc[-2]
                data[name] = {"price": last, "change_pct": (last - prev) / prev * 100}
        except Exception:
            pass
            
    return data

@st.cache_data(ttl=300)
def fetch_macro_snapshot():
    """Fetch live macro indicators using MacroFetcher."""
    mf = MacroFetcher()
    try:
        macro = mf.fetch()
        return {
            "Repo Rate": f"{macro.repo_rate}%",
            "CPI Inflation": f"{macro.cpi_inflation}%",
            "USD/INR": f"₹{macro.usdinr}",
            "10Y G-Sec": f"{macro.gsec_10y_yield}%",
        }
    except Exception:
        return {}

@st.cache_data(ttl=300)
def fetch_historical_data(ticker: str, period: str = "3mo", interval: str = "1d"):
    """Fetch historical data for charts."""
    if ticker in ["GOLD_INR", "SILVER_INR"]:
        underlying = "GC=F" if ticker == "GOLD_INR" else "SI=F"
        return fetch_synthetic_inr_commodity(underlying, period=period, interval=interval)
        
    pf = PriceFetcher()
    return pf.fetch_ohlcv(ticker, period=period, interval=interval)

def resolve_tickers(input_str: str) -> list:
    """Parse comma separated string into a list of standardized yfinance tickers."""
    raw_items = [x.strip() for x in input_str.split(',') if x.strip()]
    resolved = []
    
    aliases = {
        "gold": "GOLD_INR",
        "gold (10g inr)": "GOLD_INR",
        "silver": "SILVER_INR",
        "silver (1kg inr)": "SILVER_INR",
        "bitcoin": "BTC-INR",
        "bitcoin (inr)": "BTC-INR",
        "btc": "BTC-INR",
        "nifty": "^NSEI",
        "nifty 50": "^NSEI",
        "nifty50": "^NSEI",
        "banknifty": "^NSEBANK",
        "bank nifty": "^NSEBANK"
    }
    
    for item in raw_items:
        lower_item = item.lower()
        if lower_item in aliases:
            resolved.append(aliases[lower_item])
            continue
            
        # If it looks like a direct ticker or commodity/crypto
        if "." in item or item.startswith("^") or "=" in item or "-" in item:
            resolved.append(item.upper())
            continue
            
        # Fallback: take the first word as the ticker symbol
        first_word = item.split()[0].upper()
        first_word = "".join(e for e in first_word if e.isalnum())
        if first_word:
            resolved.append(f"{first_word}.NS")
            
    # Remove duplicates but preserve order
    return list(dict.fromkeys(resolved))



st.title("🏦 AI Quant Terminal")

tab1, tab2 = st.tabs(["💬 Agent Desk", "📈 Live Market Intelligence"])

# ---------------------------------------------------------------------
# TAB 1: Agent Desk (Chat)
# ---------------------------------------------------------------------
with tab1:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input
    if user_input := st.chat_input("Ask the Quantitative Advisor (e.g., 'Analyze the auto sector for low risk' or 'Analyze RELIANCE'):"):
        
        # User message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Agent response
        with st.chat_message("assistant"):
            status = st.status("Agent is thinking...", expanded=True)
            
            def tool_callback(fn_name, fn_args):
                """Callback injected into orchestrator to stream tool calls."""
                status.write(f"🔧 **Tool Call**: `{fn_name}`")
                status.write(f"Parameters: `{json.dumps(fn_args)}`")
            
            # Run the orchestrator directly (errors are handled internally by the agent)
            response_text = st.session_state.agent.run(user_input, tool_callback=tool_callback)
            status.update(label="Analysis Complete", state="complete", expanded=False)

            st.markdown(response_text)
            
            # Save assistant message
            st.session_state.messages.append({"role": "assistant", "content": response_text})

# ---------------------------------------------------------------------
# TAB 2: Live Market Intelligence
# ---------------------------------------------------------------------
with tab2:
    st.header("Global Market Context")
    
    # Indices
    st.subheader("Live Indices & Commodities (INR)")
    indices = fetch_live_indices()
    if indices:
        cols = st.columns(len(indices))
        for idx, (name, data) in enumerate(indices.items()):
            cols[idx].metric(
                label=name,
                value=f"₹{data['price']:,.2f}",
                delta=f"{data['change_pct']:.2f}%"
            )
    else:
        st.warning("Could not fetch live indices. Check connection.")
        
    st.divider()

    # Chart Explorer (Task 1)
    st.subheader("Chart Explorer")
    
    asset_selection = st.selectbox(
        "Select Asset to Chart:",
        ["Gold (10g INR)", "Silver (1kg INR)", "Bitcoin (INR)", "Nifty 50", "Bank Nifty", "Custom Ticker..."]
    )
    
    if asset_selection == "Custom Ticker...":
        chart_ticker = st.text_input("Enter Custom Ticker (e.g., RELIANCE, TCS):")
    else:
        chart_ticker = asset_selection
    
    col1, col2 = st.columns(2)
    with col1:
        timeframe = st.selectbox("Timeframe", ["3 Months", "1 Month", "2 Weeks", "1 Week", "1 Day"], index=0)
    with col2:
        chart_type = st.selectbox("Chart Type", ["Candlestick", "Line"], index=0)
        
    if chart_ticker:
        resolved_list = resolve_tickers(chart_ticker)
        
        if timeframe == "1 Day":
            period_str, interval_str = "1d", "5m"
        elif timeframe == "1 Week":
            period_str, interval_str = "5d", "1d"
        elif timeframe == "2 Weeks":
            period_str, interval_str = "1mo", "1d"  # Will slice to 10 trading days
        elif timeframe == "1 Month":
            period_str, interval_str = "1mo", "1d"
        else:
            period_str, interval_str = "3mo", "1d"

        for ticker in resolved_list:
            with st.spinner(f"Fetching {timeframe} data for {ticker}..."):
                hist_data = fetch_historical_data(ticker, period_str, interval_str)
                if not hist_data.empty:
                    if timeframe == "2 Weeks" and len(hist_data) > 10:
                        hist_data = hist_data.tail(10)
                        
                    if chart_type == "Candlestick":
                        fig = go.Figure(data=[go.Candlestick(
                            x=hist_data.index,
                            open=hist_data['Open'],
                            high=hist_data['High'],
                            low=hist_data['Low'],
                            close=hist_data['Close'],
                            name=ticker
                        )])
                    else:
                        fig = px.line(hist_data, x=hist_data.index, y="Close")
                        fig.update_traces(line_color="#00f2fe", line_width=2)
                        
                    fig.update_layout(
                        title=f"{ticker} - {timeframe} Price History",
                        xaxis_rangeslider_visible=False,
                        template="plotly_dark",
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.error(f"Could not load data for {ticker}. Verify the ticker format.")

    st.divider()
    
    # Macro
    st.subheader("Macro Snapshot")
    macro = fetch_macro_snapshot()
    if macro:
        m_cols = st.columns(len(macro))
        for idx, (key, val) in enumerate(macro.items()):
            m_cols[idx].metric(label=key, value=val)
    else:
        st.warning("Could not fetch macro data.")
        
    st.divider()
    
    # Sentiment (Task 2)
    st.subheader("Sentiment Scanner")
    st.markdown("Scan recent news headlines for VADER sentiment scores.")
    ticker_input = st.text_input("Enter NSE Ticker for Sentiment Scan (e.g., RELIANCE, Tata Motors):")
    if ticker_input:
        resolved_sentiment_list = resolve_tickers(ticker_input)
        for ticker in resolved_sentiment_list:
            with st.spinner(f"Scanning sentiment for {ticker}..."):
                sentiment_json = get_latest_sentiment(ticker)
                try:
                    res = json.loads(sentiment_json)
                    if "error" in res:
                        st.error(res["error"])
                    else:
                        score = res.get("sentiment_score", 0.0)
                        
                        # Plotly Indicator Gauge Chart
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=score,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': f"VADER Sentiment Score for {ticker.upper()}"},
                            gauge={
                                'axis': {'range': [-1, 1], 'tickwidth': 1},
                                'bar': {'color': "white"},
                                'steps': [
                                    {'range': [-1, -0.05], 'color': "red"},
                                    {'range': [-0.05, 0.05], 'color': "gray"},
                                    {'range': [0.05, 1], 'color': "green"}
                                ],
                                'threshold': {
                                    'line': {'color': "white", 'width': 4},
                                    'thickness': 0.75,
                                    'value': score
                                }
                            }
                        ))
                        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20), template="plotly_dark")
                        st.plotly_chart(fig_gauge, width='stretch')
                        
                        st.write(f"**Interpretation:** {res.get('interpretation', 'N/A')}")
                        st.write("**Top Headlines:**")
                        for h in res.get("top_headlines", []):
                            st.write(f"- {h['title']} (Score: `{h['score']}`)")
                except Exception as e:
                    st.error(f"Failed to parse sentiment data for {ticker}: {e}")
                
            st.markdown("---")
