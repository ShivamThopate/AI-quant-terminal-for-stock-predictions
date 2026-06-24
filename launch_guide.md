# Phase 4 Launch Guide

The Phase 4 Web UI has been successfully integrated. It acts as a native Streamlit Bloomberg-lite terminal that hosts our ReAct LLM Agent.

## How to Launch

Open your terminal, navigate to the project directory, and run:

```bash
streamlit run app.py
```

This will spin up a local web server and automatically open the UI in your default browser.

## Architectural Overview

### 1. Aesthetic Native Configuration
The dark mode, primary accent color (`#00FF7F`), and sleek typography are entirely driven by the newly created `.streamlit/config.toml` file. This means the UI is stable and free of brittle CSS hacks.

### 2. Tab 1: Agent Desk (ReAct Tool Visibility)
The `AgentOrchestrator` has been updated to accept a `tool_callback`. In `app.py`, we hook this callback directly into Streamlit's `st.status("Agent is thinking...")` context manager. 
- **What this means for you**: As the LLM reasons about your prompt, you will see a live, expanding dropdown showing exactly which backend tools it is invoking (e.g., "Fetching macro data", "Running XGBoost...").
- When the Agent returns a Portfolio Optimization JSON, `app.py` parses it and seamlessly renders a responsive Plotly Pie chart (`use_container_width=True`) alongside a dataframe.

### 3. Tab 2: Live Market Intelligence
A standalone dashboard powered by the Phase 1 fetchers.
- Provides live Nifty 50 metrics, Macro data (RBI Repo, Inflation), and a VADER Sentiment Scanner.
- **Rate-Limit Protection**: All backend calls here are wrapped in `@st.cache_data(ttl=300)` to ensure we don't accidentally get IP-banned from Yahoo Finance or RBI.

### 4. Robust State Management
Because Streamlit reruns its Python script every time a user clicks a button or types a message, traditional variables get wiped. We've bypassed this amnesia by securely housing the `AgentOrchestrator` instance and the entire chat history inside `st.session_state`. 
- **What this means for you**: You can freely bounce between the Chat and the Dashboards without the LLM forgetting the conversation context.
