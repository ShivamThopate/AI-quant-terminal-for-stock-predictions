"""
LLM Orchestrator — ReAct Agent with Function Calling (Groq)
=============================================================
Uses the OpenAI-compatible Groq API to create a true agentic
loop that can reason, call tools, observe results, and synthesize
natural-language responses.

The agent has access to 6 proprietary tools:
    1. get_stock_prediction(ticker)
    2. get_market_regime()
    3. get_latest_sentiment(ticker)
    4. get_macro_snapshot()
    5. execute_portfolio_optimization(sector, risk_level, capital)
    6. scan_top_momentum_stocks(limit)

Setup:
    pip install openai python-dotenv
    Create a .env file: GROQ_API_KEY=gsk_your_key_here
"""

import json
import logging
import os
import sys
import time
from typing import Dict, Optional

# Force UTF-8 on Windows consoles to prevent charmap UnicodeEncodeErrors from emojis
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

logger = logging.getLogger(__name__)

# System prompt defining the agent's persona and behavior
SYSTEM_PROMPT = """\
You are a Quantitative Advisor for the Indian Equity Market (NSE).
You analyze data, predict probabilities, and evaluate news sentiment, but you do NOT manage money, allocate capital, or execute trades.

YOUR CAPABILITIES:
- Predict stock direction using a Triple Barrier XGBoost model (+2% TP / -2% SL / 5-day horizon)
- Detect market regime (Bull/Bear/High_Volatility) using K-Means clustering on Nifty 50
- Analyze news sentiment using VADER NLP
- Fetch Indian macro data (RBI repo rate, CPI inflation, USD/INR, 10Y G-Sec yield)
- Scan top momentum stocks across Nifty constituents
- Scan an entire sector for ML predictions using `scan_sector_predictions`

RULES:
1. When asked about a specific stock (e.g. "Should I sell X?" or "Analyze Y"), call `get_stock_prediction` which now returns a unified Advisory Snapshot (Probabilities, Regime, and Sentiment).
2. When asked to scan or analyze an entire sector (e.g., "scan IT sector", "scan banking sector"), you MUST strictly call `scan_sector_predictions` ONCE. Do NOT call `get_stock_prediction` in a loop or in parallel for multiple stocks, as that slows down the system. `scan_sector_predictions` gives you the ML predictions for the top stocks in that sector instantly.
3. Synthesize the Advisory Snapshot to answer the user's EXACT intent. Give a direct, reasoned recommendation based on the regime, sentiment, and ML prediction.
4. If the user asks for general recommendations ("what to buy"), call `scan_top_momentum_stocks`. However, if the user asks for SPECIFIC criteria like "low risk" or "safe stocks", DO NOT blindly return momentum stocks (which are high risk). Instead, use your own knowledge to pick 2-3 appropriate Nifty 50 stocks (e.g., ITC, HINDUNILVR for low risk) and run `get_stock_prediction` on them to provide a custom, accurate response.
5. NEVER fabricate numbers. Always use your tools to fetch real data.
6. Express probabilities as percentages. Format prices in INR with commas.
7. Present the probabilities, regime, and sentiment score in clean Markdown tables or bullet points when summarizing an asset. Use rich Markdown formatting for readability and bold text for emphasis. Avoid giant paragraphs of text.
8. Always mention caveats: you are an advisor, this is not financial advice.
"""


def _build_tool_declarations():
    """Build OpenAI-format function declarations for all tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_stock_prediction",
                "description": (
                    "Get the XGBoost Triple Barrier prediction for a single NSE stock. "
                    "Returns the probability of hitting +3% take-profit vs -2% stop-loss "
                    "within 5 trading days, along with a BUY/HOLD/AVOID signal."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "NSE stock symbol, e.g., 'RELIANCE', 'TCS', 'INFY', 'MARUTI'",
                        }
                    },
                    "required": ["ticker"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_market_regime",
                "description": (
                    "Get the current market regime classification (Bull, Bear, or High_Volatility) "
                    "based on K-Means clustering of Nifty 50 returns and volatility."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_latest_sentiment",
                "description": (
                    "Get the VADER sentiment score from recent news headlines for a stock. "
                    "Returns a score from -1.0 (very negative) to +1.0 (very positive)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "NSE stock symbol, e.g., 'RELIANCE', 'TCS'",
                        }
                    },
                    "required": ["ticker"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_macro_snapshot",
                "description": (
                    "Get the current Indian macroeconomic snapshot: RBI repo rate, "
                    "CPI inflation, USD/INR exchange rate, and 10-year G-Sec bond yield."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "scan_top_momentum_stocks",
                "description": (
                    "Scans the top Nifty stocks to find the leaders in momentum. "
                    "Returns the top stocks based on 5-day return. "
                    "Use this whenever the user asks for general stock recommendations, "
                    "'what to buy', 'best stocks', or any open-ended stock discovery request."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "string",
                            "description": "The number of top stocks to return as a string integer (e.g., '5'). Default is 5.",
                        }
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "scan_sector_predictions",
                "description": (
                    "Scans a specific sector and returns the ML predictions (probabilities, sentiment, and signals) "
                    "for the top stocks in that sector simultaneously. Use this when the user asks to scan "
                    "or analyze an entire sector (e.g., 'scan IT sector' or 'analyze banking sector')."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sector": {
                            "type": "string",
                            "description": "The sector name, e.g., 'IT', 'BANKING', 'AUTO', 'FMCG', 'METAL', 'ENERGY'.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of stocks to scan. Default is 5.",
                        }
                    },
                    "required": ["sector"],
                },
            },
        },
    ]


class AgentOrchestrator:
    """
    LLM-powered ReAct agent with function calling via Groq.

    Uses the OpenAI-compatible Groq API to reason about user queries,
    call the appropriate backend tools, and synthesize natural-language
    responses.

    Usage::

        agent = AgentOrchestrator()
        response = agent.run("Analyze RELIANCE for a short-term trade")
        print(response)
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "llama-3.3-70b-versatile"):
        self._api_key = api_key or os.getenv("GROQ_API_KEY")
        self._model_name = model_name
        self._client = None

        if not self._api_key:
            logger.warning(
                "No GROQ_API_KEY found. Set it in .env or pass it directly. "
                "The agent will fall back to the static pipeline."
            )

    def _ensure_client(self):
        """Lazy-initialize the Groq client via the OpenAI SDK."""
        if self._client is not None:
            return True

        if not self._api_key:
            return False

        try:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self._api_key,
                base_url="https://api.groq.com/openai/v1",
            )
            logger.info("Groq client initialized with model: %s", self._model_name)
            return True

        except ImportError:
            logger.error(
                "openai package not installed. "
                "Run: pip install openai"
            )
            return False
        except Exception as e:
            logger.error("Failed to initialize Groq client: %s", e)
            return False

    def run(self, user_query: str, tool_callback=None) -> str:
        """
        Process a user query through the LLM agent.
        The agent will reason, call tools, and synthesize a response.
        """
        try:
            if not self._ensure_client():
                return "⚠️ *System Notice: LLM API key not configured. Please set GROQ_API_KEY.*"

            # Try primary model, then fallback models with exponential backoff
            models_to_try = [self._model_name, "llama3-groq-70b-8192-tool-use-preview", "llama-3.1-8b-instant"]
            last_error = None
            wait_time = 5

            for model in models_to_try:
                try:
                    return self._agent_loop(user_query, model_name=model, tool_callback=tool_callback)
                except Exception as e:
                    err_str = str(e).lower()
                    last_error = e
                    if "429" in err_str or "rate_limit" in err_str or "resource_exhausted" in err_str:
                        logger.warning("Rate limited on %s, waiting %ds...", model, wait_time)
                        time.sleep(wait_time)
                        wait_time = min(wait_time * 2, 60)
                        continue
                    else:
                        logger.error("Agent loop failed on %s: %s", model, e)
                        continue

            return f"⚠️ *System Notice: I encountered an internal error while executing my tools. [{str(last_error)}]. Please try rephrasing your request.*"

        except Exception as e:
            logger.error("Agent orchestrator crashed: %s", str(e))
            return f"⚠️ *System Notice: I encountered an internal error while executing my tools. [{str(e)}]. Please try rephrasing your request.*"

    def _agent_loop(self, user_query: str, model_name: str = None, tool_callback=None) -> str:
        """Execute the ReAct agent loop with OpenAI-compatible function calling."""
        model_name = model_name or self._model_name
        logger.info("Agent using model: %s", model_name)

        # Import tool functions
        from agent.tools import TOOL_REGISTRY

        # Build tool config
        tools = _build_tool_declarations()

        # Start conversation
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ]

        max_iterations = 8  # safety limit for tool-calling loops

        for iteration in range(max_iterations):
            logger.info("Agent iteration %d", iteration + 1)

            response = self._client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.3,
                max_tokens=4096,
            )

            choice = response.choices[0]
            message = choice.message

            # Check if the model wants to call functions
            if not message.tool_calls:
                # No more function calls — we have the final answer
                return message.content or "Analysis complete."

            # Append the assistant's message (with tool_calls) to conversation
            messages.append(message)

            # Process each tool call
            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                try:
                    fn_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                except json.JSONDecodeError:
                    fn_args = {}
                    logger.warning("Failed to parse tool arguments for %s: %s",
                                   fn_name, tool_call.function.arguments)

                logger.info("Tool call: %s(%s)", fn_name, fn_args)

                if tool_callback:
                    tool_callback(fn_name, fn_args)

                # Execute the tool
                tool_fn = TOOL_REGISTRY.get(fn_name)
                if tool_fn:
                    try:
                        result_str = tool_fn(**fn_args)
                    except Exception as e:
                        result_str = json.dumps({"error": f"Tool execution failed: {str(e)}"})
                else:
                    result_str = json.dumps({"error": f"Unknown tool: {fn_name}"})

                logger.info("Tool result: %s", result_str[:200])

                # Add tool result to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_str,
                })

        return "⚠️ *System Notice: I reached maximum tool iterations without finishing. Please try a simpler query.*"
