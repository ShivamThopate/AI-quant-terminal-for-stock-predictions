"""
Interactive CLI Entry Point — LLM Agent Mode
================================================
Provides a conversational chat loop powered by the Gemini LLM agent
with function calling (ReAct paradigm).

Falls back to the static pipeline if no GEMINI_API_KEY is configured.
"""

import sys
import os
import io
import logging

# Ensure project root is on sys.path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Handle Windows encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; user must set env vars manually

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(stream=sys.stderr)],
)

from agent.orchestrator import AgentOrchestrator


BANNER = r"""
+==============================================================+
|      ALGO TRADING & FINANCIAL ANALYST SYSTEM                  |
|      NSE India  |  Rs.10,00,000 Paper Capital                 |
|      Powered by XGBoost + Groq LLM Agent                     |
+==============================================================+
|  Try:                                                         |
|   * "What's the outlook for Reliance right now?"              |
|   * "Deploy 10 Lakhs into Nifty IT, low risk"                |
|   * "Compare TCS vs INFY — which one should I pick?"          |
|   * "Is this a good time to invest? Check the market regime"  |
|   * "Recommend few stocks to buy on Monday"                   |
|                                                               |
|  Type 'quit' or 'exit' to leave.                              |
+==============================================================+
"""


def main():
    print(BANNER)

    api_key = os.getenv("GROQ_API_KEY")
    if api_key and api_key != "your_groq_api_key_here":
        print("  [OK] Groq API key detected. LLM Agent mode active.\n")
        agent = AgentOrchestrator(api_key=api_key)
    else:
        print("  [!!] No GROQ_API_KEY found. Running in static pipeline mode.")
        print("       To enable AI agent: create a .env file with GROQ_API_KEY=gsk_your_key")
        print("       Get a free key at: https://console.groq.com\n")
        agent = AgentOrchestrator()  # will use fallback mode

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print()  # spacer
        try:
            response = agent.run(user_input)
            print(f"\nAgent:\n{response}\n")
            print("-" * 60)
        except Exception as exc:
            print(f"\nError: {exc}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
