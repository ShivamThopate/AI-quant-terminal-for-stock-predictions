# Final Trade Log -- End-to-End Integration Test

**Query**: `Deploy 10 Lakhs into Nifty IT, low risk.`

**Timestamp**: 2026-05-08T22:45:45.936548

**Status**: success

---

## Parsed Intent

| Field | Value |
|---|---|
| Action | analyze |
| Sector | IT |
| Tickers | 10 stocks |
| Risk | low |
| Timeframe | 1y |

## Intelligence Layer

| Metric | Value |
|---|---|
| Regime | bull |
| ML Accuracy | 0.37447065940713853 |
| ML F1 (weighted) | 0.3589242408547414 |

## ML Predictions

| Ticker | P(TP) | P(SL) | P(Neutral) | Signal |
|---|---|---|---|---|
| TCS.NS | 0.3193 | 0.3679 | 0.3128 | AVOID |
| INFY.NS | 0.2541 | 0.2458 | 0.5001 | BUY |
| HCLTECH.NS | 0.1855 | 0.1831 | 0.6315 | BUY |
| WIPRO.NS | 0.4833 | 0.1601 | 0.3566 | BUY |
| TECHM.NS | 0.3756 | 0.1878 | 0.4366 | BUY |
| PERSISTENT.NS | 0.5069 | 0.0991 | 0.3940 | BUY |
| COFORGE.NS | 0.2805 | 0.5452 | 0.1743 | AVOID |
| MPHASIS.NS | 0.3027 | 0.3031 | 0.3942 | AVOID |
| LTTS.NS | 0.3840 | 0.3148 | 0.3011 | BUY |

## Executed Trades (from paper_trades.db)

| ID | Timestamp | Ticker | Action | Qty | Price | Value | Fee |
|---|---|---|---|---|---|---|---|
| 1 | 2026-05-08T22:46:11.196963 | TECHM.NS | BUY | 120 | Rs.1,463.00 | Rs.175,560.00 | Rs.175.56 |

## Portfolio State (from paper_trades.db)

| Field | Value |
|---|---|
| Cash Balance | Rs.824,264.44 |
| Equity Value | Rs.175,560.00 |
| Total Portfolio | Rs.999,824.44 |

## Execution Summary

| Metric | Value |
|---|---|
| Starting Capital | Rs.1,000,000.00 |
| Total Invested | Rs.175,560.00 |
| Cash Remaining | Rs.824,264.44 |
| Transaction Fees | Rs.175.56 |
| Trades Executed | 1 |

## Risk Warnings

- WIPRO.NS: Max drawdown (29.4%) exceeds threshold (25.0%). REMOVED from allocation.
- TCS.NS: Max drawdown (32.7%) exceeds threshold (25.0%). REMOVED from allocation.
- LTTS.NS: Max drawdown (34.5%) exceeds threshold (25.0%). REMOVED from allocation.
- HCLTECH.NS: Max drawdown (30.3%) exceeds threshold (25.0%). REMOVED from allocation.
- TECHM.NS: Trading -11.0% below recent peak. Trailing stop-loss flag.
