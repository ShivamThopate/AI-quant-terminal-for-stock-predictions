# Triple Barrier XGBoost Model — Walk-Forward Metrics

**Regime**: bull

**Samples**: 1225 | **Features**: 21 | **CV Folds**: 5

---

## Overall Metrics

| Metric | Value |
|---|---|
| **Accuracy** | 0.3941 |
| **Precision (weighted)** | 0.3691 |
| **Recall (weighted)** | 0.3941 |
| **F1 Score (weighted)** | 0.3662 |
| Needs Retrain | Yes |

## Per-Class Metrics

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| SL_hit(-1) | 0.3169 | 0.1711 | 0.2222 | 339 |
| Neutral(0) | 0.4508 | 0.6585 | 0.5352 | 410 |
| TP_hit(+1) | 0.3109 | 0.2731 | 0.2908 | 271 |

## Latest Predictions (Nifty Auto Top 5)

| Ticker | P(Take Profit) | P(Stop Loss) | P(Neutral) | Signal |
|---|---|---|---|---|
| MARUTI.NS | 0.1082 | 0.5824 | 0.3095 | AVOID |
| M&M.NS | 0.6750 | 0.1708 | 0.1541 | BUY |
| BAJAJ-AUTO.NS | 0.0989 | 0.7423 | 0.1588 | AVOID |
| HEROMOTOCO.NS | 0.0925 | 0.5558 | 0.3517 | AVOID |
| EICHERMOT.NS | 0.4021 | 0.3107 | 0.2871 | BUY |

## Top 10 Feature Importance

| Rank | Feature | Importance |
|---|---|---|
| 1 | regime_encoded | 0.151304 |
| 2 | pe_ratio | 0.079385 |
| 3 | bb_middle | 0.067253 |
| 4 | bb_lower | 0.062198 |
| 5 | debt_to_equity | 0.060991 |
| 6 | volatility_20d | 0.058557 |
| 7 | macd_hist | 0.057283 |
| 8 | rs_vs_nifty | 0.057036 |
| 9 | macd_signal | 0.056787 |
| 10 | bb_upper | 0.056720 |
