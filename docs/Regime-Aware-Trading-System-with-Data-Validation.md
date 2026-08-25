# Regime-Aware Quantitative Trading System with Data Validation

A modular quantitative trading framework engineered for adaptive market participation through probabilistic regime detection, institutional-grade data validation, and dynamic strategy allocation.

The system integrates financial time-series preprocessing, volatility-aware regime classification, and signal generation pipelines to emulate production-style quantitative trading infrastructure.

---

## Core Capabilities

- Multi-source OHLCV data ingestion
- Institutional-style data validation framework
- Statistical feature engineering
- Volatility and trend regime detection
- Adaptive strategy switching
- Signal generation and position management
- Extensible architecture for live deployment and backtesting

---

## Validation Framework

Robust preprocessing pipeline designed to mitigate execution risk arising from anomalous market data.

### Implemented Checks
- Missing value imputation
- Timestamp integrity verification
- OHLC consistency validation
- Statistical outlier detection
- Data quality scoring

---

## Regime Detection

Market state classification using:
- Moving Average trend structure
- Rolling volatility estimation
- Probabilistic Hidden Markov Models (`hmmlearn`) *(in progress)*

### Supported Regimes
- `TREND_UP`
- `TREND_DOWN`
- `RANGE`
- `HIGH_VOL`

---

## Strategy Architecture

| Market Regime | Active Strategy |
|---|---|
| TREND_UP | Momentum |
| TREND_DOWN | Short Momentum |
| RANGE | Mean Reversion |
| HIGH_VOL | Risk-Off / Reduced Exposure |

---

## Technology Stack

- Python
- Pandas
- NumPy
- Requests
- hmmlearn
- Matplotlib *(planned)*
- Backtrader / VectorBT *(planned)*

---

## Project Structure

```text
QUANT/
│
├── data_ingestion/
├── validation/
├── features/
├── regime/
├── strategies/
├── main.py
└── README.md
