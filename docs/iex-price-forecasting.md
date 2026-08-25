
# Day-Ahead Electricity Price Forecasting — Indian Power Market (IEX)

Forecasts the next day's hourly Market Clearing Price (MCP) on the Indian Energy
Exchange (IEX) Day-Ahead Market, combining grid demand, weather, and calendar
signals. A TensorFlow sequence model is benchmarked against classical baselines,
validated with walk-forward backtesting, and wrapped in a drift-monitoring loop.

The problem it solves: participants in the DAM (discoms, generators, C&I buyers)
must decide next-day procurement before prices are known. Under-forecasting means
buying short into scarcity-priced blocks; over-forecasting wastes budget or
over-commits. Accurate short-horizon price forecasts reduce that procurement risk.

> **Data note.** The repo ships with a mechanistic synthetic data generator that
> reproduces the real drivers of DAM prices — demand seasonality, a convex supply
> stack, weather-driven cooling load, and calm/volatile regime switching — so the
> full pipeline runs out of the box. Swap in real data via `src/data_ingestion.load_real()`
> (IEX MCP + Grid-India demand + open weather) and nothing else changes.

## Architecture

```
Data sources ──▶ Validation ──▶ Feature engineering ──▶ Models ──▶ Backtest ──▶ Monitoring
(IEX, Grid-India,  (gaps,          (calendar, lags,       (naive,     (walk-       (drift PSI,
 weather, calendar)  outliers,       rolling, HMM regime)   XGBoost,    forward,      retrain
                     price caps)                            LSTM)       MAE/sMAPE)    trigger)
                                                                                        │
                                                                        retrain ◀───────┘
```

## Results (representative 1-year run, held-out test set)

| Model            | MAE (₹/MWh) | RMSE  | sMAPE |
|------------------|-------------|-------|-------|
| Seasonal-naive   | 1485        | 2069  | 40.7% |
| XGBoost          | 1057        | 1455  | 31.0% |
| **LSTM (TF)**    | **982**     | 1517  | **28.0%** |

(Numbers vary with data span and seed; this run used all 24 features including
weather and demand. Retrain on real IEX data for real figures.)

The LSTM wins on MAE and sMAPE; XGBoost is marginally better on RMSE (large
spikes). That honest split — rather than a single "our model is best" number —
is the intended takeaway: the deep model earns its complexity on average error,
while the strong tabular baseline stays competitive on tail events.

See `outputs/` for `forecast_vs_actual.png`, `error_by_horizon.png`, and
`model_comparison.png`.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/train.py
```

Runs the whole pipeline in a few minutes on CPU and writes metrics + plots to
`outputs/`.

## Layout

```
config.py                  every tunable (horizon, lookback, splits, seeds)
src/
  data_ingestion.py        synthetic generator + real-data loader stub
  data_validation.py       schema, gap-fill, price-cap, outlier flagging
  feature_engineering.py   calendar / lag / rolling / HMM-regime features
  windowing.py             tf.data sliding-window builder + train-only scaler
  models.py                SeasonalNaive, XGBoost, LSTM, Conv1D
  backtest.py              MAE/RMSE/sMAPE, error-by-horizon, walk-forward splits
  monitoring.py            PSI drift + retrain decision
data/
  fetch_weather.py         Open-Meteo -> weather.csv
  parse_iex.py             IEX .xlsx  -> price.csv
  parse_griddata.py        Grid-India .pdf -> demand.csv
  build_dataset.py         merge -> merged_dataset.csv
scripts/train.py           end-to-end runner (--data for real CSV)
```

## Design decisions worth knowing (interview-ready)

- **Walk-forward, not random split.** A shuffled train/test split leaks the
  future into the past for time series. `backtest.walk_forward_splits` uses an
  expanding window; the scaler is fit on train only.
- **sMAPE over MAPE.** Electricity prices sit near a floor at times, where plain
  MAPE explodes from near-zero denominators. sMAPE is bounded and symmetric.
- **Baselines are load-bearing.** Seasonal-naive is the bar every model must
  clear; XGBoost is the "do you even need deep learning?" test.
- **HMM regime feature.** A 2-state Gaussian HMM tags each block calm vs volatile
  — calm and scarcity regimes forecast very differently. (In a strict backtest,
  fit the HMM on the train slice only; the reference build documents this caveat.)
- **Multi-output single-shot.** The LSTM predicts all 24 blocks at once
  (`Dense(24)`), avoiding autoregressive error accumulation across the day.

## Swapping in real data — the `data/` toolkit

Scripts to turn the real downloads into a model-ready file:

```
data/
  fetch_weather.py    Open-Meteo historical API  -> data/weather.csv   (hourly, no key)
  parse_iex.py        IEX DAM snapshot .xlsx      -> data/price.csv     (auto-detects columns)
  parse_griddata.py   Grid-India monthly .pdf     -> data/demand.csv    (DAILY, pdfplumber)
  build_dataset.py    merge all three            -> data/merged_dataset.csv
```

Workflow:

```bash
# 1. Weather (fully automated; runs on your machine, needs internet)
python data/fetch_weather.py                       # edit LAT/LON + dates at top

# 2. Price — download DAM "Market Snapshot" .xlsx from iexindia.com into downloads/
python data/parse_iex.py "downloads/iex_*.xlsx"    # prints the columns it detected

# 3. Demand — download Grid-India monthly report PDFs into downloads/
python data/parse_griddata.py "downloads/grid_*.pdf"          # try this first
python data/parse_griddata.py "downloads/grid_*.pdf" --inspect  # if it misses, see raw tables

# 4. Merge -> one hourly CSV
python data/build_dataset.py

# 5. Train on the real data
python scripts/train.py --data data/merged_dataset.csv
```

Notes:
- **Price is the only required source.** Weather and demand are joined if present.
- **Demand is daily**, broadcast across each day's 24 hourly blocks — it cannot
  be hourly from monthly PDFs. Price lags + weather already carry most of the
  intraday signal, so treat demand as a bonus feature.
- The IEX and Grid-India layouts drift over time; both parsers print what they
  detected and expose hint lists / an `--inspect` mode so you can adapt them.
- Set `blocks_per_day = 96` in `config.py` (and `RESAMPLE_TO_HOURLY = False` in
  `parse_iex.py`) for true 15-minute granularity.
