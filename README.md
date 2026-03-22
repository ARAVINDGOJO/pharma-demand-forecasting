# Pharma Demand Forecasting (Starter Project)

This repo is a practical starter template for **demand forecasting in pharmaceuticals**:

- Multi-SKU time series support (`sku_id` required)
- Optional segmentation by `location_id` (store/warehouse/region)
- Backtesting (rolling-origin) with common metrics (MAE, RMSE, sMAPE, MASE)
- Baselines + ML model with lag/seasonal features
- Output forecasts to CSV for planning

## 1) Input data format

Provide a CSV with at least:

| column | required | example |
|---|---:|---|
| `date` | yes | `2026-01-31` |
| `sku_id` | yes | `AMOX500_CAP_100` |
| `y` (demand) | yes | `124` |
| `location_id` | no | `WH_01` |
| `price` | no | `3.25` |
| `promo` | no | `0/1` |
| `stockout` | no | `0/1` |

Notes:
- Use **one row per (date, sku_id[, location_id])** at your chosen frequency (daily/weekly/monthly).
- If you have stockouts, keep the row and set `stockout=1` so you can later adjust training logic.

### Wide format (like your screenshot)

If your file is like:
- one date column (often `datum`)
- many SKU columns (`M01AB`, `M01AE`, `N02BA`, ...)

Run with `--wide` and set the date column name:

```bash
python -m src.run_forecast --data data/pharma.csv --wide --date_col datum --freq M --horizon 6
```

## 2) Setup

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## 3) Quickstart

Put your data at `data/demand.csv`, then run:

```bash
python -m src.run_forecast --data data/demand.csv --freq W --horizon 12 --group_cols sku_id location_id
```

Outputs:
- `artifacts/metrics.csv` (backtest metrics by series + model)
- `artifacts/forecasts.csv` (future forecasts by series + model)

## 4) Models included

- **Seasonal Naive**: good baseline for strong seasonality
- **ETS (Exponential Smoothing)**: robust univariate baseline
- **Ridge Regression with lag features**: scalable across many SKUs and supports exogenous regressors (price/promo)

## 5) Pharma-specific considerations (what to add next)

- **Intermittent demand** (slow movers): Croston/SBA/TSB style models
- **New product launches**: cold-start using analog SKUs + attribute-based models
- **Substitutions & cannibalization**: cross-SKU features, grouped models
- **Regulatory/recall shocks**: event flags and intervention logic
- **Inventory-aware training**: down-weight or filter stockout periods to avoid learning “capped” demand

## 6) Command help

```bash
python -m src.run_forecast --help
```

## 7) Run as a live API (server)

Install deps (includes FastAPI + Uvicorn), then from the project folder:

```bash
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```

- **Health check**: `GET http://localhost:8000/health`
- **Forecast** (upload CSV): `POST http://localhost:8000/forecast`
  - Form field: `file` = your `.csv`
  - Query params (examples): `wide=true`, `date_col=datum`, `freq=M`, `horizon=6`, `include_rows=200`

Example with **curl** (wide monthly CSV):

```bash
curl -X POST "http://localhost:8000/forecast?wide=true&date_col=datum&freq=M&horizon=6&include_rows=50" -F "file=@D:/Downloads/salesmonthly.csv"
```

The API returns JSON (sample rows + `best_models_by_smape`) and also writes CSVs under **`artifacts/`** on the server (or `artifacts_dir` you pass).

## 8) Schedule automatic runs (Windows)

Use **Task Scheduler** to run the CLI on a schedule (daily/weekly/monthly):

1. Action: **Start a program**
2. Program: `python` (full path to your `python.exe` if needed)
3. Arguments: `-m src.run_forecast --data "D:\path\to\salesmonthly.csv" --wide --date_col datum --freq M --horizon 6`
4. Start in: `E:\New folder` (your project folder)

Outputs land in `E:\New folder\artifacts\` each run.

