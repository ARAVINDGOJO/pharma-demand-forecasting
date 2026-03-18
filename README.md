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

