"""
Live API server for demand forecasting.

Run locally:
  uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

Then POST a CSV to /forecast (multipart form: file + optional query params).
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from src.pipeline import ForecastPipelineParams, load_and_prepare_from_bytes, run_forecast_pipeline

app = FastAPI(title="Pharma Demand Forecasting", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/forecast")
async def forecast(
    file: UploadFile = File(..., description="CSV: long format (date, sku_id, y) or wide with --wide"),
    wide: bool = Query(False, description="Wide CSV: one date column + many SKU columns"),
    date_col: str | None = Query(None, description="Date column name (default: date or datum if wide)"),
    freq: str = Query("W", description="Pandas frequency: D, W, ME, M, Q"),
    horizon: int = Query(12, ge=1, le=120),
    n_folds: int = Query(3, ge=1, le=20),
    min_train_size: int = Query(24, ge=1, le=500),
    artifacts_dir: str = Query("artifacts", description="Where to save CSV outputs on server"),
    include_rows: int = Query(500, ge=0, le=50000, description="Max forecast rows in JSON (0 = summary only)"),
) -> JSONResponse:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    params = ForecastPipelineParams(
        wide=wide,
        date_col=date_col,
        freq=freq,
        horizon=horizon,
        group_cols=("sku_id",),
        n_folds=n_folds,
        min_train_size=min_train_size,
        artifacts_dir=artifacts_dir,
        backtest_progress=False,
    )
    try:
        df, spec = load_and_prepare_from_bytes(content=content, params=params)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    out = run_forecast_pipeline(df=df, spec=spec, params=params)
    fc = out["forecasts"]
    metrics = out["metrics"]
    best = out["best_models"]

    metrics_empty = metrics is None or metrics.empty
    payload: dict = {
        "artifacts_dir": out["artifacts_dir"],
        "forecast_rows_total": int(len(fc)),
        "metrics_rows_total": 0 if metrics_empty else int(len(metrics)),
        "files_written": {
            "forecasts": str(Path(artifacts_dir) / "forecasts.csv"),
            "metrics": None if metrics_empty else str(Path(artifacts_dir) / "metrics.csv"),
            "backtest_forecasts": None
            if out["backtest_forecasts"].empty
            else str(Path(artifacts_dir) / "backtest_forecasts.csv"),
        },
    }

    if best is not None and not best.empty:
        payload["best_models_by_smape"] = best.to_dict(orient="records")

    if include_rows > 0 and fc is not None and not fc.empty:
        take = min(include_rows, len(fc))
        payload["forecasts_sample"] = fc.head(take).to_dict(orient="records")

    return JSONResponse(content=payload)
