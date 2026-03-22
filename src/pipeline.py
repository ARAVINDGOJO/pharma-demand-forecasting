from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.forecasting.backtest import BacktestConfig, backtest_all
from src.forecasting.forecast import forecast_all
from src.forecasting.preprocess import (
    DatasetSpec,
    fill_time_gaps,
    load_demand_csv,
    load_wide_demand_csv,
    melt_wide_demand_dataframe,
    normalize_spec,
    validate_long_demand_dataframe,
)


@dataclass(frozen=True)
class ForecastPipelineParams:
    """Parameters for a full forecast run (CLI and API share this)."""

    wide: bool = False
    date_col: str | None = None
    freq: str = "W"
    horizon: int = 12
    group_cols: tuple[str, ...] = ("sku_id",)
    n_folds: int = 3
    min_train_size: int = 24
    artifacts_dir: str | Path = "artifacts"
    backtest_progress: bool = True


def load_and_prepare_dataframe(
    *,
    csv_path: str | Path,
    params: ForecastPipelineParams,
) -> tuple[pd.DataFrame, DatasetSpec]:
    """Load CSV from disk and apply fill_time_gaps."""
    if params.wide:
        date_col = params.date_col or "datum"
        df = load_wide_demand_csv(str(csv_path), date_col=date_col, group_col_name="sku_id", target_col="y")
        spec = DatasetSpec(date_col=date_col, target_col="y", group_cols=("sku_id",))
    else:
        date_col = params.date_col or "date"
        spec_norm = normalize_spec(group_cols=list(params.group_cols))
        spec = DatasetSpec(date_col=date_col, target_col="y", group_cols=spec_norm.group_cols)
        df = load_demand_csv(str(csv_path), spec)

    df = fill_time_gaps(df, freq=params.freq, spec=spec, fill_target=0.0)
    return df, spec


def load_and_prepare_from_bytes(
    *,
    content: bytes,
    params: ForecastPipelineParams,
) -> tuple[pd.DataFrame, DatasetSpec]:
    """Load CSV from raw bytes (e.g. API upload)."""
    if params.wide:
        date_col = params.date_col or "datum"
        df_raw = pd.read_csv(io.BytesIO(content))
        out = melt_wide_demand_dataframe(df_raw, date_col=date_col, group_col_name="sku_id", target_col="y", source="upload")
        spec = DatasetSpec(date_col=date_col, target_col="y", group_cols=("sku_id",))
    else:
        date_col = params.date_col or "date"
        spec_norm = normalize_spec(group_cols=list(params.group_cols))
        spec = DatasetSpec(date_col=date_col, target_col="y", group_cols=spec_norm.group_cols)
        df_raw = pd.read_csv(io.BytesIO(content))
        out = validate_long_demand_dataframe(df_raw, spec, source="upload")

    out = fill_time_gaps(out, freq=params.freq, spec=spec, fill_target=0.0)
    return out, spec


def run_forecast_pipeline(
    *,
    df: pd.DataFrame,
    spec: DatasetSpec,
    params: ForecastPipelineParams,
) -> dict[str, Any]:
    """
    Run backtest + forecast. Returns dict with DataFrames and optional best-model summary.
    """
    art = Path(params.artifacts_dir)
    art.mkdir(parents=True, exist_ok=True)

    cfg = BacktestConfig(
        freq=params.freq,
        horizon=params.horizon,
        n_folds=params.n_folds,
        min_train_size=params.min_train_size,
    )
    metrics_df, fold_fc_df = backtest_all(df, spec=spec, cfg=cfg, show_progress=params.backtest_progress)

    if not metrics_df.empty:
        metrics_df.to_csv(art / "metrics.csv", index=False)
    if not fold_fc_df.empty:
        fold_fc_df.to_csv(art / "backtest_forecasts.csv", index=False)

    fc = forecast_all(df, spec=spec, freq=params.freq, horizon=params.horizon)
    fc.to_csv(art / "forecasts.csv", index=False)

    best_models: pd.DataFrame | None = None
    if not metrics_df.empty:
        summary = (
            metrics_df.groupby(["sku_id", "model"], as_index=False)["smape"]
            .mean()
            .sort_values(["sku_id", "smape"], ascending=[True, True])
        )
        best_models = summary.groupby("sku_id", as_index=False).head(1)

    return {
        "forecasts": fc,
        "metrics": metrics_df,
        "backtest_forecasts": fold_fc_df,
        "best_models": best_models,
        "artifacts_dir": str(art.resolve()),
    }
