from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.pipeline import ForecastPipelineParams, load_and_prepare_dataframe, run_forecast_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Demand forecasting (pharma starter).")
    p.add_argument("--data", required=True, help="Path to demand CSV (needs date, y, sku_id[, location_id]).")
    p.add_argument(
        "--wide",
        action="store_true",
        help="Treat input as wide format: one date column + many SKU columns (e.g., datum, M01AB, M01AE...).",
    )
    p.add_argument(
        "--date_col",
        default=None,
        help="Date column name. Defaults: 'date' (long format) or 'datum' (wide format).",
    )
    p.add_argument("--freq", default="W", help="Pandas frequency code: D/W/M/Q. Default: W.")
    p.add_argument("--horizon", type=int, default=12, help="Forecast horizon in periods of --freq. Default: 12.")
    p.add_argument(
        "--group_cols",
        nargs="*",
        default=["sku_id"],
        help="Grouping columns. Example: --group_cols sku_id location_id",
    )
    p.add_argument("--artifacts_dir", default="artifacts", help="Where to write outputs. Default: artifacts/")
    p.add_argument("--n_folds", type=int, default=3, help="Backtest folds. Default: 3.")
    p.add_argument("--min_train_size", type=int, default=24, help="Min train points per series. Default: 24.")
    p.add_argument(
        "--preview",
        action="store_true",
        help="Print a small preview of outputs (forecasts head + best models) to the terminal.",
    )
    p.add_argument("--preview_rows", type=int, default=20, help="Rows to show for --preview. Default: 20.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    params = ForecastPipelineParams(
        wide=args.wide,
        date_col=args.date_col,
        freq=args.freq,
        horizon=args.horizon,
        group_cols=tuple(args.group_cols) if args.group_cols else ("sku_id",),
        n_folds=args.n_folds,
        min_train_size=args.min_train_size,
        artifacts_dir=args.artifacts_dir,
        backtest_progress=True,
    )

    df, spec = load_and_prepare_dataframe(csv_path=args.data, params=params)
    out = run_forecast_pipeline(df=df, spec=spec, params=params)

    fc = out["forecasts"]
    metrics_df = out["metrics"]

    art = Path(params.artifacts_dir)
    print(f"Wrote: {art / 'forecasts.csv'}")
    if not metrics_df.empty:
        print(f"Wrote: {art / 'metrics.csv'}")
        print(f"Wrote: {art / 'backtest_forecasts.csv'}")

    if args.preview:
        n = max(1, int(args.preview_rows))
        print("\n=== Preview: forecasts (first rows) ===")
        with pd.option_context("display.max_rows", n, "display.width", 120, "display.max_columns", 50):
            print(fc.head(n).to_string(index=False))

        if not metrics_df.empty:
            print("\n=== Preview: best models by avg sMAPE (lower is better) ===")
            summary = (
                metrics_df.groupby(["sku_id", "model"], as_index=False)["smape"]
                .mean()
                .sort_values(["sku_id", "smape"], ascending=[True, True])
            )
            best = summary.groupby("sku_id", as_index=False).head(1)
            with pd.option_context("display.width", 120, "display.max_columns", 20):
                print(best.to_string(index=False))


if __name__ == "__main__":
    main()
