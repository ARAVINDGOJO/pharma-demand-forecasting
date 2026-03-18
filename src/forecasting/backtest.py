from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

from .metrics import mae, mase, rmse, smape
from .models import ETSModel, RidgeLagModel, SeasonalNaive
from .preprocess import DatasetSpec, extend_future_frame, infer_season_length, make_features


@dataclass(frozen=True)
class BacktestConfig:
    freq: str
    horizon: int
    n_folds: int = 3
    min_train_size: int = 24


def _rolling_cutoffs(n: int, horizon: int, n_folds: int, min_train: int) -> list[int]:
    """
    Returns train_end indices (exclusive) for each fold.
    """
    # last fold ends at n-horizon
    last = n - horizon
    if last <= min_train:
        return []
    step = max(1, horizon)
    cutoffs = []
    c = last
    while c > min_train and len(cutoffs) < n_folds:
        cutoffs.append(c)
        c -= step
    return list(reversed(cutoffs))


def backtest_one_series(
    g: pd.DataFrame,
    *,
    spec: DatasetSpec,
    cfg: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (metrics_df, fold_forecasts_df) for one series.
    """
    date_col = spec.date_col
    target_col = spec.target_col
    season = infer_season_length(cfg.freq)

    g = g.sort_values(date_col).reset_index(drop=True)
    y_all = g[target_col].to_numpy(dtype=float)
    cutoffs = _rolling_cutoffs(len(g), cfg.horizon, cfg.n_folds, cfg.min_train_size)
    if not cutoffs:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    metrics_rows = []
    fc_rows = []
    for fold_i, train_end in enumerate(cutoffs, start=1):
        train = g.iloc[:train_end].copy()
        test = g.iloc[train_end : train_end + cfg.horizon].copy()
        if len(test) < cfg.horizon:
            continue

        y_train = train[target_col].to_numpy(dtype=float)
        y_test = test[target_col].to_numpy(dtype=float)

        models = {
            "seasonal_naive": SeasonalNaive(season).fit(y_train),
            "ets": ETSModel(season).fit(y_train),
        }

        # ML model needs features + iterative forecasting.
        hist_plus_future = extend_future_frame(train, freq=cfg.freq, spec=spec, horizon=cfg.horizon)
        feat_df, feat_cols = make_features(hist_plus_future, freq=cfg.freq, spec=spec, horizon=cfg.horizon)
        ml = RidgeLagModel(alpha=1.0).fit(feat_df, feat_cols, target_col=target_col)

        # iterative: fill predictions so lags roll forward
        feat_df = feat_df.sort_values(date_col).reset_index(drop=True)
        future_mask = feat_df["_is_future"].to_numpy()
        future_idx = np.where(future_mask)[0]
        yhat_ml = []
        for j, idx in enumerate(future_idx):
            row = feat_df.iloc[[idx]].copy()
            # recompute features for this row (lags already computed from filled history)
            pred = ml.predict(row).yhat[0]
            yhat_ml.append(pred)
            feat_df.loc[idx, target_col] = pred
            # update subsequent lag columns affected by this new target:
            # easiest: recompute all features after each step for correctness
            if j < len(future_idx) - 1:
                feat_df2, feat_cols2 = make_features(feat_df, freq=cfg.freq, spec=spec, horizon=cfg.horizon)
                feat_df = feat_df2
                feat_cols = feat_cols2
                ml = RidgeLagModel(alpha=1.0).fit(
                    feat_df.loc[~feat_df["_is_future"]],
                    feat_cols,
                    target_col=target_col,
                )

        yhat_ml = np.asarray(yhat_ml, dtype=float)
        preds = {
            "ridge_lags": yhat_ml,
            "seasonal_naive": models["seasonal_naive"].predict(cfg.horizon).yhat,
            "ets": models["ets"].predict(cfg.horizon).yhat,
        }

        for name, yhat in preds.items():
            row = {
                "fold": fold_i,
                "model": name,
                "mae": mae(y_test, yhat),
                "rmse": rmse(y_test, yhat),
                "smape": smape(y_test, yhat),
                "mase": mase(y_test, yhat, y_train, season_length=season),
                "train_end_date": train[date_col].max(),
                "test_start_date": test[date_col].min(),
                "test_end_date": test[date_col].max(),
            }
            metrics_rows.append(row)

            fold_fc = test[[date_col]].copy()
            fold_fc["fold"] = fold_i
            fold_fc["model"] = name
            fold_fc["y"] = y_test
            fold_fc["yhat"] = yhat
            fc_rows.append(fold_fc)

    metrics_df = pd.DataFrame(metrics_rows)
    fc_df = pd.concat(fc_rows, ignore_index=True) if fc_rows else pd.DataFrame()
    return metrics_df, fc_df


def backtest_all(
    df: pd.DataFrame,
    *,
    spec: DatasetSpec,
    cfg: BacktestConfig,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = list(spec.group_cols)
    metrics_all = []
    fc_all = []

    iterator = df.groupby(group_cols, sort=False)
    if show_progress:
        iterator = tqdm(list(iterator), desc="Backtesting series")

    for key, g in iterator:
        m, f = backtest_one_series(g, spec=spec, cfg=cfg)
        if m.empty:
            continue
        if not isinstance(key, tuple):
            key = (key,)
        for c, v in zip(group_cols, key):
            m[c] = v
            if not f.empty:
                f[c] = v
        metrics_all.append(m)
        if not f.empty:
            fc_all.append(f)

    metrics_df = pd.concat(metrics_all, ignore_index=True) if metrics_all else pd.DataFrame()
    fc_df = pd.concat(fc_all, ignore_index=True) if fc_all else pd.DataFrame()
    return metrics_df, fc_df

