from __future__ import annotations

import numpy as np
import pandas as pd

from .models import ETSModel, RidgeLagModel, SeasonalNaive
from .preprocess import DatasetSpec, extend_future_frame, infer_season_length, make_features


def forecast_one_series(
    g: pd.DataFrame,
    *,
    spec: DatasetSpec,
    freq: str,
    horizon: int,
) -> pd.DataFrame:
    date_col = spec.date_col
    target_col = spec.target_col
    season = infer_season_length(freq)

    g = g.sort_values(date_col).reset_index(drop=True)
    y = g[target_col].to_numpy(dtype=float)

    sn = SeasonalNaive(season).fit(y)
    ets = ETSModel(season).fit(y)

    # ML: iterative, using future frame
    hist_plus_future = extend_future_frame(g, freq=freq, spec=spec, horizon=horizon)
    feat_df, feat_cols = make_features(hist_plus_future, freq=freq, spec=spec, horizon=horizon)
    ml = RidgeLagModel(alpha=1.0).fit(feat_df, feat_cols, target_col=target_col)

    feat_df = feat_df.sort_values(date_col).reset_index(drop=True)
    future_mask = feat_df["_is_future"].to_numpy()
    future_idx = np.where(future_mask)[0]
    yhat_ml = []
    for j, idx in enumerate(future_idx):
        row = feat_df.iloc[[idx]].copy()
        pred = ml.predict(row).yhat[0]
        yhat_ml.append(pred)
        feat_df.loc[idx, target_col] = pred
        if j < len(future_idx) - 1:
            feat_df2, feat_cols2 = make_features(feat_df, freq=freq, spec=spec, horizon=horizon)
            feat_df = feat_df2
            feat_cols = feat_cols2
            ml = RidgeLagModel(alpha=1.0).fit(
                feat_df.loc[~feat_df["_is_future"]],
                feat_cols,
                target_col=target_col,
            )

    future_dates = feat_df.loc[feat_df["_is_future"], date_col].to_numpy()
    out = pd.DataFrame({date_col: future_dates})
    out["model"] = "ridge_lags"
    out["yhat"] = np.asarray(yhat_ml, dtype=float)

    out_sn = pd.DataFrame({date_col: future_dates, "model": "seasonal_naive", "yhat": sn.predict(horizon).yhat})
    out_ets = pd.DataFrame({date_col: future_dates, "model": "ets", "yhat": ets.predict(horizon).yhat})
    out = pd.concat([out, out_sn, out_ets], ignore_index=True)
    out["yhat"] = np.maximum(0.0, out["yhat"].astype(float))
    return out


def forecast_all(
    df: pd.DataFrame,
    *,
    spec: DatasetSpec,
    freq: str,
    horizon: int,
) -> pd.DataFrame:
    group_cols = list(spec.group_cols)
    rows = []
    for key, g in df.groupby(group_cols, sort=False):
        fc = forecast_one_series(g, spec=spec, freq=freq, horizon=horizon)
        if not isinstance(key, tuple):
            key = (key,)
        for c, v in zip(group_cols, key):
            fc[c] = v
        rows.append(fc)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

