from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    date_col: str = "date"
    target_col: str = "y"
    group_cols: tuple[str, ...] = ("sku_id",)


def _normalize_group_cols(group_cols: Iterable[str] | None) -> tuple[str, ...]:
    if not group_cols:
        return ("sku_id",)
    cols = tuple([c for c in group_cols if c])
    if not cols:
        return ("sku_id",)
    return cols


def load_demand_csv(path: str, spec: DatasetSpec) -> pd.DataFrame:
    df = pd.read_csv(path)
    if spec.date_col not in df.columns:
        raise ValueError(f"Missing date column '{spec.date_col}' in {path}")
    for c in spec.group_cols:
        if c not in df.columns:
            raise ValueError(f"Missing group column '{c}' in {path}")
    if spec.target_col not in df.columns:
        raise ValueError(f"Missing target column '{spec.target_col}' in {path}")

    df = df.copy()
    df[spec.date_col] = pd.to_datetime(df[spec.date_col], errors="raise")
    df = df.sort_values(list(spec.group_cols) + [spec.date_col])
    df[spec.target_col] = pd.to_numeric(df[spec.target_col], errors="coerce").astype(float)
    if df[spec.target_col].isna().any():
        bad = int(df[spec.target_col].isna().sum())
        raise ValueError(f"Found {bad} rows with non-numeric '{spec.target_col}'. Fix or drop them.")
    return df


def load_wide_demand_csv(
    path: str,
    *,
    date_col: str = "datum",
    group_col_name: str = "sku_id",
    target_col: str = "y",
) -> pd.DataFrame:
    """
    Loads "wide" demand data where each SKU is a column.

    Example columns: datum, M01AB, M01AE, N02BA, ...

    Returns a "long" dataframe with columns: [date, sku_id, y].
    """
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"Missing date column '{date_col}' in {path}")

    value_cols = [c for c in df.columns if c != date_col]
    if not value_cols:
        raise ValueError(f"No SKU columns found (only '{date_col}' present) in {path}")

    out = df.melt(id_vars=[date_col], value_vars=value_cols, var_name=group_col_name, value_name=target_col)
    out = out.dropna(subset=[target_col]).copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="raise")
    out[target_col] = pd.to_numeric(out[target_col], errors="coerce").astype(float)
    if out[target_col].isna().any():
        bad = int(out[target_col].isna().sum())
        raise ValueError(f"Found {bad} rows with non-numeric '{target_col}'. Fix or drop them.")

    out = out.sort_values([group_col_name, date_col]).reset_index(drop=True)
    return out


def infer_season_length(freq: str) -> int:
    f = (freq or "").upper().strip()
    if f in {"D"}:
        return 7
    if f in {"W", "W-SUN", "W-MON"}:
        return 52
    if f in {"M", "MS"}:
        return 12
    if f in {"Q", "QS"}:
        return 4
    return 1


def fill_time_gaps(
    df: pd.DataFrame,
    *,
    freq: str,
    spec: DatasetSpec,
    fill_target: float = 0.0,
) -> pd.DataFrame:
    """
    Ensures each series has a complete date index at the specified freq.
    Missing target values are filled with fill_target (default: 0).
    """
    date_col = spec.date_col
    target_col = spec.target_col
    group_cols = list(spec.group_cols)

    out = []
    for key, g in df.groupby(group_cols, sort=False):
        g = g.sort_values(date_col).copy()
        idx = pd.date_range(g[date_col].min(), g[date_col].max(), freq=freq)
        g2 = g.set_index(date_col).reindex(idx)
        g2.index.name = date_col
        if not isinstance(key, tuple):
            key = (key,)
        for c, v in zip(group_cols, key):
            g2[c] = v
        if target_col in g2.columns:
            g2[target_col] = g2[target_col].fillna(fill_target).astype(float)
        else:
            g2[target_col] = float(fill_target)
        # preserve other columns via forward fill (safe default)
        other_cols = [c for c in g.columns if c not in (group_cols + [date_col, target_col])]
        for c in other_cols:
            g2[c] = g2[c].ffill()
        out.append(g2.reset_index())

    return pd.concat(out, ignore_index=True)


def make_features(
    df: pd.DataFrame,
    *,
    freq: str,
    spec: DatasetSpec,
    horizon: int,
    lags: tuple[int, ...] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Creates lag + calendar features for ML model.
    Returns (df_with_features, feature_cols).
    """
    date_col = spec.date_col
    target_col = spec.target_col
    group_cols = list(spec.group_cols)

    if lags is None:
        season = infer_season_length(freq)
        lags = tuple(sorted(set([1, 2, 3, 4, 7, season, season * 2] if season > 1 else [1, 2, 3, 4, 7])))

    df = df.copy()
    df = df.sort_values(group_cols + [date_col])

    # calendar features
    dt = pd.to_datetime(df[date_col])
    df["dow"] = dt.dt.dayofweek.astype(int)
    df["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    df["month"] = dt.dt.month.astype(int)
    df["year"] = dt.dt.year.astype(int)

    # lags per series
    for L in lags:
        df[f"lag_{L}"] = df.groupby(group_cols, sort=False)[target_col].shift(L)

    # rolling means
    for w in (3, 7, 14, 28):
        # Use transform to keep index aligned and avoid MultiIndex pitfalls across pandas versions.
        df[f"roll_mean_{w}"] = df.groupby(group_cols, sort=False)[target_col].transform(
            lambda s: s.shift(1).rolling(window=w, min_periods=max(1, w // 2)).mean()
        )

    # mark forecast rows (future)
    if "_is_future" not in df.columns:
        df["_is_future"] = False

    feature_cols = [c for c in df.columns if c.startswith("lag_") or c.startswith("roll_")] + [
        "dow",
        "weekofyear",
        "month",
        "year",
    ]

    # include known regressors if present
    for c in ["price", "promo", "stockout"]:
        if c in df.columns:
            feature_cols.append(c)

    # drop features that are entirely NA
    feature_cols = [c for c in feature_cols if not df[c].isna().all()]
    return df, feature_cols


def extend_future_frame(
    history: pd.DataFrame,
    *,
    freq: str,
    spec: DatasetSpec,
    horizon: int,
) -> pd.DataFrame:
    date_col = spec.date_col
    target_col = spec.target_col
    group_cols = list(spec.group_cols)

    history = history.copy()
    history["_is_future"] = False
    rows = [history]
    for key, g in history.groupby(group_cols, sort=False):
        last = pd.to_datetime(g[date_col].max())
        future_dates = pd.date_range(last, periods=horizon + 1, freq=freq)[1:]
        fut = pd.DataFrame({date_col: future_dates})
        if not isinstance(key, tuple):
            key = (key,)
        for c, v in zip(group_cols, key):
            fut[c] = v
        fut[target_col] = np.nan
        fut["_is_future"] = True

        # carry forward known regressors if present (best-effort defaults)
        for c in ["price", "promo", "stockout"]:
            if c in history.columns:
                if c in g.columns and not g[c].isna().all():
                    fut[c] = g[c].iloc[-1]
                else:
                    fut[c] = 0.0
        rows.append(fut)

    return pd.concat(rows, ignore_index=True)


def normalize_spec(*, group_cols: Iterable[str] | None) -> DatasetSpec:
    return DatasetSpec(group_cols=_normalize_group_cols(group_cols))

