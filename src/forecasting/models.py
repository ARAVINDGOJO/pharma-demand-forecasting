from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing


@dataclass(frozen=True)
class ForecastResult:
    yhat: np.ndarray


class SeasonalNaive:
    def __init__(self, season_length: int):
        self.season_length = max(1, int(season_length))
        self._last_season: np.ndarray | None = None

    def fit(self, y: np.ndarray) -> "SeasonalNaive":
        y = np.asarray(y, dtype=float)
        if len(y) == 0:
            self._last_season = np.array([0.0])
            return self
        s = min(self.season_length, len(y))
        self._last_season = y[-s:].copy()
        return self

    def predict(self, horizon: int) -> ForecastResult:
        if self._last_season is None:
            raise RuntimeError("Model not fit")
        h = int(horizon)
        if h <= 0:
            return ForecastResult(yhat=np.array([], dtype=float))
        season = self._last_season
        reps = int(np.ceil(h / len(season)))
        yhat = np.tile(season, reps)[:h]
        return ForecastResult(yhat=yhat)


class ETSModel:
    def __init__(self, season_length: int):
        self.season_length = max(1, int(season_length))
        self._fit = None
        self._fallback = None

    def fit(self, y: np.ndarray) -> "ETSModel":
        y = np.asarray(y, dtype=float)
        if len(y) < 3:
            self._fallback = float(np.mean(y)) if len(y) else 0.0
            self._fit = None
            return self
        seasonal = "add" if self.season_length > 1 and len(y) >= 2 * self.season_length else None
        try:
            mod = ExponentialSmoothing(
                y,
                trend="add",
                seasonal=seasonal,
                seasonal_periods=self.season_length if seasonal else None,
                initialization_method="estimated",
            )
            self._fit = mod.fit(optimized=True)
        except Exception:
            self._fit = None
            self._fallback = float(np.mean(y))
        return self

    def predict(self, horizon: int) -> ForecastResult:
        h = int(horizon)
        if h <= 0:
            return ForecastResult(yhat=np.array([], dtype=float))
        if self._fit is None:
            return ForecastResult(yhat=np.full(h, float(getattr(self, "_fallback", 0.0))))
        fc = self._fit.forecast(h)
        return ForecastResult(yhat=np.asarray(fc, dtype=float))


class RidgeLagModel:
    """
    Global-ish model: trained per-series in this template for simplicity.
    You can later upgrade to a single global model across all series.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self._pipe: Pipeline | None = None
        self._feature_cols: list[str] | None = None

    def fit(self, df: pd.DataFrame, feature_cols: list[str], target_col: str) -> "RidgeLagModel":
        train = df.loc[~df["_is_future"]].copy()
        train = train.dropna(subset=feature_cols + [target_col])
        X = train[feature_cols].to_numpy(dtype=float)
        y = train[target_col].to_numpy(dtype=float)
        if len(y) < 5:
            self._pipe = None
            self._feature_cols = feature_cols
            return self
        self._pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=self.alpha))])
        self._pipe.fit(X, y)
        self._feature_cols = feature_cols
        return self

    def predict(self, df_future: pd.DataFrame) -> ForecastResult:
        if self._feature_cols is None:
            raise RuntimeError("Model not fit")
        h = int(df_future.shape[0])
        if h == 0:
            return ForecastResult(yhat=np.array([], dtype=float))
        if self._pipe is None:
            # fallback to simple mean of available lag_1 if present
            if "lag_1" in df_future.columns and not df_future["lag_1"].isna().all():
                base = float(df_future["lag_1"].dropna().iloc[0])
            else:
                base = 0.0
            return ForecastResult(yhat=np.full(h, base))
        X = df_future[self._feature_cols].to_numpy(dtype=float)
        yhat = self._pipe.predict(X)
        yhat = np.maximum(0.0, np.asarray(yhat, dtype=float))  # demand can't be negative
        return ForecastResult(yhat=yhat)

