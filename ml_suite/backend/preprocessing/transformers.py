"""Custom transformers. These live in a module (not closures or lambdas) so a
fitted pipeline can be pickled and reloaded in a fresh process."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DateTimeFeatures(BaseEstimator, TransformerMixin):
    """Expand each datetime column into ordinal parts a model can use."""

    PARTS = ("year", "month", "day", "dayofweek", "hour")

    def fit(self, X, y=None):
        self.columns_ = list(pd.DataFrame(X).columns)
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).copy()
        out = pd.DataFrame(index=frame.index)
        for col in frame.columns:
            series = pd.to_datetime(frame[col], errors="coerce")
            for part in self.PARTS:
                out[f"{col}__{part}"] = getattr(series.dt, part).astype("float64")
            # Absolute position in time, useful for trends.
            out[f"{col}__epoch"] = series.astype("int64").where(
                series.notna(), np.nan) / 1e9
        return out.fillna(-1.0).to_numpy()

    def get_feature_names_out(self, input_features=None):
        names = input_features if input_features is not None else self.columns_
        return np.array(
            [f"{c}__{p}" for c in names for p in (*self.PARTS, "epoch")]
        )
