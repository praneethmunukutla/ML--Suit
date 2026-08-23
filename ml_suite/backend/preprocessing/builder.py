"""Builds the ColumnTransformer that turns a raw DataFrame into a numeric
matrix. The result is fitted as part of the model Pipeline, so a saved model
carries its own imputers, scalers, and encoders with it."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from backend.core.exceptions import ValidationError
from backend.core.logging_config import get_logger
from backend.preprocessing import profiler
from backend.preprocessing.transformers import DateTimeFeatures

log = get_logger(__name__)


def split_columns(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, list[str]]:
    """Group the selected features by the treatment each one needs."""
    n_rows = len(df)
    groups: dict[str, list[str]] = {
        "numeric": [], "categorical": [], "high_card": [], "datetime": [], "dropped": []
    }
    for col in feature_cols:
        kind = profiler.classify_column(df[col], n_rows)
        if kind == profiler.NUMERIC:
            groups["numeric"].append(col)
        elif kind == profiler.BOOLEAN:
            groups["categorical"].append(col)
        elif kind == profiler.CATEGORICAL:
            groups["categorical"].append(col)
        elif kind == profiler.HIGH_CARD:
            groups["high_card"].append(col)
        elif kind == profiler.DATETIME:
            groups["datetime"].append(col)
        else:  # identifier or empty carries no signal
            groups["dropped"].append(col)
    return groups


def build_preprocessor(df: pd.DataFrame, feature_cols: list[str]) -> tuple[ColumnTransformer, dict]:
    """Assemble the preprocessing stage and report what it decided."""
    groups = split_columns(df, feature_cols)
    used = groups["numeric"] + groups["categorical"] + groups["high_card"] + groups["datetime"]
    if not used:
        raise ValidationError(
            "None of the selected features are usable",
            detail=f"Dropped as identifiers or empty: {groups['dropped']}",
        )

    transformers = []
    if groups["numeric"]:
        transformers.append((
            "numeric",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            groups["numeric"],
        ))
    if groups["categorical"]:
        transformers.append((
            "categorical",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                # unknown categories at predict time must not raise
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            groups["categorical"],
        ))
    if groups["high_card"]:
        transformers.append((
            "high_cardinality",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OrdinalEncoder(handle_unknown="use_encoded_value",
                                          unknown_value=-1)),
                ("scale", StandardScaler()),
            ]),
            groups["high_card"],
        ))
    if groups["datetime"]:
        transformers.append((
            "datetime",
            Pipeline([
                ("expand", DateTimeFeatures()),
                ("scale", StandardScaler()),
            ]),
            groups["datetime"],
        ))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    report = {
        "numeric": groups["numeric"],
        "categorical": groups["categorical"],
        "high_cardinality": groups["high_card"],
        "datetime": groups["datetime"],
        "dropped": groups["dropped"],
        "used_features": used,
    }
    log.info(
        "Preprocessor: %d numeric, %d categorical, %d high-card, %d datetime, %d dropped",
        len(groups["numeric"]), len(groups["categorical"]),
        len(groups["high_card"]), len(groups["datetime"]), len(groups["dropped"]),
    )
    return preprocessor, report


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Names after encoding, for feature-importance charts."""
    try:
        return [str(n) for n in preprocessor.get_feature_names_out()]
    except Exception:  # not every transformer implements it cleanly
        return []
