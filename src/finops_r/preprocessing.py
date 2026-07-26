from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .constants import BASE_NUMERIC_FEATURES, CATEGORICAL_FEATURES, RSOP_FEATURES


def feature_columns(use_rsop: bool) -> tuple[list[str], list[str]]:
    numeric = list(BASE_NUMERIC_FEATURES)
    if use_rsop:
        numeric.extend(RSOP_FEATURES)
    return numeric, list(CATEGORICAL_FEATURES)


def build_preprocessor(use_rsop: bool) -> ColumnTransformer:
    numeric, categorical = feature_columns(use_rsop)
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)],
        remainder="drop",
        sparse_threshold=0.0,
    )


def select_features(frame: pd.DataFrame, use_rsop: bool) -> pd.DataFrame:
    numeric, categorical = feature_columns(use_rsop)
    return frame[list(numeric) + list(categorical)]
