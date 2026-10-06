"""Data split and feature encoding shared by all models.

The Preprocessor is fitted on the TRAIN period only and then applied to
valid/test/API requests, so no information from the future leaks into training.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import (
    CATEGORICAL, FEATURES_PARQUET, MIN_CATEGORY_COUNT, NUMERIC, TARGET,
    TRAIN_END_DAY, VALID_END_DAY,
)

# Heavy-tailed counts/amounts: log1p before scaling keeps the neural net stable.
LOG_FEATURES = [
    "amt", "dist1", "c1", "c2", "c5", "c6", "c9", "c11", "c13", "c14",
    "uid_prev_tx", "uid_prev_mean_amt", "uid_prev_std_amt", "amt_to_uid_mean",
    "uid_tx_last_1h", "uid_tx_last_24h", "card1_tx_last_24h",
    "secs_since_prev_tx", "uid_same_amt_prev",
]

UNKNOWN = 0  # code for missing values and categories unseen in train
RARE = 1     # code for categories seen fewer than MIN_CATEGORY_COUNT times


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(FEATURES_PARQUET)
    train = df[df["day"] < TRAIN_END_DAY]
    valid = df[(df["day"] >= TRAIN_END_DAY) & (df["day"] < VALID_END_DAY)]
    test = df[df["day"] >= VALID_END_DAY]
    return train, valid, test


@dataclass
class Preprocessor:
    vocab: dict[str, dict[str, int]] = field(default_factory=dict)
    mean: dict[str, float] = field(default_factory=dict)
    std: dict[str, float] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        for col in CATEGORICAL:
            counts = df[col].value_counts()
            frequent = counts[counts >= MIN_CATEGORY_COUNT].index
            self.vocab[col] = {value: i + 2 for i, value in enumerate(sorted(frequent))}
        num = self._log_numeric(df)
        self.mean = num.mean().to_dict()
        self.std = num.std().replace(0, 1).to_dict()
        return self

    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col in CATEGORICAL:
            codes = df[col].map(self.vocab[col])
            # Present but not frequent in train -> RARE; missing -> UNKNOWN
            codes = codes.where(codes.notna(), np.where(df[col].isna(), UNKNOWN, RARE))
            out[col] = codes.astype("int64")
        return out

    def cardinalities(self) -> list[int]:
        """Number of distinct codes per categorical column (for embedding tables)."""
        return [len(self.vocab[col]) + 2 for col in CATEGORICAL]

    # --- Gradient boosting input: raw numbers (trees handle NaN and scale) ---
    def transform_tree(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([df[NUMERIC].astype("float32"), self.encode_categorical(df)], axis=1)

    # --- Neural net / linear model input: scaled numbers + missing flags ---
    def transform_dense(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        num = self._log_numeric(df)
        missing = num.isna().astype("float32").add_suffix("_missing")
        scaled = ((num - pd.Series(self.mean)) / pd.Series(self.std)).fillna(0.0)
        x_num = pd.concat([scaled, missing], axis=1).to_numpy(dtype="float32")
        x_cat = self.encode_categorical(df).to_numpy(dtype="int64")
        return x_num, x_cat

    @staticmethod
    def _log_numeric(df: pd.DataFrame) -> pd.DataFrame:
        num = df[NUMERIC].astype("float64").copy()
        for col in LOG_FEATURES:
            num[col] = np.sign(num[col]) * np.log1p(num[col].abs())
        return num


def as_category(x: pd.DataFrame, pre: Preprocessor) -> pd.DataFrame:
    """Mark encoded columns as pandas 'category' with a fixed set of codes,
    so XGBoost/LightGBM split on them as categories, not as ordered numbers."""
    x = x.copy()
    for col, n in zip(CATEGORICAL, pre.cardinalities()):
        x[col] = pd.Categorical(x[col], categories=range(n))
    return x


def target(df: pd.DataFrame) -> np.ndarray:
    return df[TARGET].to_numpy(dtype="float32")
