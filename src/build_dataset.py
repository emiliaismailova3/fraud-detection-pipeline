"""Export the SQL feature table to Parquet for model training.

Parquet is a columnar file format: much smaller and faster to read than CSV,
and it keeps column types, so we do not re-parse numbers on every run.

Run:  python -m src.build_dataset
"""
import pandas as pd

from src.config import CATEGORICAL, FEATURES_PARQUET, NUMERIC, PROCESSED, TARGET
from src.db import get_engine

COLUMNS = ["transaction_id", "transaction_dt", "day", TARGET, *NUMERIC, *CATEGORICAL]


def main() -> None:
    query = f"SELECT {', '.join(COLUMNS)} FROM features.transactions ORDER BY transaction_dt"
    df = pd.read_sql(query, get_engine())

    # Identifier columns come out of Postgres as floats; store all categoricals as text.
    for col in CATEGORICAL:
        df[col] = df[col].map(lambda v: None if pd.isna(v) else str(v).removesuffix(".0"))

    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURES_PARQUET, index=False)
    print(f"{len(df):,} rows x {df.shape[1]} columns -> {FEATURES_PARQUET}")


if __name__ == "__main__":
    main()
