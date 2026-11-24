"""Feature-set ablation: how much do the raw columns we do NOT use cost us?

The served model uses 45 features: the ones a real-time system can supply cheaply
(amount, card, address, email, a few counters) plus our SQL behaviour features. The
Kaggle files have ~430 columns, most of them Vesta's anonymised engineered features
(V1..V339). This script retrains LightGBM with growing feature sets on the same
time split and reports the test metrics, so the trade-off is measured, not guessed.

  base       the 45 features of the served model
  + C/D/M    remaining counters (C), time deltas (D) and match flags (M)
  + V        Vesta's 339 anonymised features
  + identity device / browser / network columns (id_01..id_38, device_info)

The extra columns are exported once from PostgreSQL to data/processed/raw_extra.parquet.
Run:  python -m src.feature_ablation
"""
import io

import pandas as pd

from src.config import CATEGORICAL, NUMERIC, PROCESSED, RANDOM_SEED, REPORTS, TRAIN_END_DAY, VALID_END_DAY
from src.db import get_engine
from src.evaluate import evaluate
from src.preprocess import Preprocessor, as_category, load_splits, target
from src.train_gbm import fit_lightgbm

EXTRA_PARQUET = PROCESSED / "raw_extra.parquet"


def export_extra_columns() -> pd.DataFrame:
    """Every raw column not already in the feature table, via COPY (much faster than read_sql)."""
    engine = get_engine()
    with engine.connect() as conn:
        cols = pd.read_sql(
            "SELECT table_name, column_name FROM information_schema.columns "
            "WHERE table_schema = 'raw' ORDER BY table_name, ordinal_position", conn)
    used = set(NUMERIC + CATEGORICAL)
    tx = [c for c in cols.query("table_name == 'train_transaction'").column_name
          if c[0] in "cdmv" and c[1:].isdigit() and c not in used] + ["dist2"]
    ident = [c for c in cols.query("table_name == 'train_identity'").column_name
             if c.startswith("id_") or c == "device_info"]
    query = (f"SELECT t.transaction_id, {', '.join('t.' + c for c in tx)}, {', '.join('i.' + c for c in ident)} "
             "FROM raw.train_transaction t LEFT JOIN raw.train_identity i USING (transaction_id)")
    buffer = io.StringIO()
    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cur, cur.copy(f"COPY ({query}) TO STDOUT WITH (FORMAT csv, HEADER true)") as copy:
            for chunk in copy:
                buffer.write(bytes(chunk).decode())
    finally:
        raw_conn.close()
    buffer.seek(0)
    df = pd.read_csv(buffer, low_memory=False)
    df.to_parquet(EXTRA_PARQUET, index=False)
    print(f"exported {df.shape[1] - 1} extra columns -> {EXTRA_PARQUET}")
    return df


def extra_frame(d: pd.DataFrame, cols: list[str], categories: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=d.index)
    for c in cols:
        if c in categories:
            out[c] = pd.Categorical(d[c].astype(object), categories=categories[c])
        else:
            out[c] = d[c].astype("float32")
    return out


def main() -> None:
    extra = pd.read_parquet(EXTRA_PARQUET) if EXTRA_PARQUET.exists() else export_extra_columns()
    train, valid, test = (d.merge(extra, on="transaction_id", how="left").set_index(d.index)
                          for d in load_splits())
    pre = Preprocessor().fit(train)

    extra_cols = [c for c in extra.columns if c != "transaction_id"]
    text_cols = [c for c in extra_cols if not pd.api.types.is_numeric_dtype(train[c])]
    categories = {c: pd.Index(train[c].dropna().astype(object).unique()) for c in text_cols}
    cdm = [c for c in extra_cols if c[0] in "cdm" and c[1:].isdigit()]
    v = [c for c in extra_cols if c[0] == "v" and c[1:].isdigit()]
    ident = [c for c in extra_cols if c not in cdm + v]
    feature_sets = {
        "base (served model)": [],
        "+ C/D/M": cdm,
        "+ C/D/M + V": cdm + v,
        "+ C/D/M + V + identity": cdm + v + ident,
    }

    rows = []
    for name, cols in feature_sets.items():
        x = [pd.concat([as_category(pre.transform_tree(d), pre), extra_frame(d, cols, categories)], axis=1)
             for d in (train, valid, test)]
        model = fit_lightgbm(x[0], target(train), x[1], target(valid), seed=RANDOM_SEED, log_every=0)
        metrics = evaluate(target(test), model.predict_proba(x[2])[:, 1])
        rows.append({"feature set": name, "features": x[0].shape[1], **metrics})
        print(rows[-1])

    table = pd.DataFrame(rows).set_index("feature set")
    (REPORTS / "feature_ablation.md").write_text(
        f"LightGBM, same time split (train < day {TRAIN_END_DAY}, valid < day {VALID_END_DAY}, test after), "
        f"seed {RANDOM_SEED}.\n\n" + table.to_markdown() + "\n")


if __name__ == "__main__":
    main()
