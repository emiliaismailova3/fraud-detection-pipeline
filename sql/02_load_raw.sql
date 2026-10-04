-- Load the Kaggle CSVs into the raw tables.
-- Runs inside the Postgres container, where data/raw is mounted at /data/raw.
-- COPY is Postgres' bulk loader: much faster than row-by-row INSERTs.

TRUNCATE raw.train_transaction, raw.train_identity;

COPY raw.train_transaction FROM '/data/raw/train_transaction.csv'
    WITH (FORMAT csv, HEADER true);

COPY raw.train_identity FROM '/data/raw/train_identity.csv'
    WITH (FORMAT csv, HEADER true);

-- Refresh planner statistics after a bulk load
ANALYZE raw.train_transaction;
ANALYZE raw.train_identity;
