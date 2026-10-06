"""Paths, feature lists and the time split, shared by every script."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

FEATURES_PARQUET = PROCESSED / "features.parquet"
TARGET = "is_fraud"

# Time-based split on `day` (days since the dataset's reference point, 1..183).
# Train on the past, tune on the following month, report on the last month.
# A random split would leak future behaviour into training and overstate quality.
TRAIN_END_DAY = 122   # train:  day < 122   (~4 months)
VALID_END_DAY = 152   # valid:  122 <= day < 152,  test: day >= 152

# Continuous features, passed to the models as numbers.
NUMERIC = [
    "amt", "log_amt", "amt_cents", "hour", "dist1",
    "c1", "c2", "c5", "c6", "c9", "c11", "c13", "c14",
    "d1", "d2", "d3", "d4", "d10", "d15",
    "has_identity", "email_mismatch",
    "uid_prev_tx", "uid_prev_mean_amt", "uid_prev_std_amt", "amt_to_uid_mean",
    "uid_tx_last_1h", "uid_tx_last_24h", "card1_tx_last_24h",
    "secs_since_prev_tx", "uid_same_amt_prev",
]

# Categorical features, encoded as integer codes learned on the train period.
CATEGORICAL = [
    "product_cd", "card4", "card6", "p_emaildomain", "r_emaildomain",
    "m4", "m5", "m6", "device_type",
    # Card / address identifiers: numbers in the file, but their values are
    # labels (bank ID, region code), not quantities.
    "card1", "card2", "card3", "card5", "addr1", "addr2",
]

# Categories seen fewer times than this in train are folded into "rare".
MIN_CATEGORY_COUNT = 30

# Alert budget used for the business metric: the fraud team can manually
# review the top 1% riskiest transactions.
ALERT_RATE = 0.01

RANDOM_SEED = 42
