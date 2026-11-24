"""Unsupervised baseline: Isolation Forest.

Isolation Forest never sees the labels. It builds random trees that split on a
random feature at a random value; unusual points (far from the bulk of the
data) get isolated in few splits, so a SHORT average path length = anomaly.
The anomaly score is the negated sklearn `score_samples` (higher = more anomalous).

Why include it: in production, labels arrive late (chargebacks take weeks) or
not at all for new fraud patterns. This shows how far "unusual" gets us without
labels, and how much the supervised models add on top.

`contamination` only sets the cut-off for `predict()` (the share of points
labelled anomalous); it does not change the ranking, so PR-AUC/ROC-AUC do not
depend on it. We set it to the fraud rate in train.

Run:  python -m src.train_iforest   (after src.train_gbm, which fits the preprocessor)
"""
import json

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from src.config import MODELS, RANDOM_SEED, REPORTS
from src.evaluate import at_threshold, evaluate, save_metrics
from src.preprocess import Preprocessor, load_splits, target


def main() -> None:
    train, valid, test = load_splits()
    y_tr, y_te = target(train), target(test)

    # Same scaled numeric features + missing-value flags as the neural net.
    # Isolation Forest needs numbers; the categorical codes are labels, not
    # quantities, so they are left out.
    pre: Preprocessor = joblib.load(MODELS / "preprocessor.joblib")
    x_tr, _ = pre.transform_dense(train)
    x_va, _ = pre.transform_dense(valid)
    x_te, _ = pre.transform_dense(test)

    fraud_rate = float(y_tr.mean())
    model = IsolationForest(
        n_estimators=300,
        max_samples=256,          # small subsamples isolate anomalies best (original paper)
        contamination=fraud_rate,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(x_tr)  # labels are NOT used

    test_scores = -model.score_samples(x_te)
    np.save(MODELS / "test_scores_iforest.npy", test_scores)
    np.save(MODELS / "valid_scores_iforest.npy", -model.score_samples(x_va))

    metrics = evaluate(y_te, test_scores)
    # Recall/precision at the model's own cut-off (contamination = train fraud rate)
    own_cutoff = at_threshold(y_te, test_scores, -model.offset_)
    metrics.update({f"{k}_at_contamination": v for k, v in own_cutoff.items()})
    metrics["contamination"] = round(fraud_rate, 4)
    save_metrics("iforest", metrics)


if __name__ == "__main__":
    main()
