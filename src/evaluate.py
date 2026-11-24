"""Metrics for an imbalanced fraud problem.

- PR-AUC (average precision): the main metric. With 3.5% fraud, ROC-AUC looks
  good even for weak models because true negatives are plentiful; PR-AUC only
  rewards finding fraud without flooding the team with false alarms.
- ROC-AUC: reported because it is the Kaggle competition metric.
- Recall / precision at a 1% alert budget: the business view. If analysts can
  review the top 1% riskiest transactions, what share of fraud do they catch,
  and how many of the reviewed cases are really fraud?
- Precision / recall / F1 at a decision threshold: a threshold turns scores into
  yes/no labels. It is always chosen on the VALIDATION month and then applied
  unchanged to the test month.
"""
import json

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from src.config import ALERT_RATE, REPORTS


def evaluate(y_true: np.ndarray, y_score: np.ndarray, alert_rate: float = ALERT_RATE) -> dict:
    threshold = np.quantile(y_score, 1 - alert_rate)
    flagged = y_score >= threshold
    caught = (flagged & (y_true == 1)).sum()
    return {
        "pr_auc": round(float(average_precision_score(y_true, y_score)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4),
        f"recall_at_{alert_rate:.0%}_alerts": round(float(caught / y_true.sum()), 4),
        f"precision_at_{alert_rate:.0%}_alerts": round(float(caught / flagged.sum()), 4),
    }


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Score threshold that maximises F1 (balance of precision and recall)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    # precision/recall have one more element than thresholds (the end point)
    return float(thresholds[np.argmax(f1[:-1])])


def at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    """Precision, recall and F1 when every score >= threshold is labelled fraud."""
    flagged = y_score >= threshold
    tp = float((flagged & (y_true == 1)).sum())
    precision = tp / flagged.sum() if flagged.any() else 0.0
    recall = tp / y_true.sum()
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "alert_rate": round(float(flagged.mean()), 4),
    }


def money_at_alert_budget(y_true: np.ndarray, y_score: np.ndarray, amount: np.ndarray,
                          alert_rate: float = ALERT_RATE) -> dict:
    """The business view in dollars: of all fraudulent money in the period, how much
    sits in the transactions the team reviews (top `alert_rate` of scores)?

    Counting cases treats a $5 card test like a $2,000 purchase; the fraud team cares
    about the money, so this is reported next to case-based recall.
    """
    flagged = y_score >= np.quantile(y_score, 1 - alert_rate)
    fraud = y_true == 1
    fraud_usd = float(amount[fraud].sum())
    caught_usd = float(amount[flagged & fraud].sum())
    return {
        "fraud_usd_total": round(fraud_usd),
        "fraud_usd_caught": round(caught_usd),
        "usd_recall": round(caught_usd / fraud_usd, 4),
        "reviewed_usd_that_is_fraud": round(caught_usd / float(amount[flagged].sum()), 4),
    }


def bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray, groups: np.ndarray,
                 metric=average_precision_score, n_boot: int = 1000, seed: int = 0,
                 level: float = 0.95) -> tuple[float, float]:
    """Confidence interval for a metric by resampling whole DAYS with replacement.

    Transactions on the same day are correlated (same fraud campaign, same
    merchants), so resampling single rows would make the interval too narrow.
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    index_by_group = {g: np.flatnonzero(groups == g) for g in unique}
    stats = []
    for _ in range(n_boot):
        picked = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_by_group[g] for g in picked])
        if y_true[idx].min() == y_true[idx].max():
            continue
        stats.append(metric(y_true[idx], y_score[idx]))
    alpha = (1 - level) / 2
    low, high = np.quantile(stats, [alpha, 1 - alpha])
    return round(float(low), 4), round(float(high), 4)


def save_metrics(name: str, metrics: dict) -> None:
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / f"metrics_{name}.json"
    path.write_text(json.dumps(metrics, indent=2))
    print(f"{name}: {json.dumps(metrics)}")
