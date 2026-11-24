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


def save_metrics(name: str, metrics: dict) -> None:
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / f"metrics_{name}.json"
    path.write_text(json.dumps(metrics, indent=2))
    print(f"{name}: {json.dumps(metrics)}")
