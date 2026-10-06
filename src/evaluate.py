"""Metrics for an imbalanced fraud problem.

- PR-AUC (average precision): the main metric. With 3.5% fraud, ROC-AUC looks
  good even for weak models because true negatives are plentiful; PR-AUC only
  rewards finding fraud without flooding the team with false alarms.
- ROC-AUC: reported because it is the Kaggle competition metric.
- Recall / precision at a 1% alert budget: the business view. If analysts can
  review the top 1% riskiest transactions, what share of fraud do they catch,
  and how many of the reviewed cases are really fraud?
"""
import json

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

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


def save_metrics(name: str, metrics: dict) -> None:
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / f"metrics_{name}.json"
    path.write_text(json.dumps(metrics, indent=2))
    print(f"{name}: {json.dumps(metrics)}")
