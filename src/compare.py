"""Compare all models on the test period: table + precision-recall curve.

For every model the decision threshold is chosen on the VALIDATION month
(the one that maximises F1) and then applied unchanged to the test month, so
precision / recall / F1 are honest out-of-sample numbers. The API's own
threshold (top 1% of validation scores, see train_gbm.py) is reported too.

Run:  python -m src.compare
"""
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from src.config import FIGURES, MODELS, REPORTS
from src.evaluate import at_threshold, best_f1_threshold
from src.preprocess import load_splits, target

matplotlib.use("Agg")  # render to files, no window (we are in a terminal)

MODELS_TO_COMPARE = {
    "iforest": "Isolation Forest (unsupervised)",
    "logreg": "Logistic regression (baseline)",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "pytorch": "PyTorch MLP",
}


def main() -> None:
    _, valid, test = load_splits()
    y, y_valid = target(test), target(valid)

    rows = []
    fig, ax = plt.subplots(figsize=(7, 5))
    for key, label in MODELS_TO_COMPARE.items():
        metrics = json.loads((REPORTS / f"metrics_{key}.json").read_text())
        scores = np.load(MODELS / f"test_scores_{key}.npy")
        assert len(scores) == len(y), f"{key}: scores must cover the same test rows as every other model"
        threshold = best_f1_threshold(y_valid, np.load(MODELS / f"valid_scores_{key}.npy"))
        rows.append({
            "model": label,
            "pr_auc": metrics["pr_auc"],
            "roc_auc": metrics["roc_auc"],
            **{k: v for k, v in at_threshold(y, scores, threshold).items() if k != "alert_rate"},
            "threshold (max F1 on valid)": round(threshold, 4),
            "recall_at_1%_alerts": metrics["recall_at_1%_alerts"],
            "precision_at_1%_alerts": metrics["precision_at_1%_alerts"],
        })
        precision, recall, _ = precision_recall_curve(y, scores)
        ax.plot(recall, precision, label=f"{label} (PR-AUC {metrics['pr_auc']:.3f})")

    ax.axhline(y.mean(), color="grey", linestyle="--", label=f"Random ({y.mean():.3f})")
    ax.set(xlabel="Recall (share of fraud caught)", ylabel="Precision (share of alerts that are fraud)",
           title="Precision-recall on the test month", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "pr_curves.png", dpi=150)

    table = pd.DataFrame(rows).set_index("model")
    (REPORTS / "model_comparison.md").write_text(table.to_markdown() + "\n")
    print(table.to_string())

    # The threshold actually used by the API (LightGBM, top 1% of validation scores)
    api_config = json.loads((MODELS / "api_config.json").read_text())
    api_scores = np.load(MODELS / f"test_scores_{api_config['model']}.npy")
    api_result = {"model": api_config["model"], "threshold": round(api_config["alert_threshold"], 4),
                  "rule": "top 1% of validation scores (alert budget)",
                  **at_threshold(y, api_scores, api_config["alert_threshold"])}
    (REPORTS / "api_threshold_test.json").write_text(json.dumps(api_result, indent=2))
    print(f"\nAPI threshold on the test month: {json.dumps(api_result)}")

    # Feature importance chart for the README
    importance = pd.read_csv(REPORTS / "lightgbm_feature_importance.csv", index_col=0).iloc[:, 0]
    top = importance.head(15)[::-1] / importance.sum()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.index, top.values, color="#3b6ea5")
    ax.set(xlabel="Share of total gain", title="LightGBM: top 15 features")
    fig.tight_layout()
    fig.savefig(FIGURES / "feature_importance.png", dpi=150)


if __name__ == "__main__":
    main()
