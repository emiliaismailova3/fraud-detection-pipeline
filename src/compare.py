"""Compare all models on the test period: table + precision-recall curve.

Run:  python -m src.compare
"""
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from src.config import FIGURES, MODELS, REPORTS
from src.preprocess import load_splits, target

matplotlib.use("Agg")  # render to files, no window (we are in a terminal)

MODELS_TO_COMPARE = {
    "logreg": "Logistic regression (baseline)",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "pytorch": "PyTorch MLP",
}


def main() -> None:
    _, _, test = load_splits()
    y = target(test)

    rows = []
    fig, ax = plt.subplots(figsize=(7, 5))
    for key, label in MODELS_TO_COMPARE.items():
        metrics = json.loads((REPORTS / f"metrics_{key}.json").read_text())
        rows.append({"model": label, **metrics})
        precision, recall, _ = precision_recall_curve(y, np.load(MODELS / f"test_scores_{key}.npy"))
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
