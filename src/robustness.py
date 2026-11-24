"""How much can the headline numbers be trusted? And what do they mean in dollars?

1. Seed stability: LightGBM is retrained with 5 random seeds (row/column subsampling
   changes). If PR-AUC moves by more than the gap between two models, that gap means nothing.
2. Day-level bootstrap: a 95% interval for each model's test PR-AUC, resampling whole days.
3. Dollars: share of fraudulent money that sits in the top-1% alerts, per model.
4. Ensemble: rank-average of LightGBM, XGBoost and the MLP. Whether to use it is decided on
   the VALIDATION month; the test month only reports the result.

Needs the scores saved by train_gbm.py, train_nn.py and train_iforest.py.
Run:  python -m src.robustness
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

from src.config import MODELS, REPORTS
from src.evaluate import bootstrap_ci, evaluate, money_at_alert_budget
from src.preprocess import Preprocessor, as_category, load_splits, target
from src.train_gbm import fit_lightgbm

SEEDS = [0, 1, 2, 3, 4]
LABELS = {
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "pytorch": "PyTorch MLP",
    "logreg": "Logistic regression (baseline)",
    "iforest": "Isolation Forest (unsupervised)",
    "ensemble": "Ensemble (rank average: LightGBM + XGBoost + MLP)",
}
ENSEMBLE_PARTS = ["lightgbm", "xgboost", "pytorch"]


def rank_average(score_list: list[np.ndarray]) -> np.ndarray:
    """Average of ranks scaled to 0..1: the models' scores live on different scales."""
    return np.mean([rankdata(s) / len(s) for s in score_list], axis=0)


def main() -> None:
    train, valid, test = load_splits()
    y_tr, y_va, y_te = target(train), target(valid), target(test)

    # 1. Seed stability of the best model
    pre = Preprocessor().fit(train)
    x_tr, x_va, x_te = (as_category(pre.transform_tree(d), pre) for d in (train, valid, test))
    seed_pr_auc = []
    for seed in SEEDS:
        model = fit_lightgbm(x_tr, y_tr, x_va, y_va, seed=seed, log_every=0)
        seed_pr_auc.append(average_precision_score(y_te, model.predict_proba(x_te)[:, 1]))
        print(f"seed {seed}: test PR-AUC {seed_pr_auc[-1]:.4f}")

    # 2-4. Saved test scores of every model, plus the ensemble
    test_scores = {k: np.load(MODELS / f"test_scores_{k}.npy") for k in LABELS if k != "ensemble"}
    valid_scores = {k: np.load(MODELS / f"valid_scores_{k}.npy") for k in ENSEMBLE_PARTS}
    test_scores["ensemble"] = rank_average([test_scores[k] for k in ENSEMBLE_PARTS])
    valid_ensemble = rank_average([valid_scores[k] for k in ENSEMBLE_PARTS])
    valid_pr_auc = {"ensemble": average_precision_score(y_va, valid_ensemble),
                    "lightgbm": average_precision_score(y_va, valid_scores["lightgbm"])}
    np.save(MODELS / "test_scores_ensemble.npy", test_scores["ensemble"])

    days, amount = test["day"].to_numpy(), test["amt"].to_numpy()
    rows = []
    for key, label in LABELS.items():
        s = test_scores[key]
        low, high = bootstrap_ci(y_te, s, days)
        rows.append({"model": label, **evaluate(y_te, s),
                     "pr_auc_95ci": f"{low:.3f}–{high:.3f}",
                     **money_at_alert_budget(y_te, s, amount)})
    table = pd.DataFrame(rows).set_index("model").sort_values("pr_auc", ascending=False)

    summary = {
        "lightgbm_seed_pr_auc": [round(v, 4) for v in seed_pr_auc],
        "lightgbm_seed_pr_auc_mean": round(float(np.mean(seed_pr_auc)), 4),
        "lightgbm_seed_pr_auc_std": round(float(np.std(seed_pr_auc, ddof=1)), 4),
        "valid_pr_auc_lightgbm": round(valid_pr_auc["lightgbm"], 4),
        "valid_pr_auc_ensemble": round(valid_pr_auc["ensemble"], 4),
        "ensemble_chosen_on_validation": bool(valid_pr_auc["ensemble"] > valid_pr_auc["lightgbm"]),
        "test_fraud_usd_total": int(table["fraud_usd_total"].iloc[0]),
    }
    # 5. Which transactions should the team review? Ranking by probability catches many
    #    small frauds; ranking by expected loss (probability x amount) targets the money.
    #    Compared on validation and test with the same 1% review budget.
    ranking_rows = []
    for split, df, scores in [("validation", valid, np.load(MODELS / "valid_scores_lightgbm.npy")),
                              ("test", test, test_scores["lightgbm"])]:
        y_split, amt = target(df), df["amt"].to_numpy()
        for rule, rank_by in [("fraud probability", scores), ("expected loss (probability x amount)", scores * amt)]:
            cases = evaluate(y_split, rank_by)
            money = money_at_alert_budget(y_split, rank_by, amt)
            ranking_rows.append({"month": split, "review the top 1% by": rule,
                                 "share of fraud cases caught": cases["recall_at_1%_alerts"],
                                 "reviewed cases that are fraud": cases["precision_at_1%_alerts"],
                                 "share of fraud dollars caught": money["usd_recall"],
                                 "fraud dollars caught": money["fraud_usd_caught"],
                                 "fraud dollars in month": money["fraud_usd_total"]})
    ranking = pd.DataFrame(ranking_rows).set_index(["month", "review the top 1% by"])
    (REPORTS / "alert_ranking.md").write_text(ranking.to_markdown() + "\n")
    print(ranking.to_string())

    (REPORTS / "robustness.json").write_text(json.dumps(summary, indent=2))
    (REPORTS / "robustness.md").write_text(
        table.drop(columns="fraud_usd_total").to_markdown() + "\n\n"
        + f"LightGBM test PR-AUC over {len(SEEDS)} seeds: {summary['lightgbm_seed_pr_auc_mean']:.4f} "
          f"± {summary['lightgbm_seed_pr_auc_std']:.4f} (mean ± std)\n"
    )
    print(table.to_string())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
