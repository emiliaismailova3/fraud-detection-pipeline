"""Classical ML: logistic regression baseline, XGBoost and LightGBM.

Every model is trained on the train period, early-stopped on the validation
period, and reported on the untouched test period (the last month).

Run:  python -m src.train_gbm
"""
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

from src.config import ALERT_RATE, MODELS, RANDOM_SEED
from src.evaluate import evaluate, save_metrics
from src.preprocess import Preprocessor, as_category, load_splits, target


def main() -> None:
    train, valid, test = load_splits()
    y_tr, y_va, y_te = target(train), target(valid), target(test)
    print(f"train {len(train):,} | valid {len(valid):,} | test {len(test):,} rows")

    pre = Preprocessor().fit(train)
    MODELS.mkdir(exist_ok=True)
    joblib.dump(pre, MODELS / "preprocessor.joblib")

    # 1. Baseline: logistic regression on scaled numeric features only.
    #    Any serious model has to beat this to justify its complexity.
    xn_tr, _ = pre.transform_dense(train)
    xn_te, _ = pre.transform_dense(test)
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(xn_tr, y_tr)
    save_metrics("logreg", evaluate(y_te, logreg.predict_proba(xn_te)[:, 1]))

    x_tr = as_category(pre.transform_tree(train), pre)
    x_va = as_category(pre.transform_tree(valid), pre)
    x_te = as_category(pre.transform_tree(test), pre)

    # 2. XGBoost. `hist` bins features into histograms: fast on 400k rows.
    xgb_model = xgb.XGBClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.7,
        tree_method="hist",
        enable_categorical=True,
        max_cat_to_onehot=1,
        eval_metric="aucpr",
        early_stopping_rounds=100,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    xgb_model.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], verbose=200)
    xgb_model.save_model(MODELS / "xgboost.json")
    save_metrics("xgboost", evaluate(y_te, xgb_model.predict_proba(x_te)[:, 1]))

    # 3. LightGBM. Grows trees leaf-wise (num_leaves) instead of level-wise.
    lgb_model = lgb.LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=127,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        cat_smooth=20,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        x_tr, y_tr,
        eval_set=[(x_va, y_va)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )
    lgb_model.booster_.save_model(MODELS / "lightgbm.txt")
    save_metrics("lightgbm", evaluate(y_te, lgb_model.predict_proba(x_te)[:, 1]))

    # Which features does the best tree model rely on?
    importance = pd.Series(
        lgb_model.booster_.feature_importance("gain"), index=x_tr.columns
    ).sort_values(ascending=False)
    importance.to_csv(MODELS.parent / "reports" / "lightgbm_feature_importance.csv")
    print(importance.head(15).round(0))

    # Alert threshold for the API, chosen on VALIDATION scores (never on test):
    # the score above which the top 1% of transactions fall.
    valid_scores = lgb_model.predict_proba(x_va)[:, 1]
    api_config = {"model": "lightgbm", "alert_threshold": float(np.quantile(valid_scores, 1 - ALERT_RATE))}
    (MODELS / "api_config.json").write_text(json.dumps(api_config, indent=2))

    np.save(MODELS / "test_scores_lightgbm.npy", lgb_model.predict_proba(x_te)[:, 1])
    np.save(MODELS / "test_scores_xgboost.npy", xgb_model.predict_proba(x_te)[:, 1])
    np.save(MODELS / "test_scores_logreg.npy", logreg.predict_proba(xn_te)[:, 1])


if __name__ == "__main__":
    main()
