"""Unit tests for the threshold metrics (no data or trained models needed)."""
import numpy as np
import pytest

from src.evaluate import at_threshold, best_f1_threshold, bootstrap_ci, money_at_alert_budget


def test_at_threshold_counts():
    y = np.array([1, 1, 0, 0, 0, 1])
    s = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.3])
    m = at_threshold(y, s, 0.5)          # flags the first three: 2 fraud, 1 legit
    assert m["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["f1"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["alert_rate"] == 0.5


def test_nothing_flagged_gives_zero_not_error():
    m = at_threshold(np.array([1, 0]), np.array([0.2, 0.1]), 0.99)
    assert m == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "alert_rate": 0.0}


def test_money_at_alert_budget_weights_by_amount():
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.1, 0.5, 0.2])
    amt = np.array([300.0, 100.0, 50.0, 10.0])
    m = money_at_alert_budget(y, s, amt, alert_rate=0.25)   # reviews only the top score
    assert m["fraud_usd_total"] == 400
    assert m["fraud_usd_caught"] == 300
    assert m["usd_recall"] == 0.75
    assert m["reviewed_usd_that_is_fraud"] == 1.0


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    s = y * 0.5 + rng.random(2000)
    days = np.repeat(np.arange(40), 50)
    low, high = bootstrap_ci(y, s, days, n_boot=200)
    from sklearn.metrics import average_precision_score
    assert low < average_precision_score(y, s) < high


def test_best_f1_threshold_separates_perfectly_ranked_scores():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    t = best_f1_threshold(y, s)
    assert at_threshold(y, s, t)["f1"] == 1.0
