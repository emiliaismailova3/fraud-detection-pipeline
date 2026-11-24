"""Unit tests for the threshold metrics (no data or trained models needed)."""
import numpy as np
import pytest

from src.evaluate import at_threshold, best_f1_threshold


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


def test_best_f1_threshold_separates_perfectly_ranked_scores():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    t = best_f1_threshold(y, s)
    assert at_threshold(y, s, t)["f1"] == 1.0
