import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from latedeliveries.pipeline import (
    CATEGORY_FEATURES,
    NUMERIC_FEATURES,
    build_pipeline,
    split_orders,
)


@pytest.mark.parametrize("column", ["order_id", "customer_rating", "late"])
def test_feature_list_excludes_columns(column):
    assert column not in NUMERIC_FEATURES + CATEGORY_FEATURES


def test_split_sizes_are_75_25(orders, split):
    X_train, X_test, _, _ = split
    assert len(X_train) == 2250
    assert len(X_test) == 750


def test_split_is_stratified(orders, split):
    _, _, y_train, y_test = split
    overall = orders["late"].mean()
    assert y_train.mean() == pytest.approx(overall, abs=0.002)
    assert y_test.mean() == pytest.approx(overall, abs=0.002)


def test_split_is_repeatable(orders, split):
    again = split_orders(orders)
    assert list(again[0].index) == list(split[0].index)
    assert list(again[1].index) == list(split[1].index)


@pytest.fixture(scope="module")
def fitted(split):
    X_train, _, y_train, _ = split
    return build_pipeline(LogisticRegression(max_iter=1000)).fit(X_train, y_train)


def test_probabilities_are_valid(fitted, split):
    proba = fitted.predict_proba(split[1])
    assert proba.shape == (len(split[1]), 2)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert np.allclose(proba.sum(axis=1), 1)


def test_pipeline_handles_missing_value_and_unseen_category(fitted):
    order = pd.DataFrame(
        [
            {
                "distance_km": 3.0,
                "items": 2,
                "hour": 19,
                "prep_minutes": np.nan,
                "courier_months": 12,
                "weather": "hail",
                "day_type": "weekend",
                "vehicle": "bike",
            }
        ]
    )
    proba = fitted.predict_proba(order)[0, 1]
    assert 0 <= proba <= 1
