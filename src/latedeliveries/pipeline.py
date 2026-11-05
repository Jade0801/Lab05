"""Feature lists, the train/test split and the one pipeline used everywhere."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from latedeliveries.data import make_orders

RANDOM_STATE = 42
TEST_SIZE = 0.25
TARGET = "late"

# Every column here is known at the moment the order is placed.
# prep_minutes is the kitchen's estimate recorded when the order is accepted.
NUMERIC_FEATURES = [
    "distance_km",
    "items",
    "hour",
    "prep_minutes",
    "courier_months",
]
CATEGORY_FEATURES = ["weather", "day_type", "vehicle"]

# Columns deliberately left out, and why.
EXCLUDED_COLUMNS = {
    "order_id": "A label only. Its number carries no information about lateness.",
    "customer_rating": (
        "Given by the customer after the delivery, so it is not known when the "
        "order is placed. It is generated from the outcome (a leak)."
    ),
    "late": "The target.",
}


def load_orders() -> pd.DataFrame:
    """Return the provided order table."""
    return make_orders()


def split_orders(frame: pd.DataFrame):
    """Return X_train, X_test, y_train, y_test: stratified 75/25, seeded."""
    features = NUMERIC_FEATURES + CATEGORY_FEATURES
    return train_test_split(
        frame[features],
        frame[TARGET],
        test_size=TEST_SIZE,
        stratify=frame[TARGET],
        random_state=RANDOM_STATE,
    )


def build_pipeline(model, numeric=None, categorical=None) -> Pipeline:
    """Wrap any scikit-learn model in the shared preprocessing pipeline."""
    numeric = NUMERIC_FEATURES if numeric is None else numeric
    categorical = CATEGORY_FEATURES if categorical is None else categorical
    numeric_steps = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("numeric", numeric_steps, list(numeric)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(categorical),
            ),
        ]
    )
    return Pipeline([("prep", preprocess), ("model", model)])
