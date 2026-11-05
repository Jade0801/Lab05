"""Metrics, the cost rule, the threshold choice and model comparison."""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from latedeliveries.pipeline import RANDOM_STATE, build_pipeline

VOUCHER_COST = 4
LATE_COST = 15
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.05), 2)

SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}


def make_cv() -> StratifiedKFold:
    """The one cross-validation splitter used for every comparison."""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def order_cost(y_true, flagged) -> int:
    """$4 for every order flagged, plus $15 for every late order not flagged."""
    y_true = np.asarray(y_true).astype(bool)
    flagged = np.asarray(flagged).astype(bool)
    vouchers = int(flagged.sum())
    missed = int((y_true & ~flagged).sum())
    return vouchers * VOUCHER_COST + missed * LATE_COST


def cost_by_threshold(y_true, proba, thresholds=THRESHOLDS) -> pd.DataFrame:
    """Total cost at each threshold. An order is flagged when proba >= threshold."""
    proba = np.asarray(proba)
    rows = [
        {"threshold": float(t), "cost": order_cost(y_true, proba >= t)}
        for t in thresholds
    ]
    return pd.DataFrame(rows)


def choose_threshold(y_true, proba, thresholds=THRESHOLDS):
    """Return (threshold, cost) with the lowest cost. Ties pick the lower one."""
    table = cost_by_threshold(y_true, proba, thresholds)
    best = table.loc[table["cost"].idxmin()]
    return float(best["threshold"]), int(best["cost"])


def policy_costs(y_true, proba, threshold) -> dict:
    """Cost of flagging nobody, flagging everybody and using the model."""
    y_true = np.asarray(y_true)
    return {
        "flag_nobody": order_cost(y_true, np.zeros(len(y_true), dtype=bool)),
        "flag_everybody": order_cost(y_true, np.ones(len(y_true), dtype=bool)),
        "model": order_cost(y_true, np.asarray(proba) >= threshold),
    }


def classification_report_at(y_true, proba, threshold) -> dict:
    """Confusion matrix and metrics at the chosen threshold."""
    flagged = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, flagged, labels=[0, 1]).ravel()
    return {
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "precision": float(precision_score(y_true, flagged, zero_division=0)),
        "recall": float(recall_score(y_true, flagged, zero_division=0)),
        "f1": float(f1_score(y_true, flagged, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
    }


def candidate_models() -> dict:
    """The baseline and the real candidates, by name."""
    return {
        "baseline_dummy": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(max_iter=1000),
        "logistic_balanced": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
    }


def cross_validate_pipeline(model, X, y, numeric=None, categorical=None) -> dict:
    """Mean and standard deviation of each metric over five stratified folds."""
    pipe = build_pipeline(model, numeric, categorical)
    scores = cross_validate(pipe, X, y, cv=make_cv(), scoring=SCORING)
    summary = {}
    for name in SCORING:
        values = scores[f"test_{name}"]
        summary[f"{name}_mean"] = float(values.mean())
        summary[f"{name}_std"] = float(values.std())
    return summary


def compare_models(X, y) -> pd.DataFrame:
    """Cross-validate every candidate on the training data only."""
    rows = []
    for name, model in candidate_models().items():
        rows.append({"model": name, **cross_validate_pipeline(model, X, y)})
    return pd.DataFrame(rows)
