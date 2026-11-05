"""Run everything: python -m latedeliveries.train"""

import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_predict

from latedeliveries.evaluation import (
    LATE_COST,
    VOUCHER_COST,
    choose_threshold,
    classification_report_at,
    compare_models,
    cost_by_threshold,
    cross_validate_pipeline,
    make_cv,
    policy_costs,
)
from latedeliveries.pipeline import (
    CATEGORY_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    build_pipeline,
    load_orders,
    split_orders,
)

ROOT = Path(__file__).resolve().parents[2]

# Chosen after reading reports/cv_results.csv. See the README for the reasoning.
CHOSEN_MODEL = "logistic_regression"
GRID = {
    "model__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
    "model__class_weight": [None, "balanced"],
}


def leakage_check(frame, X_train, y_train) -> dict:
    """Cross-validate again with the excluded customer_rating added back."""
    with_rating = frame.loc[X_train.index, NUMERIC_FEATURES + ["customer_rating"]]
    with_rating = with_rating.join(X_train[CATEGORY_FEATURES])
    honest = cross_validate_pipeline(
        LogisticRegression(max_iter=1000), X_train, y_train
    )
    leaky = cross_validate_pipeline(
        LogisticRegression(max_iter=1000),
        with_rating,
        y_train,
        numeric=NUMERIC_FEATURES + ["customer_rating"],
    )
    return {
        "roc_auc_without_rating": honest["roc_auc_mean"],
        "roc_auc_with_rating": leaky["roc_auc_mean"],
        "difference": leaky["roc_auc_mean"] - honest["roc_auc_mean"],
    }


def main(root: Path = ROOT) -> dict:
    """Recreate the model, the reports and the model info from nothing."""
    models_dir = root / "models"
    reports_dir = root / "reports"
    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    frame = load_orders()
    X_train, X_test, y_train, y_test = split_orders(frame)

    # Step 2: honest comparison, training data only.
    comparison = compare_models(X_train, y_train)
    comparison.round(4).to_csv(reports_dir / "cv_results.csv", index=False)
    untuned = comparison.set_index("model").loc[CHOSEN_MODEL]

    # Step 3: tune the chosen model.
    search = GridSearchCV(
        build_pipeline(LogisticRegression(max_iter=1000)),
        GRID,
        scoring="roc_auc",
        cv=make_cv(),
    )
    search.fit(X_train, y_train)
    best_index = search.best_index_
    tuned_mean = float(search.cv_results_["mean_test_score"][best_index])
    tuned_std = float(search.cv_results_["std_test_score"][best_index])
    gain = tuned_mean - float(untuned["roc_auc_mean"])
    tuning = {
        "scoring": "roc_auc",
        "combinations": len(search.cv_results_["params"]),
        "best_params": dict(search.best_params_),
        "tuned_roc_auc_mean": tuned_mean,
        "tuned_roc_auc_std": tuned_std,
        "untuned_roc_auc_mean": float(untuned["roc_auc_mean"]),
        "untuned_roc_auc_std": float(untuned["roc_auc_std"]),
        "gain": gain,
        "gain_larger_than_fold_spread": bool(gain > float(untuned["roc_auc_std"])),
    }
    best_model = search.best_estimator_

    # Step 4: threshold from out-of-fold probabilities on the training data.
    oof = cross_val_predict(
        best_model, X_train, y_train, cv=make_cv(), method="predict_proba"
    )[:, 1]
    threshold, oof_cost = choose_threshold(y_train, oof)
    cost_by_threshold(y_train, oof).to_csv(
        reports_dir / "threshold_costs.csv", index=False
    )

    # Step 5: the test set is scored once, here.
    final = build_pipeline(LogisticRegression(max_iter=1000))
    final.set_params(**search.best_params_)
    final.fit(X_train, y_train)
    test_proba = final.predict_proba(X_test)[:, 1]
    test_report = classification_report_at(y_test, test_proba, threshold)
    costs = policy_costs(y_test, test_proba, threshold)
    metrics = {
        "threshold": threshold,
        "voucher_cost": VOUCHER_COST,
        "late_cost": LATE_COST,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "late_rate_test": float(np.mean(y_test)),
        "train_out_of_fold_cost_at_threshold": oof_cost,
        "test": test_report,
        "test_cost": costs,
        "saving_vs_flag_nobody": costs["flag_nobody"] - costs["model"],
    }
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    # Step 6: save the pipeline and its description.
    joblib.dump(final, models_dir / "late_orders.joblib")
    info = {
        "model": CHOSEN_MODEL,
        "scikit_learn_version": sklearn.__version__,
        "seed": RANDOM_STATE,
        "threshold": threshold,
        "numeric_features": NUMERIC_FEATURES,
        "category_features": CATEGORY_FEATURES,
        "best_params": tuning["best_params"],
        "tuning": tuning,
        "test_results": {"metrics": test_report, "cost": costs},
    }
    (models_dir / "model_info.json").write_text(json.dumps(info, indent=2) + "\n")

    leak = leakage_check(frame, X_train, y_train)
    (reports_dir / "leakage_check.json").write_text(json.dumps(leak, indent=2) + "\n")
    return metrics


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2))
