"""Unit tests for saving and loading the trained model pipeline."""

import joblib
from sklearn.linear_model import LogisticRegression

from latedeliveries.pipeline import build_pipeline


def test_save_and_load_model(tmp_path, sample_orders_df):
    """Test that saving a fitted pipeline via joblib and
    loading it back yields identical predictions.
    """
    features = [
        "distance_km",
        "items",
        "hour",
        "prep_minutes",
        "courier_months",
        "weather",
        "day_type",
        "vehicle",
    ]
    X = sample_orders_df[features]
    y = sample_orders_df["late"]

    pipe = build_pipeline(LogisticRegression())
    pipe.fit(X, y)

    # Save model into pytest temporary directory (tmp_path)
    model_file = tmp_path / "late_orders_test.joblib"
    joblib.dump(pipe, model_file)

    # Load the model back
    loaded_pipe = joblib.load(model_file)

    # Compare predictions to ensure 100% match
    orig_preds = pipe.predict(X)
    loaded_preds = loaded_pipe.predict(X)

    assert (orig_preds == loaded_preds).all()
