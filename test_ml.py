import pytest
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics


def test_one():
    """
    Test that train_model returns a fitted model with a predict method
    """
    # Small toy dataset just to confirm the function trains a real model
    X_train = [[0, 1], [1, 0], [1, 1], [0, 0]]
    y_train = [1, 0, 1, 0]
    model = train_model(X_train, y_train)
    # Model should exist and be able to make predictions
    assert model is not None
    assert hasattr(model, "predict")


def test_two():
    """
    Test that inference returns predictions of the expected length
    """
    # Reuse the same toy dataset to train a quick model
    X_train = [[0, 1], [1, 0], [1, 1], [0, 0]]
    y_train = [1, 0, 1, 0]
    model = train_model(X_train, y_train)
    # Predictions should have one value per input row
    preds = inference(model, X_train)
    assert len(preds) == len(y_train)
    


def test_three():
    """
    Test that compute_model_metrics returns expected values for a known input.
    """
    # Predictions perfectly match the true labels here
    y = [1, 1, 0, 0]
    preds = [1, 1, 0, 0]
    precision, recall, fbeta = compute_model_metrics(y, preds)
    # A perfect match should give a perfect score on all three metrics
    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0
