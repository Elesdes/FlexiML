import pytest
from fleximl import LinearModel


def test_linear_model_predict():
    model = LinearModel()
    model.slope = 2.0
    model.intercept = 1.0
    predictions = model.predict([2.0])
    assert predictions[0] == pytest.approx(5.0)


def test_linear_model_train():
    model = LinearModel()
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 4.0, 6.0, 8.0, 10.0]
    model.train(x, y, learning_rate=0.01, epochs=1000)
    assert model.slope == pytest.approx(2.0, abs=0.1)
    assert model.intercept == pytest.approx(0.0, abs=0.1)


def test_linear_model_train_default_params():
    model = LinearModel()
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 4.0, 6.0, 8.0, 10.0]
    model.train(x, y)  # Using default learning_rate and epochs
    assert model.slope == pytest.approx(2.0, abs=0.1)
    assert model.intercept == pytest.approx(0.0, abs=0.1)


def test_linear_model_mse():
    model = LinearModel()
    model.slope = 2.0
    model.intercept = 0.0
    x = [1.0, 2.0, 3.0]
    y = [2.0, 4.0, 6.0]
    mse = model.mse(x, y)
    assert mse == pytest.approx(0.0, abs=1e-10)


def test_linear_model_predict_multiple():
    model = LinearModel()
    model.slope = 2.0
    model.intercept = 1.0
    predictions = model.predict([1.0, 2.0, 3.0])
    assert predictions == pytest.approx([3.0, 5.0, 7.0])


def test_linear_model_train_input_validation():
    model = LinearModel()
    x = [1.0, 2.0, 3.0]
    y = [2.0, 4.0]  # Mismatched length
    with pytest.raises(ValueError):
        model.train(x, y)


def test_linear_model_mse_input_validation():
    model = LinearModel()
    x = [1.0, 2.0, 3.0]
    y = [2.0, 4.0]  # Mismatched length
    with pytest.raises(ValueError):
        model.mse(x, y)
