"""
Тесты для класса LinearRegression.

Проверяются три режима обучения (sgd, gd, norm_eq), корректность предсказаний,
метрика R², поведение при неверных аргументах и ранняя остановка.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from numl.linear import LinearRegression


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

@pytest.fixture
def linear_data():
    """Синтетический датасет: y = 3·x₁ + 2·x₂ + 1 с малым шумом."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 2))
    y = 3 * X[:, 0] + 2 * X[:, 1] + 1 + rng.normal(0, 0.1, 200)
    return X, y


@pytest.fixture
def simple_1d():
    """Идеальная линейная зависимость y = 2x (без шума)."""
    X = np.linspace(-1, 1, 100).reshape(-1, 1)
    y = 2 * X.ravel()
    return X, y


# ---------------------------------------------------------------------------
# Инициализация и валидация аргументов
# ---------------------------------------------------------------------------

class TestLinearRegressionInit:
    def test_invalid_method_raises(self):
        """Передача неизвестного метода должна вызывать ValueError."""
        with pytest.raises(ValueError, match="Неизвестный метод"):
            LinearRegression(method="adam")

    def test_valid_methods_accepted(self):
        """Все допустимые методы принимаются без исключений."""
        for m in ("sgd", "gd", "norm_eq"):
            model = LinearRegression(method=m)
            assert model.method == m

    def test_default_params(self):
        """Параметры по умолчанию соответствуют документации."""
        model = LinearRegression()
        assert model.method == "sgd"
        assert model.lr == 0.01
        assert model.epochs == 1000
        assert model.batch_size == 32


# ---------------------------------------------------------------------------
# Обучение: три режима
# ---------------------------------------------------------------------------

class TestLinearRegressionFit:
    @pytest.mark.parametrize("method", ["sgd", "gd", "norm_eq"])
    def test_fit_returns_self(self, method, linear_data):
        """fit() возвращает экземпляр модели (поддержка цепочки вызовов)."""
        X, y = linear_data
        model = LinearRegression(method=method, lr=0.1, epochs=300)
        result = model.fit(X, y)
        assert result is model

    @pytest.mark.parametrize("method", ["sgd", "gd", "norm_eq"])
    def test_weights_initialized_after_fit(self, method, linear_data):
        """После обучения w и b должны быть инициализированы."""
        X, y = linear_data
        model = LinearRegression(method=method, lr=0.1, epochs=300).fit(X, y)
        assert model.w is not None
        assert model.w.shape == (2,)
        assert isinstance(model.b, float)

    def test_norm_eq_exact_solution(self, simple_1d):
        """norm_eq на данных без шума должен давать w ≈ 2, b ≈ 0."""
        X, y = simple_1d
        model = LinearRegression(method="norm_eq").fit(X, y)
        assert abs(model.w[0] - 2.0) < 1e-6
        assert abs(model.b) < 1e-6

    def test_norm_eq_no_loss_history(self, linear_data):
        """При norm_eq итераций нет — loss_history должен оставаться пустым."""
        X, y = linear_data
        model = LinearRegression(method="norm_eq").fit(X, y)
        assert model.loss_history == []

    @pytest.mark.parametrize("method", ["sgd", "gd"])
    def test_loss_decreases(self, method, linear_data):
        """Функция потерь должна убывать в процессе обучения (SGD/GD)."""
        X, y = linear_data
        model = LinearRegression(method=method, lr=0.1, epochs=200).fit(X, y)
        first = np.mean(model.loss_history[:5])
        last = np.mean(model.loss_history[-5:])
        assert last < first

    def test_early_stopping_triggered(self, simple_1d):
        """Ранняя остановка должна прерывать обучение до исчерпания эпох."""
        X, y = simple_1d
        model = LinearRegression(method="gd", lr=0.5, epochs=10000, batch_size=100)
        model.fit(X, y, tol=1e-4)
        assert len(model.loss_history) < 10000


# ---------------------------------------------------------------------------
# Предсказание и метрика
# ---------------------------------------------------------------------------

class TestLinearRegressionPredict:
    @pytest.mark.parametrize("method", ["sgd", "gd", "norm_eq"])
    def test_predict_shape(self, method, linear_data):
        """predict() должен возвращать вектор длиной n_samples."""
        X, y = linear_data
        model = LinearRegression(method=method, lr=0.1, epochs=300).fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (200,)

    @pytest.mark.parametrize("method", ["sgd", "gd", "norm_eq"])
    def test_r2_high_on_linear_data(self, method, linear_data):
        """R² на линейных данных должен быть выше 0.95."""
        X, y = linear_data
        model = LinearRegression(method=method, lr=0.1, epochs=500).fit(X, y)
        r2 = model.score(X, y)
        assert r2 > 0.95, f"R²={r2:.4f} при method='{method}'"

    def test_r2_perfect_fit(self, simple_1d):
        """На данных без шума norm_eq должен давать R² ≈ 1."""
        X, y = simple_1d
        model = LinearRegression(method="norm_eq").fit(X, y)
        assert model.score(X, y) > 0.9999

    def test_predict_before_fit_raises(self):
        """predict() до вызова fit() должен вызывать ошибку (w=None)."""
        model = LinearRegression()
        X = np.random.randn(10, 2)
        with pytest.raises(Exception):
            model.predict(X)
