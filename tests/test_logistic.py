"""
Тесты для класса LogisticRegression.

Проверяются методы обучения (sgd, gd), запрет norm_eq,
корректность вероятностей, метрика accuracy и пороговая классификация.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from numl.logistic import LogisticRegression


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_data():
    """
    Линейно разделимые классы.
    Класс 0: центр (-2, -2), класс 1: центр (2, 2).
    """
    rng = np.random.default_rng(0)
    X0 = rng.normal(-2, 0.8, (150, 2))
    X1 = rng.normal( 2, 0.8, (150, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 150 + [1] * 150)
    return X, y


@pytest.fixture
def noisy_data():
    """Труднее разделимые классы с перекрытием (для проверки устойчивости)."""
    rng = np.random.default_rng(7)
    X0 = rng.normal(-0.5, 1.0, (200, 2))
    X1 = rng.normal( 0.5, 1.0, (200, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 200 + [1] * 200)
    return X, y


# ---------------------------------------------------------------------------
# Инициализация
# ---------------------------------------------------------------------------

class TestLogisticRegressionInit:
    def test_default_params(self):
        """Параметры по умолчанию соответствуют документации."""
        model = LogisticRegression()
        assert model.lr == 0.1
        assert model.epochs == 1000
        assert model.batch_size == 32
        assert model.threshold == 0.5

    def test_custom_threshold(self):
        """Порог классификации задаётся при инициализации."""
        model = LogisticRegression(threshold=0.7)
        assert model.threshold == 0.7


# ---------------------------------------------------------------------------
# Запрет norm_eq
# ---------------------------------------------------------------------------

class TestLogisticNormEqForbidden:
    def test_norm_eq_raises_value_error(self, binary_data):
        """Вызов fit() с method='norm_eq' должен вызывать ValueError."""
        X, y = binary_data
        model = LogisticRegression()
        with pytest.raises(ValueError, match="norm_eq"):
            model.fit(X, y, method="norm_eq")

    def test_norm_eq_error_message_informative(self, binary_data):
        """Сообщение об ошибке должно объяснять причину запрета."""
        X, y = binary_data
        model = LogisticRegression()
        with pytest.raises(ValueError, match="аналитического решения"):
            model.fit(X, y, method="norm_eq")


# ---------------------------------------------------------------------------
# Обучение
# ---------------------------------------------------------------------------

class TestLogisticRegressionFit:
    @pytest.mark.parametrize("method", ["sgd", "gd"])
    def test_fit_returns_self(self, method, binary_data):
        """fit() возвращает экземпляр модели."""
        X, y = binary_data
        model = LogisticRegression(lr=0.1, epochs=200)
        assert model.fit(X, y, method=method) is model

    @pytest.mark.parametrize("method", ["sgd", "gd"])
    def test_loss_decreases(self, method, binary_data):
        """BCE должна убывать в ходе обучения."""
        X, y = binary_data
        model = LogisticRegression(lr=0.1, epochs=300).fit(X, y, method=method)
        assert np.mean(model.loss_history[:10]) > np.mean(model.loss_history[-10:])

    def test_weights_shape_after_fit(self, binary_data):
        """Форма весов должна соответствовать числу признаков."""
        X, y = binary_data
        model = LogisticRegression(lr=0.1, epochs=100).fit(X, y)
        assert model.w.shape == (2,)

    def test_early_stopping_triggered(self, binary_data):
        """Ранняя остановка должна прерывать обучение на хорошо разделимых данных."""
        X, y = binary_data
        model = LogisticRegression(lr=0.5, epochs=5000)
        model.fit(X, y, method="gd", tol=1e-5)
        assert len(model.loss_history) < 5000


# ---------------------------------------------------------------------------
# Предсказание вероятностей
# ---------------------------------------------------------------------------

class TestLogisticPredict:
    def test_predict_returns_probabilities(self, binary_data):
        """predict() должен возвращать значения строго в (0, 1)."""
        X, y = binary_data
        model = LogisticRegression(lr=0.1, epochs=200).fit(X, y)
        proba = model.predict(X)
        assert proba.shape == (300,)
        assert np.all(proba > 0) and np.all(proba < 1)

    def test_predict_class_binary_output(self, binary_data):
        """predict_class() должен возвращать только 0 и 1."""
        X, y = binary_data
        model = LogisticRegression(lr=0.1, epochs=200).fit(X, y)
        labels = model.predict_class(X)
        assert set(np.unique(labels)).issubset({0, 1})

    def test_predict_class_uses_instance_threshold(self, binary_data):
        """Порог экземпляра должен использоваться по умолчанию."""
        X, y = binary_data
        model = LogisticRegression(lr=0.1, epochs=200, threshold=0.9).fit(X, y)
        # С очень высоким порогом большинство объектов будет классифицировано как 0
        labels = model.predict_class(X)
        assert np.mean(labels == 0) > 0.3

    def test_predict_class_custom_threshold_overrides(self, binary_data):
        """Явный threshold в predict_class() должен перекрывать self.threshold."""
        X, y = binary_data
        model = LogisticRegression(lr=0.1, epochs=200, threshold=0.1).fit(X, y)
        # threshold=0.1 (всё → 1), но передаём 0.9 (большинство → 0)
        labels_high = model.predict_class(X, threshold=0.9)
        labels_low  = model.predict_class(X, threshold=0.1)
        assert np.mean(labels_low) > np.mean(labels_high)


# ---------------------------------------------------------------------------
# Метрика accuracy
# ---------------------------------------------------------------------------

class TestLogisticScore:
    def test_score_on_separable_data(self, binary_data):
        """На линейно разделимых данных accuracy должна превышать 0.95."""
        X, y = binary_data
        model = LogisticRegression(lr=0.1, epochs=500).fit(X, y)
        acc = model.score(X, y)
        assert acc > 0.95, f"Accuracy={acc:.4f}"

    def test_score_range(self, noisy_data):
        """score() всегда возвращает значение в диапазоне [0, 1]."""
        X, y = noisy_data
        model = LogisticRegression(lr=0.05, epochs=200).fit(X, y)
        acc = model.score(X, y)
        assert 0.0 <= acc <= 1.0

    def test_perfect_score_on_trivial_data(self):
        """На идеально разделимых данных должна достигаться accuracy = 1.0."""
        X = np.array([[-10.], [-9.], [9.], [10.]])
        y = np.array([0, 0, 1, 1])
        model = LogisticRegression(lr=0.5, epochs=1000).fit(X, y)
        assert model.score(X, y) == 1.0


# ---------------------------------------------------------------------------
# Активация и потери
# ---------------------------------------------------------------------------

class TestLogisticInternals:
    def test_sigmoid_bounds(self):
        """
        Сигмоид должен возвращать значения строго в (0, 1) для типичных входов.
        На границах clip (±500) float64 округляет результат до 0.0 / 1.0 —
        это ожидаемое поведение численно стабильной реализации.
        """
        model = LogisticRegression()
        z = np.array([-10., -1., 0., 1., 10.])
        out = model._activation(z)
        assert np.all(out > 0) and np.all(out < 1)

    def test_sigmoid_at_zero(self):
        """σ(0) = 0.5."""
        model = LogisticRegression()
        assert abs(model._activation(np.array([0.0]))[0] - 0.5) < 1e-9

    def test_bce_non_negative(self):
        """BCE всегда неотрицательна."""
        model = LogisticRegression()
        y      = np.array([0., 1., 0., 1.])
        y_pred = np.array([0.1, 0.9, 0.2, 0.8])
        assert model._compute_loss(y, y_pred) >= 0

    def test_bce_perfect_prediction(self):
        """BCE при идеальных предсказаниях близка к нулю."""
        model = LogisticRegression()
        y      = np.array([0., 1., 0., 1.])
        y_pred = np.array([1e-9, 1 - 1e-9, 1e-9, 1 - 1e-9])
        assert model._compute_loss(y, y_pred) < 1e-6
