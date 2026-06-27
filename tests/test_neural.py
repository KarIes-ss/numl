"""
Тесты для класса MNISTNeuralNetwork.

Проверяются инициализация весов, прямой и обратный проходы, цикл обучения,
корректность предсказаний и вычисление вспомогательных метрик.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from numl.neural import MNISTNeuralNetwork


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

@pytest.fixture
def model():
    """Сеть со стандартной архитектурой и фиксированным seed."""
    np.random.seed(42)
    return MNISTNeuralNetwork(lr=0.01, epochs=5, batch_size=64)


@pytest.fixture
def small_model():
    """Уменьшенная сеть для быстрых тестов (hidden1=16, hidden2=8)."""
    np.random.seed(0)
    return MNISTNeuralNetwork(
        input_size=784, hidden1=16, hidden2=8,
        output_size=10, lr=0.05, epochs=3, batch_size=32
    )


@pytest.fixture
def dummy_data():
    """
    Синтетический датасет: 500 изображений 28×28 (пиксели ∈ [0, 1]),
    метки — 10 равномерно распределённых классов.
    """
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, (500, 784)).astype(np.float32)
    y = np.tile(np.arange(10), 50)  # 50 примеров на класс
    return X, y


@pytest.fixture
def tiny_data():
    """
    Минимальный датасет: 20 объектов, по 2 на класс.
    Используется в тестах прямого прохода и формы тензоров.
    """
    rng = np.random.default_rng(2)
    X = rng.standard_normal((20, 784))
    y = np.repeat(np.arange(10), 2)
    return X, y


# ---------------------------------------------------------------------------
# Инициализация
# ---------------------------------------------------------------------------

class TestNeuralNetworkInit:
    def test_weight_shapes(self, model):
        """Форма весовых матриц должна соответствовать архитектуре сети."""
        assert model.W[1].shape == (128, 784)
        assert model.W[2].shape == (64,  128)
        assert model.W[3].shape == (10,  64)

    def test_bias_shapes(self, model):
        """Векторы смещений должны быть столбцами нужных размеров."""
        assert model.b[1].shape == (128, 1)
        assert model.b[2].shape == (64,  1)
        assert model.b[3].shape == (10,  1)

    def test_biases_initialized_to_zero(self, model):
        """Смещения инициализируются нулями."""
        for l in [1, 2, 3]:
            assert np.all(model.b[l] == 0)

    def test_weights_nonzero(self, model):
        """Веса инициализируются ненулевыми значениями (инициализация He)."""
        for l in [1, 2, 3]:
            assert np.any(model.W[l] != 0)

    def test_he_init_scale(self, model):
        """
        При инициализации He стандартное отклонение весов первого слоя
        должно быть близко к sqrt(2/784) ≈ 0.0505.
        """
        expected_std = np.sqrt(2 / 784)
        actual_std   = np.std(model.W[1])
        assert abs(actual_std - expected_std) < 0.01

    def test_empty_history_before_fit(self, model):
        """До обучения история потерь и точности должна быть пустой."""
        assert model.loss_history == []
        assert model.acc_history  == []

    def test_custom_architecture(self):
        """Конструктор должен поддерживать произвольные размеры слоёв."""
        net = MNISTNeuralNetwork(input_size=100, hidden1=32, hidden2=16, output_size=5)
        assert net.W[1].shape == (32, 100)
        assert net.W[2].shape == (16, 32)
        assert net.W[3].shape == (5,  16)


# ---------------------------------------------------------------------------
# Активационные функции
# ---------------------------------------------------------------------------

class TestActivations:
    def test_relu_positive(self, model):
        """ReLU пропускает положительные значения без изменений."""
        z = np.array([1., 2., 3.])
        np.testing.assert_array_equal(model._relu(z), z)

    def test_relu_negative_zeroed(self, model):
        """ReLU обнуляет отрицательные значения."""
        z = np.array([-3., -1., 0.])
        np.testing.assert_array_equal(model._relu(z), np.zeros(3))

    def test_relu_grad_positive(self, model):
        """Производная ReLU равна 1 для положительных z."""
        z = np.array([0.1, 1.0, 5.0])
        np.testing.assert_array_equal(model._relu_grad(z), np.ones(3))

    def test_relu_grad_negative(self, model):
        """Производная ReLU равна 0 для отрицательных z."""
        z = np.array([-3., -0.1])
        np.testing.assert_array_equal(model._relu_grad(z), np.zeros(2))

    def test_softmax_sums_to_one(self, model):
        """Softmax по столбцам должен давать сумму 1 по каждому объекту."""
        z = np.random.randn(10, 5)
        out = model._softmax(z)
        np.testing.assert_allclose(out.sum(axis=0), np.ones(5), atol=1e-9)

    def test_softmax_non_negative(self, model):
        """Вероятности softmax должны быть неотрицательными."""
        z = np.random.randn(10, 8)
        assert np.all(model._softmax(z) >= 0)

    def test_softmax_numerical_stability(self, model):
        """Softmax не должен возвращать nan/inf при больших входных значениях."""
        z = np.array([[1000., -1000.]] * 10)
        out = model._softmax(z)
        assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# Прямой проход
# ---------------------------------------------------------------------------

class TestForwardPass:
    def test_output_shape(self, model):
        """Выходной слой должен иметь форму (10, batch_size)."""
        X = np.random.randn(784, 16)
        cache = model._forward(X)
        assert cache["a3"].shape == (10, 16)

    def test_intermediate_shapes(self, model):
        """Промежуточные активации должны соответствовать архитектуре."""
        X = np.random.randn(784, 8)
        cache = model._forward(X)
        assert cache["a1"].shape == (128, 8)
        assert cache["a2"].shape == (64,  8)

    def test_output_is_probability_distribution(self, model):
        """Выход softmax должен быть распределением вероятностей по каждому объекту."""
        X = np.random.randn(784, 10)
        cache = model._forward(X)
        col_sums = cache["a3"].sum(axis=0)
        np.testing.assert_allclose(col_sums, np.ones(10), atol=1e-9)

    def test_cache_keys_present(self, model):
        """Кэш прямого прохода должен содержать все необходимые ключи."""
        X = np.random.randn(784, 4)
        cache = model._forward(X)
        for key in ["a0", "z1", "a1", "z2", "a2", "z3", "a3"]:
            assert key in cache, f"Ключ '{key}' отсутствует в cache"


# ---------------------------------------------------------------------------
# Вспомогательные методы
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_one_hot_shape(self, model):
        """One-hot матрица должна иметь форму (n_classes, n_samples)."""
        y = np.array([0, 3, 7, 9])
        oh = model._to_one_hot(y, n_classes=10)
        assert oh.shape == (10, 4)

    def test_one_hot_correct_encoding(self, model):
        """Единица должна стоять на позиции, соответствующей метке класса."""
        y = np.array([2, 5])
        oh = model._to_one_hot(y, n_classes=10)
        assert oh[2, 0] == 1 and oh[5, 1] == 1
        assert oh.sum() == 2  # ровно по одной единице на объект

    def test_cce_non_negative(self, model):
        """CCE всегда неотрицательна."""
        y_pred = np.full((10, 5), 0.1)
        Y = model._to_one_hot(np.array([0, 1, 2, 3, 4]))
        assert model._compute_loss(y_pred, Y) >= 0

    def test_cce_perfect_prediction(self, model):
        """CCE при уверенных верных предсказаниях близка к нулю."""
        y_true = np.array([3])
        Y = model._to_one_hot(y_true)
        y_pred = np.zeros((10, 1))
        y_pred[3, 0] = 1.0 - 1e-9
        y_pred += 1e-10  # числовая стабильность
        y_pred /= y_pred.sum(axis=0)
        loss = model._compute_loss(y_pred, Y)
        assert loss < 0.01

    def test_accuracy_all_correct(self, model):
        """_compute_accuracy при полностью верных предсказаниях = 1.0."""
        y_true = np.arange(10)
        y_pred = np.zeros((10, 10))
        y_pred[np.arange(10), np.arange(10)] = 1.0
        acc = model._compute_accuracy(y_pred.T, y_true)  # (n_classes, m)
        # Исправляем на правильный порядок осей согласно реализации
        y_pred_col = np.zeros((10, 10))
        for i in range(10):
            y_pred_col[i, i] = 1.0
        assert model._compute_accuracy(y_pred_col, y_true) == 1.0

    def test_accuracy_all_wrong(self, model):
        """_compute_accuracy при полностью неверных предсказаниях = 0.0."""
        y_true = np.zeros(5, dtype=int)
        y_pred = np.zeros((10, 5))
        y_pred[1, :] = 1.0  # всегда предсказывает класс 1, а истина — 0
        assert model._compute_accuracy(y_pred, y_true) == 0.0


# ---------------------------------------------------------------------------
# Обучение
# ---------------------------------------------------------------------------

class TestFit:
    def test_fit_returns_self(self, small_model, dummy_data):
        """fit() возвращает экземпляр модели."""
        X, y = dummy_data
        result = small_model.fit(X, y)
        assert result is small_model

    def test_history_length_equals_epochs(self, small_model, dummy_data):
        """Длина loss_history должна равняться числу эпох (без ранней остановки)."""
        X, y = dummy_data
        small_model.fit(X, y)
        assert len(small_model.loss_history) == small_model.epochs
        assert len(small_model.acc_history)  == small_model.epochs

    def test_loss_non_negative_during_training(self, small_model, dummy_data):
        """Все значения потерь в ходе обучения должны быть неотрицательными."""
        X, y = dummy_data
        small_model.fit(X, y)
        assert all(l >= 0 for l in small_model.loss_history)

    def test_weights_change_after_fit(self, tiny_data):
        """Веса должны изменяться в ходе обучения."""
        net = MNISTNeuralNetwork(hidden1=16, hidden2=8, lr=0.01, epochs=2)
        W1_before = net.W[1].copy()
        X, y = tiny_data
        net.fit(X, y)
        assert not np.allclose(net.W[1], W1_before)

    def test_loss_decreases_over_training(self, dummy_data):
        """Средняя потеря в последних эпохах должна быть меньше, чем в первых."""
        net = MNISTNeuralNetwork(hidden1=32, hidden2=16, lr=0.05, epochs=20)
        X, y = dummy_data
        net.fit(X, y)
        first = np.mean(net.loss_history[:3])
        last  = np.mean(net.loss_history[-3:])
        assert last < first, f"Loss не убывает: {first:.4f} → {last:.4f}"


# ---------------------------------------------------------------------------
# Предсказание
# ---------------------------------------------------------------------------

class TestPredict:
    def test_predict_proba_shape(self, small_model, dummy_data):
        """predict_proba() должен возвращать матрицу (n_samples, n_classes)."""
        X, y = dummy_data
        small_model.fit(X, y)
        proba = small_model.predict_proba(X)
        assert proba.shape == (500, 10)

    def test_predict_proba_sums_to_one(self, small_model, dummy_data):
        """Строки predict_proba() должны суммироваться в 1."""
        X, y = dummy_data
        small_model.fit(X, y)
        row_sums = small_model.predict_proba(X).sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(500), atol=1e-6)

    def test_predict_labels_shape(self, small_model, dummy_data):
        """predict() должен возвращать вектор меток длиной n_samples."""
        X, y = dummy_data
        small_model.fit(X, y)
        preds = small_model.predict(X)
        assert preds.shape == (500,)

    def test_predict_labels_in_valid_range(self, small_model, dummy_data):
        """Предсказанные метки должны принадлежать множеству {0, …, 9}."""
        X, y = dummy_data
        small_model.fit(X, y)
        preds = small_model.predict(X)
        assert np.all((preds >= 0) & (preds <= 9))

    def test_score_above_chance(self, dummy_data):
        """После достаточного обучения accuracy должна превышать случайный уровень (0.1)."""
        net = MNISTNeuralNetwork(hidden1=64, hidden2=32, lr=0.05, epochs=30)
        X, y = dummy_data
        net.fit(X, y)
        assert net.score(X, y) > 0.15, "Модель не лучше случайного угадывания"
