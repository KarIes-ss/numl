import numpy as np
import matplotlib.pyplot as plt
from .base import BaseModel


class LinearRegression(BaseModel):
    """
    Линейная регрессия на основе SGD, полного GD или нормального уравнения.

    Наследует от BaseModel весь цикл обучения. Переопределяет только:
    - _activation(): тождественная функция (z → z)
    - _compute_loss(): MSE

    Дополнительно поддерживает аналитическое решение через нормальное
    уравнение (method='norm_eq'), которое недоступно в LogisticRegression.

    Attributes:
        method (str): Метод оптимизации: 'sgd', 'gd' или 'norm_eq'.
        (остальные атрибуты унаследованы от BaseModel)

    Example:
        >>> model = LinearRegression(method='sgd', lr=0.01, epochs=500)
        >>> model.fit(X_train, y_train)
        >>> print(model.score(X_test, y_test))
    """

    SUPPORTED_METHODS = ("sgd", "gd", "norm_eq")

    def __init__(self, method: str = "sgd", lr: float = 0.01,
                 epochs: int = 1000, batch_size: int = 32):
        """
        Args:
            method: 'sgd'      — mini-batch градиентный спуск (перемешивание + батчи);
                    'gd'       — полный градиентный спуск (вся выборка за один шаг);
                    'norm_eq'  — аналитическое решение через нормальное уравнение.
            lr: Скорость обучения (игнорируется при method='norm_eq').
            epochs: Максимальное число эпох (игнорируется при method='norm_eq').
            batch_size: Размер мини-батча (используется только при method='sgd').

        Raises:
            ValueError: Если передан неподдерживаемый метод.
        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Неизвестный метод '{method}'. "
                f"Доступные: {self.SUPPORTED_METHODS}"
            )
        super().__init__(lr=lr, epochs=epochs, batch_size=batch_size)
        self.method = method

    def _activation(self, z: np.ndarray) -> np.ndarray:
        """Тождественная функция активации для линейной регрессии."""
        return z

    def _compute_loss(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Squared Error: (1/n) * Σ(y - ŷ)².

        Args:
            y: Истинные значения.
            y_pred: Предсказанные значения.

        Returns:
            Скалярное значение MSE.
        """
        return float(np.mean((y - y_pred) ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray, tol: float = 1e-6) -> "LinearRegression":
        """
        Обучение модели.

        При method='norm_eq' решает θ = (XᵀX)⁻¹Xᵀy аналитически — без итераций.
        При method='gd'      делегирует в BaseModel.fit(method='gd').
        При method='sgd'     делегирует в BaseModel.fit(method='sgd').

        Args:
            X: Матрица признаков (n_samples, n_features).
            y: Целевой вектор (n_samples,).
            tol: Порог ранней остановки (только для sgd/gd).

        Returns:
            self
        """
        if self.method == "norm_eq":
            n = X.shape[0]
            X_b = np.c_[np.ones(n), X]
            theta, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)
            self.b = float(theta[0])
            self.w = theta[1:]
            return self

        return super().fit(X, y, method=self.method, tol=tol)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Коэффициент детерминации R².

        R² = 1 - SS_res / SS_tot. Ближе к 1 — лучше.

        Args:
            X: Матрица признаков.
            y: Истинные значения.

        Returns:
            Значение R² ∈ (-∞, 1].
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return float(1 - ss_res / ss_tot)