import numpy as np
import matplotlib.pyplot as plt
from .base import BaseModel


class LogisticRegression(BaseModel):
    """
    Логистическая регрессия для бинарной классификации на основе градиентного спуска.

    Наследует от BaseModel весь цикл обучения. Переопределяет:
    - _activation(): сигмоид σ(z) = 1 / (1 + e⁻ᶻ)
    - _compute_loss(): Binary Cross-Entropy

    Поддерживает два режима оптимизации:
    - 'sgd': стохастический градиентный спуск по мини-батчам (по умолчанию).
    - 'gd':  полный (пакетный) градиентный спуск по всей выборке.

    Аналитическое решение (norm_eq) для логистической регрессии не существует,
    поэтому метод 'norm_eq' явно запрещён с информативной ошибкой.

    Математическая основа:
        Линейный выход: z = wᵀx + b
        Вероятность:    ŷ = σ(z) ∈ (0, 1)
        Потери (BCE):   L = -(1/n) Σ [y·log(ŷ) + (1-y)·log(1-ŷ)]
        Градиент:       ∂L/∂w = (1/n) · Xᵀ(ŷ - y)   [после упрощения]
        Обновление:     w ← w - α · ∂L/∂w

    Attributes:
        threshold (float): Порог классификации для predict_class().
        (остальные атрибуты унаследованы от BaseModel)

    Example:
        >>> model = LogisticRegression(lr=0.1, epochs=1000)
        >>> model.fit(X_train, y_train)
        >>> print(model.score(X_test, y_test))
        >>> proba = model.predict(X_test)          # вероятности
        >>> labels = model.predict_class(X_test)   # метки 0/1
    """

    UNSUPPORTED_METHODS = ("norm_eq",)

    def __init__(self, lr: float = 0.1, epochs: int = 1000,
                 batch_size: int = 32, threshold: float = 0.5):
        """
        Args:
            lr: Скорость обучения. Для логистической регрессии хорошо работают
                значения 0.01–0.1 (в отличие от линейной, где часто нужно 0.001).
            epochs: Максимальное число эпох.
            batch_size: Размер мини-батча (используется только при method='sgd').
            threshold: Порог для predict_class(). По умолчанию 0.5.
        """
        super().__init__(lr=lr, epochs=epochs, batch_size=batch_size)
        self.threshold = threshold

    def fit(self, X: np.ndarray, y: np.ndarray,
            tol: float = 1e-6, method: str = "sgd") -> "LogisticRegression":
        """
        Обучение модели методом mini-batch SGD.

        Args:
            X: Матрица признаков (n_samples, n_features). Должна быть
               стандартизована — сигмоид чувствителен к масштабу признаков.
            y: Бинарные метки {0, 1} формы (n_samples,).
            tol: Порог ранней остановки.
            method: Режим оптимизации:
                    - 'sgd' — стохастический градиентный спуск (по умолчанию);
                    - 'gd'  — полный градиентный спуск по всей выборке.
                    Использование 'norm_eq' вызовет ValueError.

        Returns:
            self

        Raises:
            ValueError: При method='norm_eq'.
        """
        if method in self.UNSUPPORTED_METHODS:
            raise ValueError(
                "LogisticRegression не поддерживает method='norm_eq': "
                "для логистической регрессии не существует аналитического решения. "
            )
        return super().fit(X, y, method=method, tol=tol)

    def _activation(self, z: np.ndarray) -> np.ndarray:
        """
        Сигмоидная функция активации: σ(z) = 1 / (1 + e⁻ᶻ).

        Преобразует линейный выход z ∈ (-∞, +∞) в вероятность ∈ (0, 1).
        Численно стабильная реализация через np.clip предотвращает overflow
        при больших отрицательных z.

        Args:
            z: Линейный выход формы (n_samples,).

        Returns:
            Вероятности формы (n_samples,).
        """
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def _compute_loss(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Binary Cross-Entropy: -(1/n) Σ [y·log(ŷ) + (1-y)·log(1-ŷ)].

        BCE выводится из максимизации правдоподобия (MLE) и является
        выпуклой функцией — гарантирует глобальный минимум в отличие от MSE.

        Args:
            y: Бинарные метки {0, 1} формы (n_samples,).
            y_pred: Предсказанные вероятности ∈ (0, 1) формы (n_samples,).

        Returns:
            Скалярное значение BCE ≥ 0.
        """
        eps = 1e-9  # защита от log(0) при ŷ = 0.0 или ŷ = 1.0
        return float(-np.mean(
            y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps)
        ))

    def predict_class(self, X: np.ndarray,
                      threshold: float = None) -> np.ndarray:
        """
        Предсказание меток классов {0, 1}.

        Применяет порог к вероятностям из predict():
            класс = 1, если ŷ ≥ threshold, иначе 0.

        Args:
            X: Матрица признаков формы (n_samples, n_features).
            threshold: Порог классификации. Если None — используется
                       self.threshold, заданный при инициализации.

        Returns:
            Целочисленный вектор меток {0, 1} формы (n_samples,).
        """
        t = threshold if threshold is not None else self.threshold
        return (self.predict(X) >= t).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Точность классификации (accuracy).

        Accuracy = (TP + TN) / n. Для несбалансированных классов
        рассмотри дополнительные метрики: precision, recall, F1, ROC-AUC.

        Args:
            X: Матрица признаков.
            y: Истинные бинарные метки {0, 1}.

        Returns:
            Доля правильно классифицированных объектов ∈ [0, 1].
        """
        return float(np.mean(self.predict_class(X) == y))