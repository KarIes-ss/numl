import numpy as np
import matplotlib.pyplot as plt


class BaseModel:
    """
    Базовый класс для моделей линейного семейства.

    Реализует общий цикл mini-batch SGD, раннюю остановку и визуализацию
    функции потерь. Конкретные модели наследуют этот класс и переопределяют
    методы _activation() и _compute_loss().

    Attributes:
        lr (float): Скорость обучения (шаг градиентного спуска).
        epochs (int): Максимальное число эпох обучения.
        batch_size (int): Размер мини-батча для SGD.
        w (np.ndarray): Вектор весов признаков; инициализируется в fit().
        b (float): Свободный член (bias); инициализируется в fit().
        loss_history (list[float]): История значений функции потерь по эпохам.
    """

    def __init__(self, lr: float = 0.01, epochs: int = 1000, batch_size: int = 32):
        """
        Args:
            lr: Скорость обучения.
            epochs: Максимальное число полных проходов по обучающей выборке.
            batch_size: Число объектов в одном мини-батче.
        """
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.w = None
        self.b = 0.0
        self.loss_history = []

    def _activation(self, z: np.ndarray) -> np.ndarray:
        """
        Активационная функция, применяемая к линейному выходу z = Xw + b.

        В LinearRegression — тождественная (return z).
        В LogisticRegression — сигмоид (return 1 / (1 + exp(-z))).

        Args:
            z: Линейный выход формы (n_samples,).

        Returns:
            Преобразованные значения той же формы.

        Raises:
            NotImplementedError: Если дочерний класс не переопределил метод.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} должен переопределить метод _activation()"
        )

    def _compute_loss(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Функция потерь для текущих предсказаний.

        В LinearRegression — MSE.
        В LogisticRegression — Binary Cross-Entropy.

        Args:
            y: Истинные значения / метки формы (n_samples,).
            y_pred: Предсказания модели формы (n_samples,).

        Returns:
            Скалярное значение потерь.

        Raises:
            NotImplementedError: Если дочерний класс не переопределил метод.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} должен переопределить метод _compute_loss()"
        )

    def fit(self, X: np.ndarray, y: np.ndarray,
            method: str = "sgd", tol: float = 1e-6) -> "BaseModel":
        """
        Обучение модели градиентным спуском с ранней остановкой.

        Поддерживает два итеративных режима, задаваемых параметром method:
        - 'sgd': на каждой эпохе данные перемешиваются и нарезаются на батчи
                 размером batch_size. Шумные обновления, быстрая сходимость
                 на больших датасетах.
        - 'gd':  весь датасет используется как один батч. Детерминированный
                 градиент, медленнее на больших данных, зато кривая loss гладкая.

        Градиенты в обоих случаях: (1/m) * Xᵀ(ŷ - y), где ŷ = _activation(Xw + b).
        Благодаря алгебраическому упрощению BCE + сигмоид формула одинакова
        для LinearRegression и LogisticRegression — меняется только то, чем является ŷ.

        Args:
            X: Матрица признаков формы (n_samples, n_features). Ожидается
               предварительно стандартизованная (mean=0, std=1).
            y: Целевой вектор формы (n_samples,).
               Для LogisticRegression — метки {0, 1}.
            method: 'sgd' (mini-batch) или 'gd' (полный батч).
            tol: Порог ранней остановки; обучение прекращается, если изменение
                 функции потерь между эпохами меньше tol.

        Returns:
            self — для возможности цепочки вызовов: model.fit(X, y).predict(X_test).
        """
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        self.loss_history = []
        prev_loss = float("inf")

        for epoch in range(self.epochs):
            if method == "gd":
                batches = [(X, y)]
            else:
                indices = np.random.permutation(n)
                batches = [
                    (X[indices[s : s + self.batch_size]],
                     y[indices[s : s + self.batch_size]])
                    for s in range(0, n, self.batch_size)
                ]

            for X_b, y_b in batches:
                m = len(y_b)
                z = X_b @ self.w + self.b
                error = self._activation(z) - y_b

                self.w -= self.lr / m * (X_b.T @ error)
                self.b -= self.lr / m * error.sum()

            y_full_pred = self._activation(X @ self.w + self.b)
            current_loss = self._compute_loss(y, y_full_pred)
            self.loss_history.append(current_loss)

            if self._is_converged(current_loss, prev_loss, tol):
                print(f"Ранняя остановка на эпохе {epoch + 1} "
                      f"(Δloss = {abs(prev_loss - current_loss):.2e} < tol={tol})")
                break
            prev_loss = current_loss

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказание выхода модели.

        Для LinearRegression возвращает вещественные числа.
        Для LogisticRegression возвращает вероятности ∈ (0, 1).
        Чтобы получить метки классов, используй predict_class().

        Args:
            X: Матрица признаков формы (n_samples, n_features).

        Returns:
            Вектор предсказаний формы (n_samples,).
        """
        return self._activation(X @ self.w + self.b)

    def _is_converged(self, current_loss: float, prev_loss: float, tol: float) -> bool:
        """
        Проверка критерия ранней остановки.

        Args:
            current_loss: Значение функции потерь на текущей эпохе.
            prev_loss: Значение функции потерь на предыдущей эпохе.
            tol: Порог сходимости.

        Returns:
            True, если |prev_loss - current_loss| < tol.
        """
        return abs(prev_loss - current_loss) < tol

    def plot_loss(self) -> None:
        """
        Визуализация истории функции потерь по эпохам.

        Отображает кривую убывания loss с отметкой финального значения.
        Вызывать после fit().
        """
        if not self.loss_history:
            print("Модель ещё не обучена — сначала вызови fit().")
            return

        last_loss = self.loss_history[-1]
        last_epoch = len(self.loss_history) - 1

        plt.figure(figsize=(8, 4))
        plt.plot(self.loss_history, label="Loss", color="#7F77DD")
        plt.scatter(last_epoch, last_loss, color="red", zorder=5)
        plt.annotate(
            f"{last_loss:.4f}",
            xy=(last_epoch, last_loss),
            xytext=(-40, 12),
            textcoords="offset points",
            color="red",
            fontweight="bold",
        )
        plt.title(f"История функции потерь — {self.__class__.__name__}")
        plt.xlabel("Эпоха")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()