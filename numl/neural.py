import numpy as np
import matplotlib.pyplot as plt


class MNISTNeuralNetwork:
    """
    Двухслойная нейросеть прямого распространения для классификации.
    Использовать только для картинок 28x28 (784 входа).

    Архитектура:
        Вход (784) → Скрытый 1 (128, ReLU) → Скрытый 2 (64, ReLU) → Выход (10, Softmax)

    Каждый слой: z = Wa + b,  a = activation(z)
    Loss: Categorical Cross-Entropy  L = −Σ yᵢ · log(ŷᵢ)

    Обратный проход строится по цепному правилу:
        δ на выходе     = ŷ − y   (softmax + CCE упрощаются)
        δ скрытого слоя = (Wᵀ · δ_следующего) * ReLU'(z)
        ∂L/∂W           = δ · aᵀ  (вход слоя транспонирован)
        ∂L/∂b           = δ

    Attributes:
        sizes (tuple): Размеры слоёв (вход, скрытый1, скрытый2, выход).
        lr (float): Скорость обучения.
        epochs (int): Число эпох.
        batch_size (int): Размер мини-батча.
        W, b (dict): Веса и смещения слоёв 1, 2, 3.
        loss_history (list): CCE на обучающей выборке по эпохам.
        acc_history  (list): Accuracy на обучающей выборке по эпохам.
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden1: int = 128,
        hidden2: int = 64,
        output_size: int = 10,
        lr: float = 0.01,
        epochs: int = 20,
        batch_size: int = 64,
    ):
        """
        Args:
            input_size:  Размер входного вектора (784 для MNIST 28×28).
            hidden1:     Число нейронов в первом скрытом слое.
            hidden2:     Число нейронов во втором скрытом слое.
            output_size: Число классов (10 для MNIST).
            lr:          Скорость обучения.
            epochs:      Число полных проходов по данным.
            batch_size:  Размер мини-батча для SGD.
        """
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_history: list[float] = []
        self.acc_history:  list[float] = []

        # Инициализация весов по He:
        # std = sqrt(2 / n_in)
        self.W: dict[int, np.ndarray] = {
            1: np.random.randn(hidden1,      input_size) * np.sqrt(2 / input_size),
            2: np.random.randn(hidden2,      hidden1)    * np.sqrt(2 / hidden1),
            3: np.random.randn(output_size,  hidden2)    * np.sqrt(2 / hidden2),
        }
        self.b: dict[int, np.ndarray] = {
            1: np.zeros((hidden1, 1)),
            2: np.zeros((hidden2, 1)),
            3: np.zeros((output_size, 1)),
        }

    # ──────────────────────────────────────────────────────────────
    # Активационные функции и их производные
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _relu(z: np.ndarray) -> np.ndarray:
        """ReLU: max(0, z). Работает поэлементно."""
        return np.maximum(0, z)

    @staticmethod
    def _relu_grad(z: np.ndarray) -> np.ndarray:
        """
        Производная ReLU: 1 если z > 0, иначе 0.
        В backprop умножаем на неё, чтобы "заглушить" мёртвые нейроны.
        """
        return (z > 0).astype(float)

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        """
        Softmax: eᶻⁱ / Σeᶻʲ по столбцам (каждый столбец — один объект).
        Вычитаем max(z) для числовой стабильности — результат не меняется,
        но предотвращаем exp(большое число) → overflow.
        """
        z_stable = z - np.max(z, axis=0, keepdims=True)
        exp_z = np.exp(z_stable)
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    # ──────────────────────────────────────────────────────────────
    # Прямой проход
    # ──────────────────────────────────────────────────────────────

    def _forward(self, X: np.ndarray) -> dict:
        """
        Прямой проход: от входа до вероятностей.

        Сохраняем z и a каждого слоя — они понадобятся в backprop.

        Args:
            X: Матрица (n_features, batch_size) — столбцы = объекты.

        Returns:
            cache: словарь с z и a для каждого слоя.
        """
        cache = {"a0": X}

        # Слой 1: линейное преобразование + ReLU
        cache["z1"] = self.W[1] @ cache["a0"] + self.b[1]   # (128, m)
        cache["a1"] = self._relu(cache["z1"])                # (128, m)

        # Слой 2: линейное преобразование + ReLU
        cache["z2"] = self.W[2] @ cache["a1"] + self.b[2]   # (64, m)
        cache["a2"] = self._relu(cache["z2"])                # (64, m)

        # Выходной слой: линейное преобразование + Softmax
        cache["z3"] = self.W[3] @ cache["a2"] + self.b[3]   # (10, m)
        cache["a3"] = self._softmax(cache["z3"])             # (10, m) — вероятности

        return cache

    # ──────────────────────────────────────────────────────────────
    # Обратный проход (backprop)
    # ──────────────────────────────────────────────────────────────

    def _backward(self, cache: dict, Y: np.ndarray) -> dict:
        """
        Обратный проход: вычисляем градиенты всех весов через цепное правило.

        Ключевая идея: δˡ ("дельта слоя l") — это ∂L/∂zˡ.
        Зная δˡ, мы получаем:
            ∂L/∂Wˡ = δˡ · (aˡ⁻¹)ᵀ   (вход слоя)
            ∂L/∂bˡ = среднее(δˡ)
            δˡ⁻¹   = (Wˡ)ᵀ · δˡ · ReLU'(zˡ⁻¹)   (передаём δ назад)

        Args:
            cache: Результаты прямого прохода (z, a для каждого слоя).
            Y:     One-hot матрица меток (10, batch_size).

        Returns:
            grads: словарь градиентов dW1, db1, dW2, db2, dW3, db3.
        """
        m = Y.shape[1]  # размер батча

        # Выходной слой (слой 3)
        # Полный вывод: ∂L/∂z³ = ŷ − y
        delta3 = cache["a3"] - Y                                    # (10, m)

        dW3 = (1 / m) * delta3 @ cache["a2"].T                     # (10, 64)
        db3 = (1 / m) * np.sum(delta3, axis=1, keepdims=True)      # (10, 1)

        # Скрытый слой 2
        # Передаём ошибку назад через W3, затем "глушим" мёртвые нейроны через ReLU'
        delta2 = (self.W[3].T @ delta3) * self._relu_grad(cache["z2"])   # (64, m)

        dW2 = (1 / m) * delta2 @ cache["a1"].T                     # (64, 128)
        db2 = (1 / m) * np.sum(delta2, axis=1, keepdims=True)      # (64, 1)

        # Скрытый слой 1
        # То же самое — передаём δ ещё на шаг назад
        delta1 = (self.W[2].T @ delta2) * self._relu_grad(cache["z1"])   # (128, m)

        dW1 = (1 / m) * delta1 @ cache["a0"].T                     # (128, 784)
        db1 = (1 / m) * np.sum(delta1, axis=1, keepdims=True)      # (128, 1)

        return {"dW1": dW1, "db1": db1,
                "dW2": dW2, "db2": db2,
                "dW3": dW3, "db3": db3}

    # ──────────────────────────────────────────────────────────────
    # Обновление весов
    # ──────────────────────────────────────────────────────────────

    def _update(self, grads: dict) -> None:
        """
        Шаг SGD: W ← W − lr · ∂L/∂W, аналогично для b.

        Args:
            grads: Словарь градиентов из _backward().
        """
        for l in [1, 2, 3]:
            self.W[l] -= self.lr * grads[f"dW{l}"]
            self.b[l] -= self.lr * grads[f"db{l}"]

    # ──────────────────────────────────────────────────────────────
    # Вспомогательные функции
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _to_one_hot(y: np.ndarray, n_classes: int = 10) -> np.ndarray:
        """
        Преобразует вектор меток в one-hot матрицу.
        [3, 7] → [[0,0,0,1,...,0], [0,0,0,0,0,0,0,1,0,0]]

        Args:
            y:         Вектор меток (n_samples,) со значениями 0..n_classes-1.
            n_classes: Число классов.

        Returns:
            One-hot матрица (n_classes, n_samples).
        """
        one_hot = np.zeros((n_classes, y.size))
        one_hot[y, np.arange(y.size)] = 1
        return one_hot

    def _compute_loss(self, y_pred: np.ndarray, Y: np.ndarray) -> float:
        """
        Categorical Cross-Entropy: L = −(1/m) Σ Σ yᵢⱼ · log(ŷᵢⱼ).
        eps защищает от log(0) при числовых краях softmax.

        Args:
            y_pred: Вероятности (n_classes, m).
            Y:      One-hot метки  (n_classes, m).

        Returns:
            Скалярное значение CCE ≥ 0.
        """
        eps = 1e-9
        return float(-np.mean(np.sum(Y * np.log(y_pred + eps), axis=0)))

    def _compute_accuracy(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Доля правильно классифицированных объектов.

        Args:
            y_pred: Вероятности (n_classes, m).
            y_true: Метки (m,).

        Returns:
            Accuracy ∈ [0, 1].
        """
        return float(np.mean(np.argmax(y_pred, axis=0) == y_true))

    # ──────────────────────────────────────────────────────────────
    # Публичный API
    # ──────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MNISTNeuralNetwork":
        """
        Обучение сети методом mini-batch SGD.

        Args:
            X: Матрица признаков (n_samples, n_features). Должна быть
               нормализована в [0, 1] или стандартизована.
            y: Вектор меток (n_samples,) со значениями 0..9.

        Returns:
            self
        """
        n = X.shape[0]

        for epoch in range(self.epochs):
            # Перемешиваем данные
            idx = np.random.permutation(n)
            X_shuffled, y_shuffled = X[idx], y[idx]

            for start in range(0, n, self.batch_size):
                end = start + self.batch_size
                # Транспонируем батч
                X_batch = X_shuffled[start:end].T          # (784, m)
                y_batch = y_shuffled[start:end]            # (m,)
                Y_batch = self._to_one_hot(y_batch)        # (10, m)

                # Прямой → обратный → обновление
                cache = self._forward(X_batch)
                grads = self._backward(cache, Y_batch)
                self._update(grads)

            # Метрики на всей обучающей выборке раз в эпоху
            cache_full = self._forward(X.T)
            loss = self._compute_loss(cache_full["a3"], self._to_one_hot(y))
            acc  = self._compute_accuracy(cache_full["a3"], y)
            self.loss_history.append(loss)
            self.acc_history.append(acc)

            print(f"Эпоха {epoch + 1:3d}/{self.epochs}  "
                  f"Loss={loss:.4f}  Accuracy={acc:.4f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Вероятности принадлежности к каждому классу.

        Args:
            X: Матрица (n_samples, n_features).

        Returns:
            Матрица вероятностей (n_samples, n_classes).
        """
        cache = self._forward(X.T)
        return cache["a3"].T      # (n_samples, 10)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказание классов (argmax вероятностей).

        Args:
            X: Матрица (n_samples, n_features).

        Returns:
            Вектор предсказанных меток (n_samples,).
        """
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Accuracy на переданной выборке.

        Args:
            X: Матрица признаков (n_samples, n_features).
            y: Истинные метки (n_samples,).

        Returns:
            Доля правильных ответов ∈ [0, 1].
        """
        return float(np.mean(self.predict(X) == y))

    def plot_history(self) -> None:
        """
        Визуализация кривых обучения: Loss и Accuracy по эпохам.
        Вызывай после fit().
        """
        if not self.loss_history:
            print("Модель ещё не обучена — сначала вызови fit().")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(self.loss_history, color="#7F77DD", linewidth=2)
        ax1.set_title("Loss (CCE) по эпохам")
        ax1.set_xlabel("Эпоха")
        ax1.set_ylabel("CCE")
        ax1.grid(True, alpha=0.3)

        ax2.plot(self.acc_history, color="#1D9E75", linewidth=2)
        ax2.set_title("Accuracy по эпохам")
        ax2.set_xlabel("Эпоха")
        ax2.set_ylabel("Accuracy")
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

        final_acc = self.acc_history[-1]
        ax2.annotate(f"{final_acc:.4f}",
                     xy=(len(self.acc_history) - 1, final_acc),
                     xytext=(-40, -20), textcoords="offset points",
                     color="#1D9E75", fontweight="bold")

        plt.suptitle("Кривые обучения MNISTNeuralNetwork", fontsize=12)
        plt.tight_layout()
        plt.show()