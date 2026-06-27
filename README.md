# numl — ML-алгоритмы с нуля на NumPy

> Учебная реализация трёх фундаментальных алгоритмов машинного обучения без использования sklearn, PyTorch или TensorFlow.  
> Написана с целью закрепить теорию на практике.

```
LinearRegression · LogisticRegression · NeuralNetwork
```

---

## Структура проекта

```
numl/
├── numl/
│   ├── base.py        # BaseModel: цикл SGD, ранняя остановка
│   ├── linear.py      # LinearRegression (SGD / GD / normal equation)
│   ├── logistic.py    # LogisticRegression (SGD / GD)
│   ├── neural.py      # NeuralNetwork 784→128→64→10 (backprop)
│   └── __init__.py
├── tests/
│   ├── test_linear.py    # 22 теста
│   ├── test_logistic.py  # 21 тест
│   └── test_neural.py    # 34 теста
├── examples/
│   ├── california_housing.ipynb   # LinearRegression на реальных данных
│   ├── breast_cancer.ipynb        # LogisticRegression + анализ порога
│   └── mnist_neural.ipynb         # NeuralNetwork на MNIST
└── requirements.txt
```

---

## Математика

Ниже — формулы, которые реализованы в коде. Каждый раздел содержит ссылку на соответствующий файл.

### Линейная регрессия [`linear.py`](numl/linear.py)

Модель: $\hat{y} = \mathbf{w}^\top \mathbf{x} + b$

Функция потерь (MSE):

$$L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

Градиент и шаг обновления:

$$\frac{\partial L}{\partial \mathbf{w}} = \frac{1}{m} \mathbf{X}^\top (\hat{\mathbf{y}} - \mathbf{y}), \qquad \mathbf{w} \leftarrow \mathbf{w} - \alpha \cdot \frac{\partial L}{\partial \mathbf{w}}$$

Аналитическое решение (нормальное уравнение):

$$\boldsymbol{\theta} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$$

Реализовано через `np.linalg.lstsq` для численной устойчивости при вырожденных матрицах.

---

### Логистическая регрессия [`logistic.py`](numl/logistic.py)

Функция активации — сигмоид:

$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = \mathbf{w}^\top \mathbf{x} + b$$

Функция потерь (Binary Cross-Entropy, выводится из MLE):

$$L = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log \hat{y}_i + (1 - y_i) \log (1 - \hat{y}_i) \right]$$

После упрощения через цепное правило градиент принимает ту же форму, что и в линейной регрессии — это следствие выпуклости BCE:

$$\frac{\partial L}{\partial \mathbf{w}} = \frac{1}{m} \mathbf{X}^\top (\hat{\mathbf{y}} - \mathbf{y})$$

Благодаря этому `LogisticRegression` и `LinearRegression` используют один и тот же цикл обучения в `BaseModel`, переопределяя только `_activation()` и `_compute_loss()`.

> **Почему `norm_eq` запрещён**: BCE невыпукла относительно $\mathbf{w}$ в замкнутой форме — аналитического решения не существует, только итеративная оптимизация.

---

### Нейронная сеть [`neural.py`](numl/neural.py)

Архитектура:

```
Вход (784) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)
```

**Прямой проход** для слоя $l$:

$$\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}, \qquad \mathbf{a}^{[l]} = g^{[l]}(\mathbf{z}^{[l]})$$

**Функция потерь** (Categorical Cross-Entropy):

$$L = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_{ik} \log \hat{y}_{ik}$$

**Обратный проход** (цепное правило):

$$\boldsymbol{\delta}^{[3]} = \hat{\mathbf{Y}} - \mathbf{Y} \quad \text{(softmax + CCE упрощаются)}$$

$$\boldsymbol{\delta}^{[l]} = \left(\mathbf{W}^{[l+1]\top} \boldsymbol{\delta}^{[l+1]}\right) \odot \text{ReLU}'(\mathbf{z}^{[l]})$$

$$\frac{\partial L}{\partial \mathbf{W}^{[l]}} = \frac{1}{m} \boldsymbol{\delta}^{[l]} \mathbf{a}^{[l-1]\top}, \qquad \frac{\partial L}{\partial \mathbf{b}^{[l]}} = \frac{1}{m} \sum \boldsymbol{\delta}^{[l]}$$

**Инициализация весов по He** (для ReLU-сетей):

$$\mathbf{W}^{[l]} \sim \mathcal{N}\!\left(0,\ \frac{2}{n^{[l-1]}}\right)$$

---

## Быстрый старт

```bash
git clone https://github.com/KarIes-ss/numl.git
cd numl
pip install -r requirements.txt
```

```python
from numl import LinearRegression, LogisticRegression, NeuralNetwork
```

### LinearRegression

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from numl import LinearRegression

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Три режима обучения
model = LinearRegression(method="norm_eq").fit(X_train, y_train)
print(f"R² = {model.score(X_test, y_test):.4f}")

model = LinearRegression(method="sgd", lr=0.1, epochs=300).fit(X_train, y_train)
model.plot_loss()
```

### LogisticRegression

```python
from sklearn.datasets import load_breast_cancer
from numl import LogisticRegression

X, y = load_breast_cancer(return_X_y=True)
# ... предобработка ...

model = LogisticRegression(lr=0.1, epochs=500, threshold=0.5)
model.fit(X_train, y_train)

proba  = model.predict(X_test)        # вероятности ∈ (0, 1)
labels = model.predict_class(X_test)  # метки {0, 1}
print(f"Accuracy = {model.score(X_test, y_test):.4f}")
```

### NeuralNetwork

```python
from numl import NeuralNetwork

# Ожидается X нормализованный в [0, 1], y ∈ {0, …, 9}
model = NeuralNetwork(hidden1=128, hidden2=64, lr=0.01, epochs=20)
model.fit(X_train, y_train)
model.plot_history()

preds = model.predict(X_test)
print(f"Accuracy = {model.score(X_test, y_test):.4f}")
```

---

## Запуск тестов

```bash
# Установка зависимостей (только numpy и pytest)
pip install -r requirements.txt

# Все тесты
python -m pytest tests/ -v

# Один файл
python -m pytest tests/test_linear.py -v

# Конкретный класс или тест
python -m pytest tests/test_logistic.py::TestLogisticNormEqForbidden -v
python -m pytest tests/test_neural.py::TestActivations::test_softmax_numerical_stability -v
```

**Покрытие:** 77 тестов, ~1.6 сек.

| Файл | Тестов | Что проверяется |
|------|--------|-----------------|
| `test_linear.py` | 22 | Три метода обучения, R², ранняя остановка, нормальное уравнение |
| `test_logistic.py` | 21 | BCE, сигмоид, порог классификации, запрет `norm_eq` |
| `test_neural.py` | 34 | Формы тензоров, softmax, backprop, сходимость, `predict_proba` |

Тесты написаны на `pytest` с использованием фикстур и параметризации (`@pytest.mark.parametrize`). Покрывается как поведение публичного API, так и внутренняя математика (`_activation`, `_compute_loss`, `_relu_grad`).

---

## Архитектурные решения

**Шаблонный метод (Template Method) в `BaseModel`**

Весь цикл обучения (SGD, ранняя остановка, история потерь) реализован единожды в `BaseModel.fit()`. Подклассы переопределяют только два метода:

```python
def _activation(self, z): ...   # sigmoid / identity
def _compute_loss(self, y, y_pred): ...  # BCE / MSE
```

Это устраняет дублирование и позволяет добавить новую модель (например, `SVMRegression`) без изменения цикла обучения.

**Численная стабильность**

- Сигмоид: `np.clip(z, -500, 500)` предотвращает `exp(-z)` overflow
- Softmax: вычитание `max(z)` по столбцам перед `exp` (стандартный трюк log-sum-exp)
- BCE и CCE: `eps = 1e-9` защищает от `log(0)` на граничных предсказаниях

**Нормальное уравнение через `lstsq`**

Вместо явного обращения $(\mathbf{X}^\top\mathbf{X})^{-1}$ используется `np.linalg.lstsq`, которая применяет SVD-разложение — устойчива при мультиколлинеарности и вырожденных матрицах.

---

## Результаты на реальных данных

| Датасет | Модель | Метрика | Значение |
|---------|--------|---------|----------|
| California Housing | `LinearRegression (norm_eq)` | R² | ~0.606 |
| Breast Cancer Wisconsin | `LogisticRegression (sgd)` | Accuracy | ~0.982 |
| MNIST | `NeuralNetwork (128→64)` | Accuracy | ~0.970 |

---

## Зависимости

```
numpy>=1.24
matplotlib>=3.7
pytest>=7.0        # только для тестов
scikit-learn>=1.3  # только для примеров (загрузка датасетов)
```

---
