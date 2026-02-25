import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
torch.manual_seed(42)

X, y = make_classification(
    n_samples=1000,
    n_features=5,
    n_informative=3,
    n_redundant=1,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

x_train = torch.FloatTensor(X_train_scaled)
x_test = torch.FloatTensor(X_test_scaled)
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)


class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(5, 10)  # Первый скрытый слой
        self.layer2 = nn.Linear(10, 8)  # Второй скрытый слой
        self.layer3 = nn.Linear(8, 6)  # Третий скрытый слой
        self.output = nn.Linear(6, 1)  # 1 нейрон на выходе

    def forward(self, x):
        x = torch.relu(self.layer1(x)) 
        x = torch.tanh(self.layer2(x))
        x = torch.sigmoid(self.layer3(x))

        x = torch.sigmoid(self.output(x))
        return x


def train_model(model, optimizer_type='adam', epochs=500, lr=0.01):
    criterion = nn.BCELoss()

    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    print(f"Обучение с оптимизатором: {optimizer_type.upper()}")

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(x_test)
                test_preds = (test_outputs > 0.5).float()
                accuracy = (test_preds == y_test_tensor).float().mean()

            print(f'Эпоха [{epoch + 1}/{epochs}], '
                  f'Потери: {loss.item():.6f}, '
                  f'Точность: {accuracy.item():.4f}')

    model.eval()
    with torch.no_grad():
        test_outputs = model(x_test)
        test_preds = (test_outputs > 0.5).float()
        final_accuracy = (test_preds == y_test_tensor).float().mean()

    return final_accuracy.item()


print("\nАрхитектура модели:")
print("- 5 нейронов на входе")
print("- Три скрытых слоя (10, 8, 6 нейронов)")
print("- 1 нейрон на выходе")
print("- Разные функции активации для каждого скрытого слоя")
print("- Sigmoid на выходе (для бинарной классификации)\n")

model_adam = BinaryClassifier()
accuracy_adam = train_model(model_adam, optimizer_type='adam', epochs=500)

model_sgd = BinaryClassifier()
accuracy_sgd = train_model(model_sgd, optimizer_type='sgd', epochs=500)

print(f"\nAdam - Финальная точность: {accuracy_adam:.4f}")
print(f"SGD - Финальная точность: {accuracy_sgd:.4f}")

if accuracy_adam > accuracy_sgd:
    print(f"\nЛучше работает Adam (на {accuracy_adam - accuracy_sgd:.4f} точнее)")
elif accuracy_sgd > accuracy_adam:
    print(f"\nЛучше работает SGD (на {accuracy_sgd - accuracy_adam:.4f} точнее)")
else:
    print("\nОба оптимизатора показали одинаковый результат")

print("\nСкрытые слои:")
print("1. ReLU - популярный выбор, решает проблему затухающих градиентов")
print("2. Tanh - диапазон [-1, 1], центрированные выходы")
print("3. Sigmoid - диапазон [0, 1], подходит для вероятностей")
print("\nВыходной слой:")
print("Sigmoid - преобразует выход в вероятность [0, 1], идеально для бинарной классификации")
print("и хорошо сочетается с BCELoss (Binary Cross Entropy Loss)")