import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

print(f"PyTorch версия: {torch.__version__}")
print(f"CUDA доступен: {torch.cuda.is_available()}\n")

tensor1 = torch.randn(3, 3)
tensor2 = torch.randn(3, 3)

print("Тензор 1:")
print(tensor1)
print("\nТензор 2:")
print(tensor2)
print(f"\nСумма: {tensor1 + tensor2}")
print(f"\nПоэлементное умножение: {tensor1 * tensor2}")
print(f"\nТранспонирование тензора 2: {tensor2.T}")
print(f"\nСредние: tensor1={tensor1.mean():.4f}, tensor2={tensor2.mean():.4f}")
print(f"Максимумы: tensor1={tensor1.max():.4f}, tensor2={tensor2.max():.4f}\n")


class MultiplicationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


np.random.seed(42)
torch.manual_seed(42)
train_data = np.random.randn(100, 2) * 5
train_targets = train_data[:, 0] * train_data[:, 1]

x_train = torch.FloatTensor(train_data)
y_train = torch.FloatTensor(train_targets).unsqueeze(1)

model = MultiplicationModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Обучение модели")
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Эпоха [{epoch + 1}], Потери: {loss.item():.6f}')


test_data = np.random.randn(10, 2) * 5
test_targets = test_data[:, 0] * test_data[:, 1]

x_test = torch.FloatTensor(test_data)
y_test = torch.FloatTensor(test_targets).unsqueeze(1)

model.eval()
with torch.no_grad():
    predictions = model(x_test)

print("\nРезультаты тестирования:")
for i in range(3):
    a, b = test_data[i]
    expected = test_targets[i]
    predicted = predictions[i].item()
    print(f"({a:.2f} * {b:.2f}) = {expected:.4f}, предсказано: {predicted:.4f}")

test_error = torch.abs(predictions - y_test).mean().item()
print(f"\nСредняя погрешность: {test_error:.6f}")

torch.save(model.state_dict(), 'multiplication_model.pth')
print("\nМодель сохранена")

loaded_model = MultiplicationModel()
loaded_model.load_state_dict(torch.load('multiplication_model.pth'))
loaded_model.eval()
