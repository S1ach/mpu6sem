import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 20
NUM_CLASSES = 10

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_set = torchvision.datasets.CIFAR10(
    root='./Data_10',
    train=False,
    download=False,
    transform=transform
)

test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=True
)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# модель с ядрами 3x3
class CnnModel_v1(nn.Module):

    def __init__(self, num_classes=10):
        super(CnnModel_v1, self).__init__()
        # Изменено: все ядра теперь 3x3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(24)

        self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(24)

        self.fc = nn.Linear(24 * 16 * 16, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.pool(out)
        out = F.relu(self.bn3(self.conv3(out)))
        out = F.relu(self.bn4(self.conv4(out)))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# 2. Модель с новой архитектурой
class CnnModel_v2(nn.Module):
    def __init__(self, num_classes=10):
        super(CnnModel_v2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.fc = nn.Linear(32 * 16 * 16, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = out.view(out.size(0), -1)

        out = self.fc(out)
        return out


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()


print("=" * 60)
print("ЗАДАНИЯ 3, 4, 5: Анализ на 20 изображениях")
print("=" * 60)

# Используем модель v2 для демонстрации
model = CnnModel_v2(num_classes=NUM_CLASSES)
model.eval()

with torch.no_grad():
    # Получаем batch из 20 изображений
    images, labels = next(iter(test_loader))

    # Прямой проход через сеть
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    # Вывод 20 изображений
    print("\nОтображение 20 изображений (закрывайте окно для продолжения)...")
    imshow(torchvision.utils.make_grid(images, nrow=5))

    # Вывод результатов в консоль
    print("\n{:<20} {:<20} {}".format("ПРОГНОЗ СЕТИ", "ПРАВИЛЬНЫЙ ОТВЕТ", "РЕЗУЛЬТАТ"))
    print("-" * 65)

    correct_count = 0
    for i in range(BATCH_SIZE):
        predicted_class = classes[predicted[i]]
        true_class = classes[labels[i]]

        # Отметка о правильности
        mark = "✓" if predicted_class == true_class else "✗"
        if predicted_class == true_class:
            correct_count += 1

        print(f"{predicted_class:<20} {true_class:<20} {mark}")

    # Расчет точности
    accuracy = (correct_count / BATCH_SIZE) * 100

    print("-" * 65)
    print(f"ИТОГИ:")
    print(f"  Правильных ответов: {correct_count} из {BATCH_SIZE}")
    print(f"  Точность на выборке: {accuracy:.2f}%")

# Дополнительная информация о моделях
print("\n" + "=" * 60)
print("АНАЛИЗ МОДЕЛЕЙ")
print("=" * 60)

print("\nМодель v1 (ядра 3x3):")
print("  - Скорость: Выше (меньше параметров → меньше вычислений)")
print("  - Точность: Выше (по сравнению с исходной моделью)")

print("\nМодель v2 (Conv → BN → ReLU → Pool → Conv → BN → ReLU → FC):")
print("  - Скорость: Выше (упрощенная архитектура)")
print("  - Точность: Ниже (меньше слоев для извлечения признаков)")