"""1. Загрузите изображения из CIFAR10 так, как это делалось в уроке.
Примечание: обязательно примените трансформацию к ним:
transforms. Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)).
Подготовьте DataLoader для
тренировочных и тестовых данных
"""

import warnings
warnings.filterwarnings("ignore")

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

root = "./data"
BATCH_SIZE = 10

transformations = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

train_set = CIFAR10(
    train=True,
    transform=transformations,
    root=root,
    download=True
)
train_data_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

test_set = CIFAR10(
    train=False,
    transform=transformations,
    root=root,
    download=True
)

test_data_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)