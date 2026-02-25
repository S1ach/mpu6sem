import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def main() -> None:
    # Папка, куда CIFAR10 будет скачан
    root = "./Data_10"

    # Размер батча
    batch_size = 10

    # Трансформации (обязательно Normalize как в задании)
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # Датасеты CIFAR10
    train_set = datasets.CIFAR10(
        root=root,
        train=True,
        transform=transformations,
        download=True
    )

    test_set = datasets.CIFAR10(
        root=root,
        train=False,
        transform=transformations,
        download=True
    )

    # DataLoader'ы (shuffle=True только для train)
    # На Windows безопаснее num_workers=0 (иначе часто нужна доп. настройка multiprocessing)
    num_workers = 0 if os.name == "nt" else 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = (device.type == "cuda")

    train_data_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_data_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # Печать базовой информации
    print(f"Используется устройство: {device}")
    print(f"Количество тренировочных изображений: {len(train_set)}")
    print(f"Количество тестовых изображений: {len(test_set)}")
    print(f"Размер батча: {batch_size}")
    print(f"Количество батчей в тренировочных данных: {len(train_data_loader)}")
    print(f"Количество батчей в тестовых данных: {len(test_data_loader)}")

    # Проверка одного батча (чтобы было видно, что всё реально работает)
    images, labels = next(iter(train_data_loader))
    print(f"\nРазмерность батча изображений: {images.shape}")  # [B, 3, 32, 32]
    print(f"Размерность батча меток: {labels.shape}")        # [B]


if __name__ == "__main__":
    main()