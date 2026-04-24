import torch
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt


root = "./Data_10"

transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_set = torchvision.datasets.CIFAR10(train=True,  transform=transformations, root=root, download=True)
test_set  = torchvision.datasets.CIFAR10(train=False, transform=transformations, root=root, download=True)

batch_size = 20
train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_data_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)


class ImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,  12, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(24)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1)
        self.bn4   = nn.BatchNorm2d(24)
        self.fc    = nn.Linear(24 * 16 * 16, 10)

    def forward(self, inp):
        out = F.relu(self.bn1(self.conv1(inp)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.pool(out)
        out = F.relu(self.bn3(self.conv3(out)))
        out = F.relu(self.bn4(self.conv4(out)))
        out = out.view(-1, 24 * 16 * 16)
        return self.fc(out)



classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

model = ImageModel()
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)


def test_accuracy():
    model.eval()
    accuracy = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_data_loader:
            output = model(images)
            predict = torch.max(output.data, 1)[1]
            accuracy += (predict == labels).sum().item()
            total += labels.size(0)
    return 100 * accuracy / total


num_epochs = 3
best_accuracy = 0.0
model_save_path = './LearnModel_3x3.pth'

for epoch in range(num_epochs):
    model.train()
    for images, labels in train_data_loader:
        optimizer.zero_grad()
        output = model(images)
        err = loss(output, labels)
        err.backward()
        optimizer.step()
    acc = test_accuracy()
    if acc > best_accuracy:
        best_accuracy = acc
        torch.save(model.state_dict(), model_save_path)
    print('Epoch: %d; Accuracy: %d%%' % (epoch + 1, acc))



load_model = ImageModel()
load_model.load_state_dict(torch.load(model_save_path))
load_model.eval()

images, true_labels = next(iter(test_data_loader))
with torch.no_grad():
    output = load_model(images)
predict_labels = torch.max(output, 1)[1]

print('Правильные ответы:', [classes[i] for i in true_labels])
print('Прогноз сети:    ', [classes[i] for i in predict_labels])

correct = (predict_labels == true_labels).sum().item()
print('Точность на 20 картинках: %.1f%%' % (100 * correct / 20))

grid = torchvision.utils.make_grid(images, nrow=5)
grid = grid / 2 + 0.5
plt.figure(figsize=(12, 6))
plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
plt.axis('off')
plt.show()