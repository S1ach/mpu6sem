import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out


input_size = 3
hidden_size = 4
output_size = 3
learning_rate = 0.01
num_epochs = 1000

model = SimpleModel(input_size, hidden_size, output_size)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

x = torch.randn(100, input_size)
y = torch.randint(0, output_size, (100,))

for epoch in range(num_epochs):
    optimizer.zero_grad()
    out = model(x)
    error = loss(out, y)
    error.backward()
    optimizer.step()

    if (epoch + 1) >= 920 and (epoch + 1) <= 1000 and (epoch + 1) % 10 == 0:
        print(f'Эпоха [{epoch + 1}], Потери: {error.item():.4f}')
        print()