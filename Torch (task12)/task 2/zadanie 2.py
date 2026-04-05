'''
1. Создайте нейросеть со следующей архитектурой: 5 нейронов на входе, три скрытых слоя, 1 нейрон на выходе
(бинарная классификация). При этом каждый скрытый слой должен пропускаться через различных функции активации.

2. Подумайте, какая должна быть функция активация на выходе в предыдущем задании,
с учётом того, что это бинарная классификация. Добавьте её.

3. Попробуйте поменять оптимизатор. Какой лучше работает в Вашем случае: Adam или SGD?
'''

from torch import nn
import torch

LOW = -30
HIGH = 30


class Model(nn.Module):
    INPUT_SIZE = 5
    OUTPUT_SIZE = 1

    def __init__(self, hidden_1_size, hidden_2_size, hidden_3_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(self.INPUT_SIZE, hidden_1_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_1_size, hidden_2_size)
        self.tanh = nn.Tanh()
        self.fc3 = nn.Linear(hidden_2_size, hidden_3_size)
        self.leaky_relu = nn.LeakyReLU()
        self.fc4 = nn.Linear(hidden_3_size, self.OUTPUT_SIZE)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        tmp = self.fc1(x)
        tmp = self.leaky_relu(tmp)
        tmp = self.fc2(tmp)
        tmp = self.leaky_relu(tmp)
        tmp = self.fc3(tmp)
        tmp = self.leaky_relu(tmp)
        tmp = self.fc4(tmp)
        tmp = self.sigmoid(tmp)

        return tmp


class Network:
    LOW = -50
    HIGH = 50
    LEARNING_RATE = 0.1

    def __init__(self, type_of_optim):
        self.model = Model(128, 128, 128)

        if type_of_optim == "Adam":
            self.optimizator = torch.optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)
        elif type_of_optim == "SGD":
            self.optimizator = torch.optim.SGD(self.model.parameters(), lr=self.LEARNING_RATE)

        self.loss_func = nn.BCEWithLogitsLoss()

    def generate_data_for_training(self, count_of_data):
        inp_data = torch.randint(self.LOW, self.HIGH, (count_of_data, 5), dtype=torch.float32)
        out_data = torch.randint(self.LOW, self.HIGH, (count_of_data, 1), dtype=torch.float32)

        return inp_data, out_data

    def training(self, count_of_data, train_epochs, DEBUG=False):
        self.train_inp_data, self.train_out_data = self.generate_data_for_training(count_of_data)

        for epoch in range(train_epochs):
            self.optimizator.zero_grad()
            out = self.model(self.train_inp_data)
            error = self.loss_func(out, self.train_out_data)
            error.backward()
            self.optimizator.step()

            if epoch % 50 == 0 and DEBUG:
                print("Эпоха - ", epoch, "Ошибка - ", error.item())

    def get_prediction(self, a, b):
        return self.model(torch.tensor([a]))


print("-" * 10 + "Adam" + "-" * 10)
network = Network("Adam")
network.training(2000, 500, True)
print("-" * 10 + "SGD" + "-" * 10)
network = Network("SGD")
network.training(2000, 500, True)