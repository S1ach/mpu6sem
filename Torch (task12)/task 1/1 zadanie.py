import torch
from torch import nn

LOW = -30
HIGH = 30


def task2():
    print("Задача 2. Создание 2 тензора 3х3")
    t1 = torch.rand(3, 3)
    t2 = torch.rand(3, 3)
    print("Tensor 1:\n", t1)
    print("Tensor 2:\n", t2)
    return t1, t2


def task3(t1, t2):
    print("\nЗадача 3. Сложения тензоров")
    print(t1 + t2)


def task4(t1, t2):
    print("\nЗадача 4. Умножение")
    print(t1 * t2)


def task5(t2):
    print("\nЗадача 5. Транспонирование")
    print(t2.T)


def task6(t1, t2):
    print("\nЗадача 6. Ср знач")
    print("Ср занч t1:", torch.mean(t1))
    print("Ср занч t2:", torch.mean(t2))


def task7(t1, t2):
    print("\nЗадача 7. Макс занч")
    print("Max t1:", torch.max(t1))
    print("Max t2:", torch.max(t2))


class SimpleModel(nn.Module):
    def __init__(self, inp_size, hid_size, out_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(inp_size, hid_size)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hid_size, out_size)

    def forward(self, x):
        tmp = self.fc1(x)
        tmp = self.tanh(tmp)
        tmp = self.fc2(tmp)
        return tmp


class MultNetwork:
    def __init__(self, num_of_data, low_limit, high_limit):
        self.num_of_data = num_of_data
        self.low_limit = low_limit
        self.high_limit = high_limit
        self.num_train_epoch = 2000
        self.train_learning_rate = 0.01
        self.model = SimpleModel(2, 128, 1)
        self.optimizator = torch.optim.Adam(self.model.parameters(), lr=self.train_learning_rate)
        self.loss_func = nn.MSELoss()

    def generate_data_for_training(self):
        raw_inp = torch.randint(self.low_limit, self.high_limit, (self.num_of_data, 2), dtype=torch.float32)
        raw_out = (raw_inp[:, 0] * raw_inp[:, 1]).unsqueeze(1)

        self.inp_mean = raw_inp.mean(dim=0, keepdim=True)
        self.inp_std = raw_inp.std(dim=0, keepdim=True)
        self.out_mean = raw_out.mean()
        self.out_std = raw_out.std()

        inp_data = (raw_inp - self.inp_mean) / self.inp_std
        out_data = (raw_out - self.out_mean) / self.out_std

        return inp_data, out_data

    def training(self, debug=False):
        self.train_inp_data, self.train_out_data = self.generate_data_for_training()

        for epoch in range(self.num_train_epoch):
            self.optimizator.zero_grad()
            out = self.model(self.train_inp_data)
            error = self.loss_func(out, self.train_out_data)
            error.backward()
            self.optimizator.step()

            if epoch % 50 == 0 and debug:
                print("Эпоха:", epoch, "Ошибка:", error.item())

    def get_prediction(self, a, b):
        test_input = torch.tensor([[a, b]], dtype=torch.float32)
        test_input_std = (test_input - self.inp_mean) / self.inp_std
        normalized_pred = self.model(test_input_std).item()
        return normalized_pred * self.out_std + self.out_mean


def test_network(network, title):
    print("\n" + "=" * 100)
    print(title)
    data = (
        (3, 4), (3, 3), (2, 3), (9, 2), (0, 1),
        (5, 6), (0, 0), (14, 2), (20, 10), (30, 0)
    )

    for d in data:
        print("Данные:", d)
        print("Ожидается:", d[0] * d[1])
        print("Получено:", network.get_prediction(d[0], d[1]))

    print("=" * 100)


def save_model(network, filename):
    torch.save({
        "model_state_dict": network.model.state_dict(),
        "inp_mean": network.inp_mean,
        "inp_std": network.inp_std,
        "out_mean": network.out_mean,
        "out_std": network.out_std
    }, filename)


def load_model(filename):
    struct = torch.load(filename)
    network = MultNetwork(10, LOW, HIGH)

    network.model.load_state_dict(struct["model_state_dict"])
    network.inp_mean = struct["inp_mean"]
    network.inp_std = struct["inp_std"]
    network.out_mean = struct["out_mean"]
    network.out_std = struct["out_std"]

    return network


def task8_13():
    print("\nЗадачи 8-11. Обучение, тестирование и сохранение модели")

    network100 = MultNetwork(100, LOW, HIGH)
    network100.training()
    test_network(network100, "Модель, обученная на 100 примерах")
    save_model(network100, "model_100_samples.pth")

    network10 = MultNetwork(10, LOW, HIGH)
    network10.training()
    test_network(network10, "Модель, обученная на 10 примерах")
    save_model(network10, "model_10_samples.pth")

    network1000 = MultNetwork(1000, LOW, HIGH)
    network1000.training()
    test_network(network1000, "Модель, обученная на 1000 примерах")
    save_model(network1000, "model_1000_samples.pth")

    print("\nЗадачи 12-13. Загрузка модели из файла и повторная проверка")

    loaded100 = load_model("model_100_samples.pth")
    test_network(loaded100, "Загруженная модель на 100 примерах")

    loaded10 = load_model("model_10_samples.pth")
    test_network(loaded10, "Загруженная модель на 10 примерах")

    loaded1000 = load_model("model_1000_samples.pth")
    test_network(loaded1000, "Загруженная модель на 1000 примерах")


def main():
    t1, t2 = task2()
    task3(t1, t2)
    task4(t1, t2)
    task5(t2)
    task6(t1, t2)
    task7(t1, t2)
    task8_13()


if __name__ == "__main__":
    main()