import math
import torch
import csv
import re
from torch import nn, optim, Tensor
from torch.utils.data import random_split, Dataset, DataLoader
from statistics import mean, stdev


def convert_sex(sex_str: str) -> float:
    return 0.0 if sex_str == "Female" else 1.0


def get_device():
    device = (
        acc
        if (acc := torch.accelerator.current_accelerator(check_available=True))
        is not None
        else torch.device("cpu")
    )
    print(f"Using {device.type} device")
    return device


def task1():
    class CustomDataset(Dataset[tuple[Tensor, Tensor]]):
        def __init__(self, X_data: list[tuple[float, ...]], e_data: list[tuple[float]]):
            self.X = torch.tensor(X_data)
            self.e = torch.tensor(e_data)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
            return self.X[idx], self.e[idx]

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 10),
                nn.ReLU(),
                nn.Linear(10, 5),
                nn.ReLU(),
                nn.Linear(5, 1),
            )

        def forward(self, x):
            return self.net(x)

    print("--- Task 1 ---")
    raw_data: list[tuple[float, float, float]] = []
    with open("gender-height-weight.csv") as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # Header
        for row in csv_reader:
            sex = convert_sex(row[0])
            height = float(row[1])
            weight = float(row[2])
            raw_data.append((sex, height, weight))
    avg_height = mean([row[1] for row in raw_data])
    avg_weight = mean([row[2] for row in raw_data])
    data: list[tuple[float, float, float]] = [
        (d[0], (d[1] - avg_height) / avg_height, (d[2] - avg_weight) / avg_weight)
        for d in raw_data
    ]
    full_dataset = CustomDataset(
        [(row[0], row[1]) for row in data], [(row[2],) for row in data]
    )
    train_data_count = int(len(data) * 0.8)
    test_data_count = len(data) - train_data_count
    train_dataset, test_dataset = random_split(
        full_dataset, [train_data_count, test_data_count]
    )
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    device = get_device()
    model = NeuralNetwork().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for _ in range(200):
        for batch_X, batch_e in train_dataloader:
            batch_X = batch_X.to(device)
            batch_e = batch_e.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_e)
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        model.eval()
        total_loss = 0.0
        for batch_X, batch_e in test_dataloader:
            batch_X = batch_X.to(device)
            batch_e = batch_e.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_e)
            total_loss += loss.item()
        avg_loss = total_loss / len(test_dataloader)
        print(f"avg_loss: {math.sqrt(avg_loss) * avg_weight} pounds")


def task2():
    def convert_title(name: str):
        title_search = re.search(r" ([A-Za-z]+)\.", name)
        title = title_search.group(1) if title_search else " "

        is_mr = 1.0 if title == "Mr" else 0
        is_mrs = 1.0 if title in ["Mrs", "Mme"] else 0
        is_miss = 1.0 if title in ["Miss", "Ms", "Mlle"] else 0
        is_master = 1.0 if title == "Master" else 0

        return (is_mr, is_mrs, is_miss, is_master)

    class CustomDataset(Dataset[tuple[Tensor, Tensor]]):
        def __init__(self, X_data: list[tuple[float, ...]], e_data: list[tuple[float]]):
            self.X = torch.tensor(X_data)
            self.e = torch.tensor(e_data)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
            return self.X[idx], self.e[idx]

    class NeuralNetwork(nn.Module):
        def __init__(self, input_num: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_num, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.net(x)

    print("--- Task 2 ---")
    raw_data: list[tuple[float, ...]] = []
    with open("titanic.csv") as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # Header
        for row in csv_reader:
            survived = float(row[1])
            p_class = float(row[2])
            sex = convert_sex(row[4])
            age = float(row[5]) if row[5] != "" else 0
            sib_sp = float(row[6])
            parch = float(row[7])
            fare = float(row[9])
            is_mr, is_mrs, is_miss, is_master = convert_title(row[3])
            raw_data.append(
                (
                    survived,
                    p_class,
                    sex,
                    age,
                    sib_sp,
                    parch,
                    fare,
                    is_mr,
                    is_mrs,
                    is_miss,
                    is_master,
                )
            )
    avg_age = mean([row[3] for row in raw_data if row[3] != 0])
    stdev_age = stdev([row[3] for row in raw_data if row[3] != 0])
    avg_sib_sp = mean([row[4] for row in raw_data])
    stdev_sib_sp = stdev([row[4] for row in raw_data])
    avg_parch = mean([row[5] for row in raw_data])
    stdev_parch = stdev([row[5] for row in raw_data])
    avg_fare = mean([row[6] for row in raw_data])
    stdev_fare = stdev([row[6] for row in raw_data])
    x = [
        (
            (3 - d[1]) / 2,  # p_class: convert 1, 2, 3 to 1, 0.5, 0
            d[2],  # sex: 0, 1
            ((d[3] - avg_age) / stdev_age if d[3] != 0 else 0),  # age
            1 if d[3] != 0 else 0,  # add flag for missing age
            (d[4] - avg_sib_sp) / stdev_sib_sp,  # sib_sp
            (d[5] - avg_parch) / stdev_parch,  # parch
            (d[6] - avg_fare) / stdev_fare,  # fare
            d[7],
            d[8],
            d[9],
            d[10],
        )
        for d in raw_data
    ]
    e = [(d[0],) for d in raw_data]
    full_dataset = CustomDataset(x, e)
    train_data_count = int(len(full_dataset) * 0.8)
    test_data_count = len(full_dataset) - train_data_count
    train_dataset, test_dataset = random_split(
        full_dataset, [train_data_count, test_data_count]
    )
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    device = get_device()
    model = NeuralNetwork(len(train_dataset[0][0])).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for _ in range(500):
        running_loss = 0
        for batch_X, batch_e in train_dataloader:
            batch_X = batch_X.to(device)
            batch_e = batch_e.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_e)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    with torch.no_grad():
        model.eval()
        THRESHOLD = 0.5
        correct_count = 0
        for batch_X, batch_e in test_dataloader:
            batch_X = batch_X.to(device)
            batch_e = batch_e.to(device)
            outputs = model(batch_X)
            for [o], [e] in zip(outputs, batch_e):
                o_val = o.item()
                e_val = e.item()
                survival_status = 0
                if o_val > THRESHOLD:
                    survival_status = 1
                if survival_status == e_val:
                    correct_count += 1
        correct_rate = correct_count / len(test_dataset)
        print(f"correct_rate: {correct_rate * 100}%")


def main():
    task1()
    task2()


if __name__ == "__main__":
    main()
