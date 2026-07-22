import torch
import csv
import math
from statistics import mean, stdev
from network import BCE, Network, Layer, Re_LU, Linear, MSE, Sigmoid


def convert_sex(sex_str: str) -> float:
    return 0.0 if sex_str == "Female" else 1.0


def task1():
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
    nn = Network([Layer(2, 6, Re_LU), Layer(6, 5, Re_LU), Layer(5, 1, Linear)])
    # nn.display()
    xs = [[row[0], row[1]] for row in data]
    es = [[row[2]] for row in data]
    loss_fn = MSE
    learning_rate = 0.001
    combined = list(zip(xs, es))
    for _ in range(50):
        for x, e in combined:
            outputs = nn.forward(x)
            output_gradients = loss_fn.get_output_gradients(outputs, e)
            nn.backward(output_gradients)
            nn.zero_grad(learning_rate)
    # nn.display()
    loss_sum = 0
    for x, e in combined:
        outputs = nn.forward(x)
        loss = loss_fn.get_loss(outputs, e)
        loss_sum += loss
    avg_loss = loss_sum / len(xs)
    print(f"avg_loss: {math.sqrt(avg_loss) * avg_weight} pounds")
    # print(
    #     nn.forward([1, (73.84701702 - avg_height) / avg_height])[0] * avg_weight
    #     + avg_weight
    # )


def task2():
    print("--- Task 2 ---")
    raw_data: list[tuple[float, float, float, float, float, float, float]] = []
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
            raw_data.append((survived, p_class, sex, age, sib_sp, parch, fare))
    avg_age = mean([row[3] for row in raw_data if row[3] != 0])
    stdev_age = stdev([row[3] for row in raw_data if row[3] != 0])
    avg_sib_sp = mean([row[4] for row in raw_data])
    stdev_sib_sp = stdev([row[4] for row in raw_data])
    avg_parch = mean([row[5] for row in raw_data])
    stdev_parch = stdev([row[5] for row in raw_data])
    avg_fare = mean([row[6] for row in raw_data])
    stdev_fare = stdev([row[6] for row in raw_data])
    xs = [
        [
            (3 - d[1]) / 2,  # p_class: convert 1, 2, 3 to 1, 0.5, 0
            d[2],  # sex: 0, 1
            ((d[3] - avg_age) / stdev_age if d[3] != 0 else 0),  # age
            1 if d[3] != 0 else 0,  # add flag for missing age
            (d[4] - avg_sib_sp) / stdev_sib_sp,  # sib_sp
            (d[5] - avg_parch) / stdev_parch,  # parch
            (d[6] - avg_fare) / stdev_fare,  # fare
        ]
        for d in raw_data
    ]
    es = [[d[0]] for d in raw_data]
    nn = Network(
        [
            Layer(7, 10, Re_LU),
            Layer(10, 8, Re_LU),
            Layer(8, 6, Re_LU),
            Layer(6, 1, Sigmoid),
        ]
    )
    # nn.display()
    loss_fn = BCE
    learning_rate = 0.001
    combined = list(zip(xs, es))
    for _ in range(1000):
        for x, e in combined:
            outputs = nn.forward(x)
            output_gradients = loss_fn.get_output_gradients(outputs, e)
            nn.backward(output_gradients)
            nn.zero_grad(learning_rate)
    # nn.display()
    correct_count = 0
    threshold = 0.5
    for x, e in combined:
        [expect] = e
        [output] = nn.forward(x)
        survival_status = 0
        if output > threshold:
            survival_status = 1
        if survival_status == expect:
            correct_count += 1
    correct_rate = correct_count / len(xs)
    print(f"correct_rate: {correct_rate * 100}%")


def task3():
    print("--- Task 3 ---")
    print("--- Task 3-1 ---")
    t = torch.tensor([[2, 3, 1], [5, -2, 1]])
    print(f"{t.shape}, {t.dtype}")
    print("--- Task 3-2 ---")
    t = torch.rand(3, 4, 2)
    print(t.shape)
    print(t)
    print("--- Task 3-3 ---")
    t = torch.ones(2, 1, 5)
    print(t.shape)
    print(t)
    print("--- Task 3-4 ---")
    t1 = torch.tensor([[1, 2, 4], [2, 1, 3]])
    t2 = torch.tensor([[5], [2], [1]])
    t1_t2_matmul = t1.matmul(t2)
    print(t1_t2_matmul.shape)
    print(t1_t2_matmul)
    print("--- Task 3-5 ---")
    t1 = torch.tensor([[1, 2], [2, 3], [-1, 3]])
    t2 = torch.tensor([[5, 4], [2, 1], [1, -5]])
    t1_t2_mul = t1.mul(t2)
    print(t1_t2_mul.shape)
    print(t1_t2_mul)


def main():
    task1()
    task2()
    task3()


if __name__ == "__main__":
    main()
