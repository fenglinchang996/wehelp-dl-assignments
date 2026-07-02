from dataclasses import dataclass
from math import e, log
from collections.abc import Callable


def linear(x: float) -> float:
    return x


def re_lu(x: float) -> float:
    if x > 0:
        return x
    else:
        return 0


def sigmoid(x: float) -> float:
    return 1 / (1 + e ** (-x))


def softmax(o: list[float]) -> list[float]:
    o_max = max(o)
    o_exp = [e ** (x - o_max) for x in o]
    o_exp_sum = sum(o_exp)
    return [x / o_exp_sum for x in o_exp]


def mse(outputs: list[float], expects: list[float]) -> float:
    return sum([(e - o) ** 2 for o, e in zip(outputs, expects)]) / len(outputs)


def binary_cross_entropy(outputs: list[float], expects: list[float]) -> float:
    return -sum([e * log(o) + (1 - e) * log(1 - o) for o, e in zip(outputs, expects)])


def categorical_cross_entropy(outputs: list[float], expects: list[float]) -> float:
    return -sum([e * log(o) for o, e in zip(outputs, expects)])


@dataclass
class Node:
    weight: list[float]
    bias: float = 0


@dataclass
class Layer:
    nodes: list[Node]
    activation: Callable[[list[float]], list[float]]


class Network:
    def __init__(self, layers: list[Layer]) -> None:
        self.layers = layers

    def forward(self, inputs: list[float]) -> list[float]:
        outputList = inputs
        for layer in self.layers:
            rawOutputList = []
            for node in layer.nodes:
                output = (
                    sum(
                        outputList[index] * weight
                        for index, weight in enumerate(node.weight)
                    )
                    + 1 * node.bias
                )
                rawOutputList.append(output)
            outputList = layer.activation(rawOutputList)
        return outputList


def main():
    print("--- Model 1 ---")
    nn_1 = Network(
        [
            Layer(
                [Node([0.5, 0.2], 0.3), Node([0.6, -0.6], 0.25)],
                lambda outputs: [re_lu(x) for x in outputs],
            ),
            Layer(
                [Node([0.8, -0.5], 0.6), Node([0.4, 0.5], -0.25)],
                lambda outputs: [linear(x) for x in outputs],
            ),
        ]
    )
    nn_1_outputs_1 = nn_1.forward([1.5, 0.5])
    nn_1_expects_1 = [0.8, 1]
    print(f"outputs: {nn_1_outputs_1}")
    print(f"Total Loss: {mse(nn_1_outputs_1, nn_1_expects_1)}")
    nn_1_outputs_2 = nn_1.forward([0, 1])
    nn_1_expects_2 = [0.5, 0.5]
    print(f"outputs: {nn_1_outputs_2}")
    print(f"Total Loss: {mse(nn_1_outputs_2, nn_1_expects_2)}")

    print("--- Model 2 ---")
    nn_2 = Network(
        [
            Layer(
                [Node([0.5, 0.2], 0.3), Node([0.6, -0.6], 0.25)],
                lambda outputs: [re_lu(x) for x in outputs],
            ),
            Layer(
                [Node([0.8, 0.4], -0.5)],
                lambda outputs: [sigmoid(x) for x in outputs],
            ),
        ]
    )
    nn_2_outputs_1 = nn_2.forward([0.75, 1.25])
    nn_2_expects_1 = [1.0]
    print(f"outputs: {nn_2_outputs_1}")
    print(f"Total Loss: {binary_cross_entropy(nn_2_outputs_1, nn_2_expects_1)}")
    nn_2_outputs_2 = nn_2.forward([-1, 0.5])
    nn_2_expects_2 = [0.0]
    print(f"outputs: {nn_2_outputs_2}")
    print(f"Total Loss: {binary_cross_entropy(nn_2_outputs_2, nn_2_expects_2)}")

    print("--- Model 3 ---")
    nn_3 = Network(
        [
            Layer(
                [Node([0.5, 0.2], 0.3), Node([0.6, -0.6], 0.25)],
                lambda outputs: [re_lu(x) for x in outputs],
            ),
            Layer(
                [
                    Node([0.8, -0.4], 0.6),
                    Node([0.5, 0.4], 0.5),
                    Node([0.3, 0.75], -0.5),
                ],
                lambda outputs: [sigmoid(x) for x in outputs],
            ),
        ]
    )
    nn_3_outputs_1 = nn_3.forward([1.5, 0.5])
    nn_3_expects_1 = [1.0, 0.0, 1.0]
    print(f"outputs: {nn_3_outputs_1}")
    print(f"Total Loss: {binary_cross_entropy(nn_3_outputs_1, nn_3_expects_1)}")
    nn_3_outputs_2 = nn_3.forward([0, 1])
    nn_3_expects_2 = [1.0, 1.0, 0.0]
    print(f"outputs: {nn_3_outputs_2}")
    print(f"Total Loss: {binary_cross_entropy(nn_3_outputs_2, nn_3_expects_2)}")

    print("--- Model 4 ---")
    nn_4 = Network(
        [
            Layer(
                [Node([0.5, 0.2], 0.3), Node([0.6, -0.6], 0.25)],
                lambda outputs: [re_lu(x) for x in outputs],
            ),
            Layer(
                [
                    Node([0.8, -0.4], 0.6),
                    Node([0.5, 0.4], 0.5),
                    Node([0.3, 0.75], -0.5),
                ],
                softmax,
            ),
        ]
    )
    nn_4_outputs_1 = nn_4.forward([1.5, 0.5])
    nn_4_expects_1 = [1.0, 0.0, 0.0]
    print(f"outputs: {nn_4_outputs_1}")
    print(f"Total Loss: {categorical_cross_entropy(nn_4_outputs_1, nn_4_expects_1)}")
    nn_4_outputs_2 = nn_4.forward([0, 1])
    nn_4_expects_2 = [0.0, 0.0, 1.0]
    print(f"outputs: {nn_4_outputs_2}")
    print(f"Total Loss: {categorical_cross_entropy(nn_4_outputs_2, nn_4_expects_2)}")


if __name__ == "__main__":
    main()
