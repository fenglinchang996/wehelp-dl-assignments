from dataclasses import dataclass, field
from math import e, log
from typing import Protocol


class Activation(Protocol):
    @staticmethod
    def active(x: float) -> float: ...
    @staticmethod
    def derivative(x: float) -> float: ...


class Linear:
    @staticmethod
    def active(x: float) -> float:
        return x

    @staticmethod
    def derivative(x: float) -> float:
        return 1


class Re_LU:
    @staticmethod
    def active(x: float) -> float:
        return x if x > 0 else 0

    @staticmethod
    def derivative(x: float) -> float:
        return 1 if x > 0 else 0


class Sigmoid:
    @staticmethod
    def active(x: float) -> float:
        return 1 / (1 + e ** (-x))

    @staticmethod
    def derivative(x: float) -> float:
        return Sigmoid.active(x) * (1 - Sigmoid.active(x))


class Loss(Protocol):
    @staticmethod
    def get_loss(outputs: list[float], expects: list[float]) -> float: ...
    @staticmethod
    def get_output_gradients(
        outputs: list[float], expects: list[float]
    ) -> list[float]: ...


class MSE:
    @staticmethod
    def get_loss(outputs: list[float], expects: list[float]) -> float:
        return sum([(e - o) ** 2 for o, e in zip(outputs, expects)]) / len(outputs)

    @staticmethod
    def get_output_gradients(outputs: list[float], expects: list[float]) -> list[float]:
        return [2 * (o - e) / len(outputs) for o, e in zip(outputs, expects)]


class BCE:
    @staticmethod
    def get_loss(outputs: list[float], expects: list[float]) -> float:
        return -sum(
            [e * log(o) + (1 - e) * log(1 - o) for o, e in zip(outputs, expects)]
        )

    @staticmethod
    def get_output_gradients(outputs: list[float], expects: list[float]) -> list[float]:
        return [-(e / o) + (1 - e) / (1 - o) for o, e in zip(outputs, expects)]


@dataclass
class Node:
    weights: list[float]
    bias: float
    weight_gradients: list[float] = field(default_factory=list)
    bias_gradient: float = 0


@dataclass
class Layer:
    nodes: list[Node]
    activation: Activation
    inputs: list[float] = field(default_factory=list)
    raw_outputs: list[float] = field(default_factory=list)
    outputs: list[float] = field(default_factory=list)


class Network:
    def __init__(self, layers: list[Layer]) -> None:
        self.layers = layers

    def forward(self, inputs: list[float]) -> list[float]:
        for index, layer in enumerate(self.layers):
            layer.inputs = inputs if index == 0 else self.layers[index - 1].outputs
            layer.raw_outputs = []
            for node in layer.nodes:
                raw_output = (
                    sum(
                        layer.inputs[index] * weight
                        for index, weight in enumerate(node.weights)
                    )
                    + 1 * node.bias
                )
                layer.raw_outputs.append(raw_output)
            layer.outputs = [
                layer.activation.active(raw_output) for raw_output in layer.raw_outputs
            ]
        return self.layers[len(self.layers) - 1].outputs

    def backward(self, output_gradients: list[float]) -> None:
        o_grads: list[float] = output_gradients
        for layer in reversed(self.layers):
            temp_i_grads = [0.0] * len(layer.inputs)
            for node, raw_output, o_grad in zip(
                layer.nodes, layer.raw_outputs, o_grads
            ):
                node.weight_gradients = [0] * len(node.weights)
                raw_o_grad = o_grad * layer.activation.derivative(raw_output)
                for w_idx, (weight, input) in enumerate(
                    zip(node.weights, layer.inputs)
                ):
                    node.weight_gradients[w_idx] = raw_o_grad * input
                    temp_i_grads[w_idx] += raw_o_grad * weight
                node.bias_gradient = raw_o_grad
            o_grads = temp_i_grads

    def zero_grad(self, learning_rate: float) -> None:
        for layer in self.layers:
            for node in layer.nodes:
                for w_idx, weight in enumerate(node.weights):
                    node.weights[w_idx] = (
                        weight - learning_rate * node.weight_gradients[w_idx]
                    )
                node.bias = node.bias - learning_rate * node.bias_gradient

    def display(self) -> None:
        for l_idx, layer in enumerate(self.layers):
            print(f"Layer {l_idx + 1}: ")
            for n_idx, node in enumerate(layer.nodes):
                print(f"Node {n_idx + 1}: weights: {node.weights}, bias: {node.bias}")


def task_1() -> None:
    print("---Task 1---")
    nn_1 = Network(
        [
            Layer([Node([0.5, 0.2], 0.3), Node([0.6, -0.6], 0.25)], Re_LU),
            Layer([Node([0.8, -0.5], 0.6)], Linear),
            Layer([Node([0.6], 0.4), Node([-0.3], 0.75)], Linear),
        ],
    )
    inputs_1 = [1.5, 0.5]
    expects_1 = [0.8, 1]
    loss_fn_1 = MSE
    learning_rate_1 = 0.01
    print("---Task 1-1---")
    outputs_1_1 = nn_1.forward(inputs_1)
    output_gradients_1_1 = loss_fn_1.get_output_gradients(outputs_1_1, expects_1)
    nn_1.backward(output_gradients_1_1)
    nn_1.zero_grad(learning_rate_1)
    nn_1.display()
    print("---Task 1-2---")
    for _ in range(1000):
        outputs_1_2 = nn_1.forward(inputs_1)
        loss_1_2 = loss_fn_1.get_loss(outputs_1_2, expects_1)
        print(f"loss: {loss_1_2}")
        output_gradients_1_2 = loss_fn_1.get_output_gradients(outputs_1_2, expects_1)
        nn_1.backward(output_gradients_1_2)
        nn_1.zero_grad(learning_rate_1)
    outputs_1_2 = nn_1.forward(inputs_1)
    loss_1_2 = loss_fn_1.get_loss(outputs_1_2, expects_1)
    print(f"loss: {loss_1_2}")


def task_2() -> None:
    print("---Task 2---")
    nn_2 = Network(
        [
            Layer([Node([0.5, 0.2], 0.3), Node([0.6, -0.6], 0.25)], Re_LU),
            Layer([Node([0.8, 0.4], -0.5)], Sigmoid),
        ],
    )
    inputs_2 = [0.75, 1.25]
    expects_2 = [1.0]
    loss_fn_2 = BCE
    learning_rate_2 = 0.1
    print("---Task 2-1---")
    outputs_2_1 = nn_2.forward(inputs_2)
    output_gradients_2_1 = loss_fn_2.get_output_gradients(outputs_2_1, expects_2)
    nn_2.backward(output_gradients_2_1)
    nn_2.zero_grad(learning_rate_2)
    nn_2.display()
    print("---Task 2-2---")
    for _ in range(1000):
        outputs_2_2 = nn_2.forward(inputs_2)
        loss_2_2 = loss_fn_2.get_loss(outputs_2_2, expects_2)
        print(f"loss: {loss_2_2}")
        output_gradients_2_2 = loss_fn_2.get_output_gradients(outputs_2_2, expects_2)
        nn_2.backward(output_gradients_2_2)
        nn_2.zero_grad(learning_rate_2)
    outputs_2_2 = nn_2.forward(inputs_2)
    loss_2_2 = loss_fn_2.get_loss(outputs_2_2, expects_2)
    print(f"loss: {loss_2_2}")


def main():
    task_1()
    task_2()


if __name__ == "__main__":
    main()
