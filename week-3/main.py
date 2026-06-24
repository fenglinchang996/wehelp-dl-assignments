from dataclasses import dataclass


@dataclass
class Node:
    weight: list[float]
    bias: float = 0


class Network:
    def __init__(self, layers: list[list[Node]]) -> None:
        self.layers = layers

    def forward(self, inputs: list[float]) -> list[float]:
        outputList = inputs
        for layer in self.layers:
            newOutputList = []
            for node in layer:
                output = (
                    sum(
                        outputList[index] * weight
                        for index, weight in enumerate(node.weight)
                    )
                    + 1 * node.bias
                )
                newOutputList.append(output)
            outputList = newOutputList
        return outputList


def main():
    print("Neural Network 1:")
    nn1 = Network(
        [[Node([0.5, 0.2], 0.3), Node([0.6, -0.6], 0.25)], [Node([0.8, 0.4], -0.5)]]
    )
    nn1_outputs1 = nn1.forward([1.5, 0.5])
    nn1_outputs2 = nn1.forward([0, 1])
    print(nn1_outputs1)
    print(nn1_outputs2)
    print("Neural Network 2:")
    nn2 = Network(
        [
            [Node([0.5, 1.5], 0.3), Node([0.6, -0.8], 1.25)],
            [Node([0.6, -0.8], 0.3)],
            [Node([0.5], 0.2), Node([-0.4], 0.5)],
        ]
    )
    nn2_outputs1 = nn2.forward([0.75, 1.25])
    nn2_outputs2 = nn2.forward([-1, 0.5])
    print(nn2_outputs1)
    print(nn2_outputs2)


if __name__ == "__main__":
    main()
