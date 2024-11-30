import math
from typing import Union


def _broadcast(shape1: list[int], shape2: list[int]) -> list[int]:
    len1, len2 = len(shape1), len(shape2)
    max_len = max(len1, len2)
    final_shape = []

    for i in range(max_len):
        dim1 = 1 if i >= len1 else shape1[len1 - i - 1]
        dim2 = 1 if i >= len2 else shape2[len2 - i - 1]
        assert dim1 == dim2 or dim1 == 1 or dim2 == 1, "Incompatable shapes"

        final_shape.append(max(dim1, dim2))

    return final_shape


class Node:
    def __init__(
        self, el1: Union["Tensor", "Node"], el2: Union["Tensor", "Node"], op: str
    ) -> None:
        # all of these should be refs, not data copies
        self.el1 = el1
        self.el2 = el2
        self.op = op
        self.shape = _broadcast(el1.shape, el2.shape)

    def __mul__(self, t: Union["Tensor", "Node"]) -> "Node":
        return Node(self, t, "*")

    def __div__(self, t: Union["Tensor", "Node"]) -> "Node":
        return Node(self, t, "/")

    def __add__(self, t: Union["Tensor", "Node"]) -> "Node":
        return Node(self, t, "+")

    def __sub__(self, t: Union["Tensor", "Node"]) -> "Node":
        return Node(self, t, "-")

    def __str__(self, level: int = 0) -> str:
        indent = "|   " * level
        lines = [
            f"{indent}Node: Op: {self.op}, Shape: {self.shape}",
            f"{indent}Element 1:\n{self.el1.__str__(level + 1)}",
            f"{indent}Element 2:\n{self.el2.__str__(level + 1)}",
        ]
        return "\n".join(lines)

    def eval(self) -> "Tensor":
        tensors = []
        stack = [self]

        while stack:
            curr_node = stack.pop()
            if isinstance(curr_node, Tensor):
                ...

        new_tensor = Tensor(self.shape)
        ...


class Tensor:
    def __init__(self, shape: list[int], data: list[float] | None = None):
        if data is None:
            assert 0 not in shape, "Shape dims must be positive ints"
            size = math.prod(shape)
            self.shape = shape
            self.data = [0.0] * size
        else:
            zero_count = shape.count(0)
            assert zero_count <= 1, "Only one dimension can be inferred"
            if zero_count == 1:
                known_product = math.prod(i for i in shape if i != 0)
                inferred_index = shape.index(0)
                shape[inferred_index] = len(data) // inferred_index
            assert len(data) == math.prod(shape), "The shape must match data"
            self.shape = shape
            self.data = data

    def __mul__(self, t: Union["Tensor", Node]) -> Node:
        return Node(self, t, "*")

    def __div__(self, t: Union["Tensor", Node]) -> Node:
        return Node(self, t, "/")

    def __add__(self, t: Union["Tensor", Node]) -> Node:
        return Node(self, t, "+")

    def __sub__(self, t: Union["Tensor", Node]) -> Node:
        return Node(self, t, "-")

    def __str__(self, level: int = 0) -> str:
        indent = "|   " * level
        return f"{indent}Tensor: Shape: {self.shape}, Data: {self.data}"

    def eval(self) -> "Tensor":
        return self


print(
    Tensor([1, 3], [1, 2, 3])
    - Tensor([3, 1], [1, 2, 3])
    + Tensor([3, 3], [1, 2, 3, 1, 2, 3, 1, 2, 3])
)
