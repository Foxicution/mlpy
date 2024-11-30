from math import prod
from typing import Callable, Literal


class Tensor:
    def __init__(self, shape: list[int], data: list[int] | None = None) -> None:
        if data is None:
            assert 0 not in shape, "Shape dims must be positive ints"
            size = prod(shape)
            self.shape = shape
            self.data = [0.0] * size
        else:
            zero_count = shape.count(0)
            assert zero_count <= 1, "Only one dimension can be inferred"
            if zero_count == 1:
                known_product = prod(i for i in shape if i != 0)
                inferred_index = shape.index(0)
                shape[inferred_index] = len(data) // inferred_index
            assert len(data) == prod(shape), "The shape must match data"
            self.shape = shape
            self.data = data

    def _elemwise_with_broadcast(
        self, other: "Tensor", op: Callable[[float, float], float]
    ) -> "Tensor":
        s1, d1, s2, d2 = self.shape, self.data, other.shape, other.data
        len1, len2 = len(self.shape), len(other.shape)
        max_len = max(len1, len2)
        final_shape = []

        stride1, stride2 = 1, 1

        for i in range(max_len):
            dim1 = 1 if i >= len1 else s1[len1 - i - 1]
            dim2 = 1 if i >= len2 else s2[len2 - i - 1]
            assert dim1 == dim2 or dim1 == 1 or dim2 == 1, "Incompatable shapes"

            stride1 *= dim1
            stride2 *= dim2

            if dim1 == 1 and dim2 != 1:
                nd1 = []
                for i in range(0, len(d1), stride1):
                    nd1 += self.data[i : i + stride1] * dim2
                d1 = nd1

            if dim2 == 1 and dim1 != 1:
                nd2 = []
                for i in range(0, len(d2), stride2):
                    nd2 += other.data[i : i + stride2] * dim1
                d2 = nd2

            final_shape.append(max(dim1, dim2))

        return Tensor(final_shape, [op(d1[i], d2[i]) for i in range(len(d1))])

    def __mul__(self, other: "Tensor") -> "Tensor":
        return self._elemwise_with_broadcast(other, lambda x, y: x * y)

    def __div__(self, other: "Tensor") -> "Tensor":
        return self._elemwise_with_broadcast(other, lambda x, y: x / y)

    def __add__(self, other: "Tensor") -> "Tensor":
        return self._elemwise_with_broadcast(other, lambda x, y: x + y)

    def __sub__(self, other: "Tensor") -> "Tensor":
        return self._elemwise_with_broadcast(other, lambda x, y: x - y)

    def matmul(self, other: "Tensor") -> "Tensor":
        assert len(self.shape) == 2, "First tensor is not a 2D matrix"
        assert len(other.shape) == 2, "Second tensor is not a 2D matrix"
        assert self.shape[1] == other.shape[0], "Dimesions did not match for matmul"

        # Allocate the result tensor
        result_shape = (self.shape[0], other.shape[1])
        result = Tensor(*result_shape)

        # Perform the matrix multiplication
        for i in range(result_shape[0]):
            for j in range(result_shape[1]):
                sum = 0
                for k in range(self.shape[1]):
                    sum += (
                        self.data[i * self.shape[1] + k]
                        * other.data[k * other.shape[1] + j]
                    )
                result.data[i * result_shape[1] + j] = sum

        return result

    def matadd(self, other: "Tensor") -> "Tensor":
        assert self.shape == other.shape, "Shapes don't match"
        return Tensor(self.shape, [x + y for x, y in zip(self.data, other.data)])

    def matsub(self, other: "Tensor") -> "Tensor":
        assert self.shape == other.shape, "Shapes don't match"
        return Tensor(self.shape, [x - y for x, y in zip(self.data, other.data)])

    def __str__(self) -> str:
        tensor_info = f"Tensor: {id(self):#x}"
        shape_info = f"Shape: {self.shape}"
        data_info = f"Data: {self.data}"

        max_width = max(len(tensor_info), len(shape_info), len(data_info))

        return f"{tensor_info:<{max_width}}\n{shape_info:<{max_width}}\n{data_info:<{max_width}}"


# Graph optimizations
class Node:
    def __init__(self, operation: Literal["+", "-", "*", "/"], *inputs: "Node") -> None:
        self.operation = operation
        self.inputs = inputs


print(Tensor([1, 3], list(range(3))) * Tensor([3, 1], list(range(3))))
