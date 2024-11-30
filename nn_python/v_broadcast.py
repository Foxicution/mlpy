import math

shape1 = [3, 1]
shape2 = [1, 3]
final_shape = [3, 3]


def stride(original_shape: list[int], final_shape: list[int]) -> list[int]:
    strides = []
    for dim in reversed(original_shape):
        strides.append(
            final_shape[len(final_shape) - len(strides) - 1] // dim if dim != 1 else 0
        )
    return list(reversed(strides))


stride1 = stride(shape1, final_shape)
stride2 = stride(shape2, final_shape)


def multi_dim(idx: int, shape: list[int]) -> list[int]:
    # Convert a flat index to a multi-dimensional index based on the shape
    multi_dim_index = []
    for dim in reversed(shape):
        multi_dim_index.append(idx % dim)
        idx //= dim
    return list(reversed(multi_dim_index))


def multi_dim_to_original(
    multi_dim: list[int], original_shape: list[int], strides: list[int]
) -> int:
    # Map a multi-dimensional index in the final shape back to an index in the original tensor
    original_idx = 0
    for i, stride in enumerate(strides):
        if stride != 0:  # Skip broadcasting dimensions
            original_idx += multi_dim_index[i] * stride
    return original_idx


for i in range(math.prod(final_shape)):
    multi_dim_index = multi_dim(i, final_shape)
    print(multi_dim_index)

    i1 = multi_dim_to_original(multi_dim_index, shape1, stride1)
    i2 = multi_dim_to_original(multi_dim_index, shape2, stride2)
    print(i1, i2)
